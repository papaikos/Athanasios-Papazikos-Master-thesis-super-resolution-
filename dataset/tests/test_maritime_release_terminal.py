import copy
from contextlib import nullcontext
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import maritime_release_terminal as terminal

IMAGE = 'maritime@sha256:'+'a'*64
PREFIX = 'gs://release/run'


class Blob:
    def __init__(self, client, name, generation=None):
        self.client=client; self.name=name; self.generation=generation
    def exists(self): return self.name in self.client.objects
    def upload_from_string(self,payload,if_generation_match,checksum):
        assert if_generation_match == 0 and checksum == 'crc32c'
        assert not self.exists()
        self.client.objects[self.name] = (payload,str(len(self.client.objects)+1),'test-crc')
    def reload(self):
        payload,generation,crc = self.client.objects[self.name]
        if self.generation is not None: assert str(self.generation)==generation
        self.generation=generation; self.size=len(payload); self.crc32c=crc
    def download_as_bytes(self,if_generation_match,checksum):
        payload,generation,_ = self.client.objects[self.name]
        assert int(generation)==if_generation_match and checksum=='crc32c'
        return payload+b'corrupt' if self.name==self.client.corrupt else payload


class Client:
    def __init__(self): self.objects={}; self.corrupt=None
    def bucket(self,name):
        assert name=='release'
        return SimpleNamespace(blob=lambda path,generation=None: Blob(self,path,generation))
    def list_blobs(self,bucket,prefix):
        return [SimpleNamespace(name=n) for n in self.objects if n.startswith(prefix)]


@pytest.fixture(autouse=True)
def local_fixture_lock(monkeypatch):
    monkeypatch.setattr(terminal,'evidence_lock',lambda path:nullcontext())


def attempt(tmp_path,client,phase='mother',code=0):
    selection=tmp_path/'selection.json'; selection.write_text('{"selection":[]}')
    state=terminal.begin(tmp_path,phase,selection,PREFIX,IMAGE)
    start=dict(phase=phase,job_attempt_id=state['job_attempt_id'],instance_id='12345',
               container_image_id=IMAGE,config_sha256='b'*64,producer='release_worker',
               process_terminal_verified=False)
    ref=terminal.put(client,PREFIX,'control/attempts/'+phase+'/'+state['job_attempt_id']+'-started.json',
                     terminal.json_bytes(start))
    terminal.atomic_json(tmp_path/(phase+'-attempt.json'),dict(start,start_record=ref))
    Path(state['log']).write_bytes(b'closed worker output\n')
    inspect=tmp_path/(phase+'-inspect.json')
    terminal.atomic_json(inspect,[dict(Name='/'+state['container_name'],Image='sha256:'+'a'*64,
                      State=dict(Running=False,Status='exited',ExitCode=code,FinishedAt='2026-10-06T12:00:00Z'))])
    return state,inspect,selection


@pytest.mark.parametrize('code',[0,1,137])
def test_success_and_failed_exit_receipts(tmp_path,code):
    client=Client(); state,inspect,_=attempt(tmp_path,client,code=code)
    result=terminal.publish(client,tmp_path,'mother',PREFIX,inspect,code,'12345')
    record=json.loads(terminal.pin_read(client,result['terminal_receipt']))
    assert record['state']=='EXITED' and record['exit_code']==code
    assert record['worker_process_exited'] is True and record['logs_closed'] is True
    assert record['producer']=='sdk_supervisor' and record['job_attempt_id']==state['job_attempt_id']
    assert terminal.pin_read(client,record['closed_log'])==b'closed worker output\n'
    evidence=json.loads((tmp_path/'terminal-evidence.json').read_text())
    assert evidence=={'receipts':[result['terminal_receipt']]}


def test_all_attempts_preserved_retry_limit_and_resume(tmp_path):
    client=Client()
    for number in range(1,4):
        state,inspect,selection=attempt(tmp_path,client,code=1)
        assert state['attempt_number']==number
        assert terminal.begin(tmp_path,'mother',selection,PREFIX,IMAGE)==state
        terminal.publish(client,tmp_path,'mother',PREFIX,inspect,1,'12345')
    assert len(json.loads((tmp_path/'terminal-evidence.json').read_text())['receipts'])==3
    with pytest.raises(ValueError,match='limit'): terminal.begin(tmp_path,'mother',selection,PREFIX,IMAGE)


def test_success_resume_idempotent_and_configuration_locked(tmp_path):
    client=Client(); state,inspect,selection=attempt(tmp_path,client)
    done=terminal.publish(client,tmp_path,'mother',PREFIX,inspect,0,'12345')
    assert terminal.begin(tmp_path,'mother',selection,PREFIX,IMAGE)==done
    assert terminal.publish(client,tmp_path,'mother',PREFIX,inspect,0,'12345')==done
    assert len(json.loads((tmp_path/'terminal-evidence.json').read_text())['receipts'])==1
    selection.write_text('changed')
    with pytest.raises(ValueError,match='configuration'): terminal.begin(tmp_path,'mother',selection,PREFIX,IMAGE)


@pytest.mark.parametrize('field,value',[('Running',True),('Status','running'),('ExitCode',True)])
def test_uncertain_docker_state_never_published(tmp_path,field,value):
    client=Client(); state,inspect,_=attempt(tmp_path,client)
    doc=json.loads(inspect.read_text()); doc[0]['State'][field]=value; terminal.atomic_json(inspect,doc)
    with pytest.raises(ValueError,match='Docker'): terminal.publish(client,tmp_path,'mother',PREFIX,inspect,0,'12345')
    persisted=json.loads((tmp_path/'mother-supervisor.json').read_text())
    assert persisted['worker_process_exited'] is False and persisted['logs_closed'] is False
    assert persisted['published'] is False and not (tmp_path/'terminal-evidence.json').exists()


def test_image_checkpoint_start_and_instance_mismatch_fail_closed(tmp_path):
    client=Client(); state,inspect,_=attempt(tmp_path,client)
    with pytest.raises(ValueError,match='checkpoint'): terminal.publish(client,tmp_path,'mother',PREFIX,inspect,0,'999')
    doc=json.loads(inspect.read_text()); doc[0]['Image']='sha256:'+'c'*64; terminal.atomic_json(inspect,doc)
    with pytest.raises(ValueError,match='Docker'): terminal.publish(client,tmp_path,'mother',PREFIX,inspect,0,'12345')
    assert json.loads((tmp_path/'mother-supervisor.json').read_text())['published'] is False


def test_readback_failure_no_receipt_then_resume(tmp_path):
    client=Client(); state,inspect,selection=attempt(tmp_path,client)
    client.corrupt='run/control/supervisor/mother/'+state['job_attempt_id']+'.log'
    with pytest.raises(ValueError,match='Immutable'): terminal.publish(client,tmp_path,'mother',PREFIX,inspect,0,'12345')
    assert not (tmp_path/'terminal-evidence.json').exists()
    assert terminal.begin(tmp_path,'mother',selection,PREFIX,IMAGE)==state
    client.corrupt=None
    assert terminal.publish(client,tmp_path,'mother',PREFIX,inspect,0,'12345')['published']


def test_artifact_completion_is_not_a_worker_start_or_exit(tmp_path):
    client=Client(); state,inspect,_=attempt(tmp_path,client)
    artifact=terminal.put(client,PREFIX,'control/phases/mother-complete.json',
        terminal.json_bytes(dict(state='COMPLETE',phase='mother',config_sha256='b'*64,
                                 process_terminal_verified=False)))
    checkpoint=json.loads((tmp_path/'mother-attempt.json').read_text())
    checkpoint['start_record']=artifact
    terminal.atomic_json(tmp_path/'mother-attempt.json',checkpoint)
    with pytest.raises(ValueError,match='start mismatch'):
        terminal.publish(client,tmp_path,'mother',PREFIX,inspect,0,'12345')
    assert not (tmp_path/'terminal-evidence.json').exists()
    assert json.loads((tmp_path/'mother-supervisor.json').read_text())['published'] is False


def test_builder_terminal_contract_all_four_and_failed_prior(tmp_path):
    build=pytest.importorskip('scripts.maritime_release_build')
    client=Client()
    for phase,code in [('mother',1),('mother',0),('d1',0),('d2',0),('d3',0)]:
        _,inspect,_=attempt(tmp_path,client,phase,code)
        terminal.publish(client,tmp_path,phase,PREFIX,inspect,code,'12345')
    evidence=json.loads((tmp_path/'terminal-evidence.json').read_text())
    result=build.verify_terminal_evidence(build.Store(client,PREFIX),evidence,'b'*64,IMAGE)
    assert result['verified'] is True and len(result['terminal_receipts'])==5
    incomplete=copy.deepcopy(evidence); incomplete['receipts'].pop(0)
    with pytest.raises(ValueError,match='All started'): build.verify_terminal_evidence(build.Store(client,PREFIX),incomplete,'b'*64,IMAGE)


def test_finalizer_log_separate_from_worker_evidence(tmp_path):
    client=Client(); _,inspect,_=attempt(tmp_path,client,'finalize')
    result=terminal.publish(client,tmp_path,'finalize',PREFIX,inspect,0,'12345')
    assert result['published'] and not (tmp_path/'terminal-evidence.json').exists()


def test_shell_uses_same_provisional_pilot_and_finalizer_evidence():
    shell=Path('scripts/maritime_release_supervise.sh').read_text()
    assert shell.count('--provisional --pilot')==1
    assert '--terminal-evidence /work/terminal-evidence.json' in shell
    assert 'flock -n 9' in shell and 'docker wait "$name"' in shell
    assert 'docker logs "$name"' in shell and 'docker run --rm --network host --read-only --tmpfs /tmp' in shell
    assert 'docker run -d --name "$name"' in shell
