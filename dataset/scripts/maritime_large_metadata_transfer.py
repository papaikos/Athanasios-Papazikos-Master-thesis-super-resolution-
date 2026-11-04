"""SDK staging of compressed JSON, followed by byte-identical GCE publication."""
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import uuid

from scripts.maritime_cloud_inventory import json_bytes,verify_output
from scripts.maritime_metadata_publish_gce import file_identity
from src.cloud_config import GCP_PROJECT, GCS_BUCKET

GCLOUD='C:/Program Files (x86)/Google/Cloud SDK/google-cloud-sdk/bin/gcloud.cmd'
VM='maritime-20261006-release-v1'
BUCKET=GCS_BUCKET
PREFIX='maritime-rgb/expansion-20261007/control/large-metadata-transfers-v1/'


def short_scp_inputs(files,folder):
    """Avoid Windows pscp's long relative path handling with short byte copies."""
    staged=[]
    for index,(file,remote) in enumerate(files):
        local=Path(folder)/('input-'+str(index)+Path(file).suffix)
        shutil.copyfile(file,local)
        if file_identity(local)!=file_identity(file):
            raise ValueError('SCP metadata staging identity mismatch')
        staged.append((local,remote))
    return staged


def publish_large_metadata(client,path,output_uri,folder):
    path=Path(path);folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    expected=file_identity(path);identity=expected['sha256']
    compressed=folder/(identity+'.json.gz')
    with path.open('rb') as source,compressed.open('wb') as destination:
        with gzip.GzipFile(filename='',fileobj=destination,mode='wb',mtime=0,compresslevel=1) as stream:
            shutil.copyfileobj(source,stream,8*1024**2)
    def stage(file,suffix):
        known=file_identity(file);name=PREFIX+known['sha256']+suffix;blob=client.bucket(BUCKET).blob(name)
        blob.chunk_size=8*1024**2
        if not blob.exists():blob.upload_from_filename(str(file),if_generation_match=0,checksum='crc32c',timeout=(60,300))
        blob.reload();pin=dict(uri='gs://'+BUCKET+'/'+name,generation=str(blob.generation),**known)
        assert verify_output(client,pin)==file.read_bytes()
        return pin
    wire_pin=stage(compressed,'.json.gz')
    request=dict(schema='maritime-large-metadata-publication-request-v1',compressed_wire_pin=wire_pin,
        plain_identity=expected,output_uri=output_uri)
    request_path=folder/(identity+'-request.json');request_path.write_bytes(json_bytes(request))
    request_pin=stage(request_path,'.request.json')
    pin_path=folder/(identity+'-request.pin.json');pin_path.write_bytes(json_bytes(request_pin))
    code=Path('scripts/maritime_metadata_publish_gce.py');code_sha=hashlib.sha256(code.read_bytes()).hexdigest()
    attempt=identity[:16]+'-'+uuid.uuid4().hex[:8]
    remote_code='/home/user/metadata-publisher-'+code_sha+'.py'
    remote_pin='/home/user/metadata-request-'+attempt+'.pin.json'
    remote_script='/home/user/metadata-run-'+attempt+'.sh'
    script=folder/(attempt+'-run.sh')
    script.write_text('''#!/bin/bash
set -euo pipefail
BASE=/mnt/maritime/expansion-20261007/large-metadata-transfers-v1/ATTEMPT
CODE=/mnt/maritime/expansion-20261007/expanded-current-runtime-v2/code
mkdir -p "$BASE/logs" "$BASE/scratch"
cp REMOTE_PIN "$BASE/request.pin.json"
cp REMOTE_CODE "$BASE/publisher.py"
ACTIVE=$(docker ps -q)
: > "$BASE/logs/resource-caps.txt"
if [ -n "$ACTIVE" ]; then docker inspect $ACTIVE --format '{{.Name}} CPU={{.HostConfig.NanoCpus}} MEM={{.HostConfig.Memory}}' > "$BASE/logs/resource-caps.txt"; fi
python3 -c 'import sys;from pathlib import Path;r=Path(sys.argv[1]).read_text().splitlines();assert sum(int(x.split("CPU=")[1].split()[0]) for x in r)+1000000000<=14000000000;assert sum(int(x.split("MEM=")[1]) for x in r)+2*1024**3<=56*1024**3' "$BASE/logs/resource-caps.txt"
set +e
docker run --name maritime-metadata-publish-ATTEMPT --network host --read-only --cpus 1 --memory 2g --tmpfs /tmp:rw,size=128m -e PYTHONPATH="$CODE" -e OPENBLAS_NUM_THREADS=1 -v "$CODE:$CODE:ro" -v "$BASE:$BASE:rw" -w "$CODE" sha256:13ed89722a5572eb3f0ed3176c49e0e84dcae7f13b0f40bfaca06ccc3d008995 python -u "$BASE/publisher.py" --request-pin "$BASE/request.pin.json" --scratch "$BASE/scratch" --code-sha CODE_SHA > "$BASE/logs/publication.log" 2>&1
status=$?
set -e
docker inspect maritime-metadata-publish-ATTEMPT > "$BASE/logs/container-inspect.json"
echo "$status" > "$BASE/logs/exit-code.txt"
cat "$BASE/logs/publication.log"
exit "$status"
'''.replace('ATTEMPT',attempt).replace('REMOTE_PIN',remote_pin).replace('REMOTE_CODE',remote_code).replace('CODE_SHA',code_sha),encoding='utf-8',newline='\n')
    common=['--project='+GCP_PROJECT,'--zone=europe-west4-a','--quiet']
    def command(arguments):
        result=subprocess.run([GCLOUD,'compute',*arguments,*common],capture_output=True,text=True)
        if result.returncode:
            raise RuntimeError('GCE metadata command failed: '+result.stderr[-4000:])
        return result.stdout
    with tempfile.TemporaryDirectory(prefix='marine-scp-') as short_folder:
        for file,remote in short_scp_inputs([(code,remote_code),(pin_path,remote_pin),(script,remote_script)],short_folder):
            command(['scp',str(file),VM+':'+remote])
    result=command(['ssh',VM,'--command=sudo /bin/bash '+remote_script])
    (folder/(attempt+'-publication.log')).write_text(result,encoding='utf-8')
    document=json.loads(result.strip().splitlines()[-1]);pin=document['output_pin']
    assert pin['uri']==output_uri and {k:pin[k] for k in expected}==expected
    assert document['full_expanded_and_gcs_readback_sha_crc_verified'] is True
    # The complete SHA/CRC readback occurred on GCE; root separately verifies
    # the exact returned generation/size/CRC without downloading large JSON again.
    blob=client.bucket(BUCKET).blob(output_uri.split('/'+BUCKET+'/',1)[1],generation=int(pin['generation']))
    blob.reload()
    assert str(blob.generation)==pin['generation'] and int(blob.size)==pin['bytes'] and blob.crc32c==pin['crc32c']
    (folder/(attempt+'-result.json')).write_bytes(json_bytes(document))
    return pin
