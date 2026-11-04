from scripts.maritime_large_metadata_transfer import short_scp_inputs


def test_scp_staging_retains_bytes_and_remote_identity(tmp_path):
    source=tmp_path/'deep-metadata';source.mkdir()
    files=[]
    for i in range(3):
        path=source/(str(i)+'-original-name.json');path.write_bytes(bytes([i])*257)
        files.append((path,'/home/user/immutable-'+str(i)+'.json'))
    stage=tmp_path/'short';stage.mkdir()
    staged=short_scp_inputs(files,stage)
    assert [remote for _,remote in staged]==[remote for _,remote in files]
    assert all(a.read_bytes()==b.read_bytes() for (a,_),(b,_) in zip(staged,files))
    assert all(path.parent==stage and path.name.startswith('input-') for path,_ in staged)
