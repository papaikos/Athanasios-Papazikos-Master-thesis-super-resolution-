"""Atomic local state and separately readback-verified GCS publication."""
import hashlib
import json
import os
from pathlib import Path
import tempfile

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    os.replace(temporary,path)

def atomic_checkpoint(state,directory,step):
    import torch
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    fd,temp=tempfile.mkstemp(dir=directory,suffix='.tmp');os.close(fd)
    try:
        with open(temp,'wb') as f:torch.save(state,f);f.flush();os.fsync(f.fileno())
        digest=sha256(temp);path=directory/f'step-{step:09d}-{digest}.pt';os.replace(temp,path)
    finally:
        if os.path.exists(temp):os.unlink(temp)
    receipt={'sha256':digest,'bytes':path.stat().st_size,'step':step,'filename':path.name}
    load_checkpoint(path,receipt)
    return path,receipt

def load_checkpoint(path,receipt):
    import torch
    if Path(path).stat().st_size!=receipt['bytes'] or sha256(path)!=receipt['sha256']:raise ValueError('Checkpoint integrity mismatch')
    value=torch.load(path,map_location='cpu',weights_only=False)
    if value.get('schema')!=1:raise ValueError('Checkpoint schema mismatch')
    return value

class GCSStore:
    def __init__(self,uri,client=None):
        from google.cloud import storage
        if not uri.startswith('gs://'):raise ValueError('GCS output required')
        bucket,_,self.prefix=uri[5:].partition('/');self.bucket=(client or storage.Client()).bucket(bucket)
    def upload_verified(self,path,relative):
        path=Path(path);digest=sha256(path);blob=self.bucket.blob(self.prefix.rstrip('/')+'/'+relative)
        from google.api_core.exceptions import PreconditionFailed
        blob.metadata={'sha256':digest}
        try:blob.upload_from_filename(str(path),if_generation_match=0,checksum='crc32c')
        except PreconditionFailed:pass
        blob.reload()
        fd,temp=tempfile.mkstemp();os.close(fd)
        try:
            blob.download_to_filename(temp,if_generation_match=int(blob.generation),checksum='crc32c')
            if sha256(temp)!=digest or Path(temp).stat().st_size!=path.stat().st_size:raise ValueError('Remote readback mismatch')
        finally:os.unlink(temp)
        return {'uri':f'gs://{self.bucket.name}/{blob.name}','generation':str(blob.generation),'sha256':digest,'bytes':path.stat().st_size}
    def download_verified(self,relative,destination,receipt):
        blob=self.bucket.blob(self.prefix.rstrip('/')+'/'+relative)
        destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
        fd,temporary=tempfile.mkstemp(dir=destination.parent,suffix='.tmp');os.close(fd)
        try:
            blob.download_to_filename(temporary,if_generation_match=int(receipt['generation']),checksum='crc32c')
            if sha256(temporary)!=receipt['sha256'] or Path(temporary).stat().st_size!=receipt['bytes']:raise ValueError('Remote download integrity mismatch')
            os.replace(temporary,destination)
        finally:
            if os.path.exists(temporary):os.unlink(temporary)
        return receipt

    def pointer(self,relative,value):
        blob=self.bucket.blob(self.prefix.rstrip('/')+'/'+relative)
        try:blob.reload();generation=int(blob.generation)
        except Exception as error:
            from google.api_core.exceptions import NotFound
            if not isinstance(error,NotFound):raise
            generation=0
        # reload() retains the old object's crc32c. A fresh Blob prevents that
        # checksum from being sent as the expected checksum for replacement bytes.
        blob=self.bucket.blob(blob.name)
        blob.upload_from_string(json.dumps(value,allow_nan=False),content_type='application/json',if_generation_match=generation,checksum='crc32c')

    def sync_file(self,path,relative):
        """Mutable logs/status use conditional replacement; never JSON-wrap JSONL."""
        blob=self.bucket.blob(self.prefix.rstrip('/')+'/'+relative)
        from google.api_core.exceptions import NotFound
        try:blob.reload();generation=int(blob.generation)
        except NotFound:generation=0
        blob=self.bucket.blob(blob.name)
        with tempfile.TemporaryDirectory() as directory:
            snapshot=Path(directory)/'snapshot';readback=Path(directory)/'readback'
            # Copy a bounded prefix: appends may continue while the upload runs,
            # but neither its size nor its contents can change underneath the SDK.
            with open(path,'rb') as source,snapshot.open('wb') as target:
                remaining=os.fstat(source.fileno()).st_size
                while remaining:
                    chunk=source.read(min(8*1024*1024,remaining))
                    if not chunk:raise ValueError('Source truncated while taking upload snapshot')
                    target.write(chunk);remaining-=len(chunk)
            digest=sha256(snapshot);size=snapshot.stat().st_size
            blob.upload_from_filename(str(snapshot),if_generation_match=generation,checksum='crc32c')
            verified_generation=int(blob.generation)
            blob.download_to_filename(str(readback),if_generation_match=verified_generation,checksum='crc32c')
            if sha256(readback)!=digest or readback.stat().st_size!=size:raise ValueError('Mutable object readback mismatch')
            return {'uri':f'gs://{self.bucket.name}/{blob.name}','generation':str(verified_generation),'sha256':digest,'bytes':size}
