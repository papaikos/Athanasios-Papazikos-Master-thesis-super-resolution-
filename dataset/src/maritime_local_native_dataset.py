"""Offline native Mother patches using exactly the cloud loader's pixel mapping.

All source files are downloaded byte originals. No cloud SDK, credentials,
network transport, padding or source resampling is performed by this loader.
"""
import json
from pathlib import Path
import threading

from src.maritime_native_dataset import NativeMaritimeDataset
from scripts.maritime_uint8_download import native_references,validate_manifest,validate_pin


def native_filename(reference):
    """Match the existing byte downloader's content-addressed suffix policy."""
    validate_pin(reference)
    suffix=Path(reference['uri']).suffix.lower()
    suffix={'.tiff':'.tif','.jpeg':'.jpg'}.get(suffix,suffix)
    if suffix not in ('.tif','.tiff','.png','.jpg','.jpeg','.jp2','.zip','.npz'):
        suffix='.bin'
    return reference['sha256']+suffix


def file_identity(path):
    stat=path.stat()
    return stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns


class LocalNativeStore:
    """Checksum once per unchanged local file; fail closed if it changes."""
    def __init__(self,objects):
        self.objects=Path(objects).resolve()
        self.verified={};self.identities={};self.lock=threading.Lock()

    def cached_source(self,reference):
        from scripts.maritime_uint8_download import _checksums
        filename=native_filename(reference)
        path=self.objects/filename
        expected=(reference['bytes'],reference['crc32c'],reference['sha256'])
        with self.lock:
            if filename in self.identities and self.identities[filename]!=expected:
                raise ValueError('Conflicting pinned identity for local native file')
            self.identities[filename]=expected
            if not path.is_file():raise FileNotFoundError('Missing downloaded native object: '+str(path))
            if not path.resolve().is_relative_to(self.objects):
                raise ValueError('Native object escapes the downloaded objects directory')
            before=file_identity(path)
            if filename in self.verified:
                if self.verified[filename]!=before:
                    raise ValueError('Verified local native file changed; preserve and investigate: '+str(path))
                return path
            sha,crc=_checksums(path)
            after=file_identity(path)
            if before!=after or before[2]!=reference['bytes'] or sha!=reference['sha256'] or crc!=reference['crc32c']:
                raise ValueError('Local native size/SHA/CRC verification failed: '+str(path))
            self.verified[filename]=after
            return path

    def trim_cache(self):
        # Files belong to the downloaded release; this loader never evicts them.
        pass

    def __getstate__(self):
        state=self.__dict__.copy();state.pop('lock')
        state['verified']={}  # Spawned workers verify their own first access.
        return state

    def __setstate__(self,state):
        self.__dict__.update(state);self.lock=threading.Lock()


class LocalNativeMaritimeDataset(NativeMaritimeDataset):
    """Use a downloaded Mother/manifest.json with its native/objects directory.

Inherited patch geometry, support checks, epoch crops, EarthView saved mapping
and tensor conversion are identical to NativeMaritimeDataset. READY is required
unless an explicit engineering-only format override is supplied.
"""
    def __init__(self,manifest,split='train',patch_size=512,seed=20261006,
                 allow_format_only=False):
        path=Path(manifest)
        self.manifest=json.loads(path.read_text(encoding='utf-8-sig'))
        validate_manifest(self.manifest,'mother')
        if self.manifest.get('status')!='READY' and not allow_format_only:
            raise ValueError('READY Mother manifest required; allow_format_only is engineering-only')
        if not isinstance(allow_format_only,bool):raise ValueError('Boolean format-only override required')
        if not isinstance(patch_size,int) or isinstance(patch_size,bool) or patch_size<1:
            raise ValueError('Positive integer patch_size required')
        if not isinstance(seed,int) or isinstance(seed,bool):raise ValueError('Integer seed required')
        self.samples=[sample for sample in self.manifest['samples'] if sample['split']==split]
        if not self.samples:raise ValueError('Empty native mother split')
        if any(sample.get('source_kind') not in ('skysat','skysat_raw_window','maxar','earthview','aerial_rgb') for sample in self.samples):
            raise ValueError('Unsupported native Mother source kind; explicit source adapter required')
        self.store=LocalNativeStore(path.parent/'native'/'objects')
        for sample in self.samples:
            for reference in native_references(sample):
                filename=native_filename(reference)
                identity=(reference['bytes'],reference['crc32c'],reference['sha256'])
                previous=self.store.identities.setdefault(filename,identity)
                if previous!=identity:raise ValueError('Conflicting pinned identity for local native file')
        self.patch_size=patch_size;self.seed=seed;self.epoch=0
