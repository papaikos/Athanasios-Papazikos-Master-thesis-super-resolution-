"""GCE-only immutable, model-predicted maritime candidate triage.

Inference images are temporary derivatives; native pixels and dataset recipes
remain unchanged. A triage report is neither semantic acceptance nor READY.
"""
import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import time
from src.cloud_config import GCP_PROJECT

import numpy as np

from src.maritime_quality import quality_metrics, rank_scene, summarize_ship_detections


ULTRALYTICS_VERSION = '8.3.0'
WEIGHTS_BYTES = 23267238
WEIGHTS_URL = 'https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s-obb.pt'


def verify_weights(path, expected_sha256):
    """Verify explicit local-VM weights; never auto-download a model."""
    if (not isinstance(expected_sha256, str) or len(expected_sha256) != 64
            or any(c not in '0123456789abcdef' for c in expected_sha256)):
        raise ValueError('Explicit lowercase weights SHA-256 required')
    path = Path(path)
    if path.stat().st_size != WEIGHTS_BYTES:
        raise ValueError('Unexpected official YOLOv8s-OBB weights size')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    if digest.hexdigest() != expected_sha256:
        raise ValueError('Weights SHA-256 mismatch')
    return dict(bytes=WEIGHTS_BYTES, sha256=digest.hexdigest(), origin_url=WEIGHTS_URL)


def candidate_rows(document, limit, source_order='identity'):
    """Normalize pinned candidates without changing existing Maxar identities."""
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError('Positive candidate limit required')
    if source_order not in ('identity','original-object'):
        raise ValueError('Unsupported source order')
    rows = document.get('candidates', document.get('selection'))
    if not isinstance(rows, list) or not rows:
        raise ValueError('Nonempty candidate manifest required')
    selected, seen = [], set()
    for entry in rows:
        kind = entry.get('source_kind', 'maxar')
        if kind not in ('maxar', 'skysat', 'skysat_raw_window', 'earthview'):
            raise ValueError('Unsupported source kind')
        row = dict(entry['source']) if entry.get('source_kind') and 'source' in entry and kind != 'earthview' else dict(entry)
        # EarthView candidate rows carry source provenance themselves; selections
        # wrap the whole candidate under source.
        if kind == 'earthview' and 'source' in entry and entry.get('schema') != 'earthview-candidate-v1':
            row = dict(entry['source'])
        identity = entry.get('sample_id')
        if row.get('sample_id', identity) != identity:
            raise ValueError('Candidate/selection identity mismatch')
        if kind == 'maxar':
            if row.get('schema') != 'maxar-mixed-native-v1':
                raise ValueError('Pinned Maxar candidate schema required')
            keys = ('native_context', 'hr')
        elif kind == 'earthview':
            if row.get('schema') != 'earthview-candidate-v1':
                raise ValueError('Pinned EarthView candidate schema required')
            keys = ('native_context', 'hr')
        else:
            keys = ('source_object',) if kind == 'skysat_raw_window' else ('mother',)
            if not isinstance(row.get('mask'), dict):
                raise ValueError('SkySat original support provenance required')
            if row['mask'].get('object'):
                keys += ('mask',)
        if not isinstance(identity, str) or not identity:
            raise ValueError('Nonempty candidate sample ID required')
        if identity in seen:
            raise ValueError('Duplicate candidate sample ID')
        seen.add(identity)
        for key in keys:
            ref = row[key]['object'] if key == 'mask' else row[key]
            if not isinstance(ref, dict) or not {'uri','generation','bytes','crc32c','sha256'}.issubset(ref):
                raise ValueError('Complete immutable source pin required')
        if kind != 'maxar':
            row.update(sample_id=identity, _triage_source_kind=kind)
        selected.append(row)
    def order(row):
        if source_order == 'original-object' and row.get('_triage_source_kind') in ('skysat','skysat_raw_window'):
            ref=row.get('source_object',row.get('mother'))
            return ('original',ref['uri'],str(ref['generation']),row['sample_id'])
        return ('materialized',row['sample_id'])
    return sorted(selected, key=order)[:limit]


def candidate_task_groups(rows, grouping='candidate', source_order='identity'):
    """Keep one original object's windows in one isolated worker task."""
    if grouping not in ('candidate','original-object'):
        raise ValueError('Unsupported task grouping')
    if grouping == 'candidate':return [[row] for row in rows]
    if source_order != 'original-object':
        raise ValueError('Original-object tasks require original-object source order')
    groups=[];previous=None;pin=None
    for row in rows:
        if row.get('_triage_source_kind') in ('skysat','skysat_raw_window'):
            reference=row.get('source_object',row.get('mother'))
            key=(reference['uri'],str(reference['generation']))
            frozen={field:reference[field] for field in ('uri','generation','sha256','bytes','crc32c')}
        else:key=None;frozen=None
        if key is not None and key == previous:
            if frozen != pin:raise ValueError('Conflicting original-object pins')
            groups[-1].append(row)
        else:groups.append([row])
        previous=key;pin=frozen
    return groups


def process_candidate_group(task):
    rows,configuration_sha=task
    return [process_candidate((row,configuration_sha)) for row in rows]


def validate_workers(workers):
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= 8:
        raise ValueError('workers must be an integer from 1 to 8')
    return workers


def validate_cpu_threads(threads):
    if isinstance(threads,bool) or not isinstance(threads,int) or not 1 <= threads <= 8:
        raise ValueError('CPU threads must be an integer from 1 to 8')
    return threads


def validate_device(device,workers):
    if device not in ('cpu','cuda:0'):
        raise ValueError('Explicit device must be cpu or cuda:0; no automatic fallback')
    if device != 'cpu' and workers != 1:
        raise ValueError('GPU triage requires one isolated model worker')
    return device


def tile_origins(length, tile_size=1024, overlap=256):
    """Cover the edge exactly; final tile may overlap more than nominal."""
    if (not isinstance(length, int) or length < 1 or not isinstance(tile_size, int)
            or tile_size < 1 or not isinstance(overlap, int) or not 0 <= overlap < tile_size):
        raise ValueError('Invalid tile geometry')
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size + 1, tile_size - overlap))
    if starts[-1] != length - tile_size:
        starts.append(length - tile_size)
    return starts


def infer_tiles(bgr, predictor, *, tile_size=1024, overlap=256):
    """Offset tile-local oriented polygons into the full inference grid."""
    if not isinstance(bgr, np.ndarray) or bgr.ndim != 3 or bgr.shape[2] != 3 or bgr.dtype != np.uint8:
        raise ValueError('Detector expects HWC uint8 BGR')
    records, windows, timings = [], [], []
    for y in tile_origins(bgr.shape[0], tile_size, overlap):
        for x in tile_origins(bgr.shape[1], tile_size, overlap):
            tile = np.ascontiguousarray(bgr[y:y+tile_size, x:x+tile_size])
            started = time.perf_counter()
            predictions = predictor(tile)
            timings.append(time.perf_counter() - started)
            windows.append([x, y, tile.shape[1], tile.shape[0]])
            for item in predictions:
                polygon = np.asarray(item['polygon'], np.float64)
                if polygon.shape != (4, 2) or not np.isfinite(polygon).all():
                    raise ValueError('Invalid tile detector quadrilateral')
                records.append(dict(class_id=item['class_id'], confidence=item['confidence'],
                                    polygon=(polygon + [x, y]).tolist()))
    return records, dict(tile_count=len(windows), tile_windows=windows,
        tile_inference_seconds=timings, inference_seconds=sum(timings),
        nominal_tile_size=tile_size, nominal_overlap=overlap,
        edge_policy='edge-aligned final tile; actual overlap may exceed nominal',
        input_shape=list(bgr.shape), input_format='HWC contiguous uint8 BGR')


def prepare_triage_image(rgb, row, *, mode='one-metre'):
    """Create native or exact physical 1m area-averaged inference RGB only."""
    from scripts.maritime_planet_intake import area_matrix
    if not isinstance(rgb, np.ndarray) or rgb.ndim != 3 or rgb.shape[0] != 3 or rgb.dtype != np.uint8:
        raise ValueError('Original native CHW uint8 RGB required')
    kind = row.get('_triage_source_kind', 'maxar')
    if kind in ('skysat', 'skysat_raw_window'):
        encoded_core = np.asarray(row['core_bounds'], np.float64)
        if (encoded_core.shape != (4,) or not np.isfinite(encoded_core).all()
                or not np.allclose(encoded_core[2:]-encoded_core[:2], [1280,1280], atol=1e-4, rtol=0)):
            raise ValueError('SkySat encoded core is not nominal 1280m')
        # Existing SkySat recipes use the nominal 0.8m pixel grid, retaining
        # encoded affine rounding separately; inference follows that same basis.
        left, _, _, top = encoded_core
        core = np.asarray([left,top-1280,left+1280,top])
        row = dict(row, product_spacing_m=.8, area_phase_m=[0,0],
                   physical_core_bounds=core.tolist(),
                   physical_context_bounds=(core+[-40,-40,40,40]).tolist())
    spacing = float(row['product_spacing_m'])
    phase = np.asarray(row['area_phase_m'], np.float64)
    core = np.asarray(row['physical_core_bounds'], np.float64)
    context = np.asarray(row['physical_context_bounds'], np.float64)
    if (not math.isfinite(spacing) or not 0 < spacing <= 1 or phase.shape != (2,)
            or not np.isfinite(phase).all() or (phase < -1e-8).any() or (phase >= spacing+1e-8).any()
            or core.shape != (4,) or context.shape != (4,) or not np.isfinite(core).all()
            or not np.isfinite(context).all() or not np.allclose(core[2:] - core[:2], [1280, 1280], atol=1e-7, rtol=0)
            or not np.allclose(context, core + [-40, -40, 40, 40], atol=1e-7, rtol=0)):
        raise ValueError('Invalid physical Maxar triage geometry')
    expected_shape = tuple(math.ceil((max(0, start)+1360)/spacing-1e-10) for start in phase[::-1])
    if rgb.shape[1:] != expected_shape:
        raise ValueError('Native contributor dimensions disagree with physical context')
    if mode == 'native':
        derivative = rgb
        bounds = [(40+phase[0])/spacing, (40+phase[1])/spacing,
                  (1320+phase[0])/spacing, (1320+phase[1])/spacing]
        output_spacing, reduction = spacing, 'none: original decoded native grid'
    elif mode == 'one-metre':
        wx = area_matrix(max(0, phase[0]), 1360, spacing, 1.0)
        wy = area_matrix(max(0, phase[1]), 1360, spacing, 1.0)
        if rgb.shape[1:] != (wy.shape[1], wx.shape[1]):
            raise ValueError('Native contributor dimensions disagree with physical context')
        reduced = np.stack([np.asarray((wy @ channel.astype(np.float64)) @ wx.T) for channel in rgb])
        derivative = np.ascontiguousarray(np.floor(np.clip(reduced, 0, 255) + 0.5), dtype=np.uint8)
        bounds, output_spacing = [40, 40, 1320, 1320], 1.0
        reduction = 'exact separable physical area average in float64, final uint8 half-up rounding; triage only'
    else:
        raise ValueError('Unknown triage spacing mode')
    bgr = np.ascontiguousarray(np.moveaxis(derivative[::-1], 0, -1))
    geometry = dict(mode=mode, pixel_spacing_m=output_spacing, core_bounds_pixels=bounds,
        native_product_spacing_m=spacing, native_context_shape=list(rgb.shape),
        native_context_phase_m=phase.tolist(), resampling=reduction,
        source_pixels_modified=False, inference_derivative_only=True)
    if kind in ('skysat','skysat_raw_window'):
        geometry['coordinate_basis'] = 'nominal 0.8m pixel grid; encoded affine retained in source provenance'
    return bgr, geometry


def initialize_detector(weights, confidence, device='cpu', cpu_threads=1):
    """Import the pinned detector lazily, with one CPU thread and no downloads."""
    import ultralytics
    import torch
    if ultralytics.__version__ != ULTRALYTICS_VERSION:
        raise ValueError('Ultralytics 8.3.0 is required')
    torch.set_num_threads(validate_cpu_threads(cpu_threads))
    torch.set_num_interop_threads(1)
    if device != 'cpu' and (not torch.cuda.is_available() or torch.cuda.device_count()<1):
        raise ValueError('Explicit CUDA triage requested but GPU is unavailable')
    model = ultralytics.YOLO(str(Path(weights).resolve()), task='obb')
    if model.names.get(1) != 'ship' or model.names.get(7) != 'harbor':
        raise ValueError('Weights do not declare expected DOTA ship/harbor classes')
    def predict(tile):
        results = model.predict(source=tile, imgsz=1024, conf=confidence,
            classes=[1, 7], device=(torch.device(device) if cpu_threads != 1 else device),
            verbose=False, save=False, augment=False)
        if cpu_threads != 1:
            predict.runtime["effective_inference_cpu_threads"]=torch.get_num_threads()
            if device == "cpu" and torch.get_num_threads() != cpu_threads:
                raise ValueError("Detector changed the explicitly requested CPU thread count")
        records = []
        for result in results:
            if result.obb is None:
                continue
            polygons = result.obb.xyxyxyxy.cpu().numpy()
            classes = result.obb.cls.cpu().numpy()
            confidences = result.obb.conf.cpu().numpy()
            records.extend(dict(class_id=int(label), confidence=float(score), polygon=polygon.tolist())
                           for polygon, label, score in zip(polygons, classes, confidences))
        return records
    predict.runtime = dict(ultralytics_version=ultralytics.__version__, torch_version=torch.__version__,
        numpy_version=np.__version__, cpu_threads=torch.get_num_threads(),
        cpu_interop_threads=torch.get_num_interop_threads(), device=device, class_names=model.names)
    if device != 'cpu':
        predict.runtime.update(cuda_version=torch.version.cuda,
            gpu_name=torch.cuda.get_device_name(0),gpu_capability=list(torch.cuda.get_device_capability(0)),
            gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
            inference_precision='float32; Ultralytics half=False default')
    return predict


_WORKER = None


def initialize_worker(project, output_prefix, scratch, weights, confidence, mode, iou, instance, device='cpu', earthview_verification='original-assembly', cpu_threads=1):
    from google.cloud import storage
    from scripts.maritime_ultimate_release import CloudStore
    import os
    global _WORKER
    store = CloudStore(storage.Client(project=project), output_prefix,
                       Path(scratch)/('worker-'+str(os.getpid())))
    _WORKER = dict(store=store, predictor=None, weights=weights, confidence=confidence,
                   mode=mode, iou=iou, instance=instance, device=device, earthview_verification=earthview_verification,cpu_threads=cpu_threads)


def load_original_context(store, row, earthview_verification='original-assembly'):
    """Reproduce canonical HR from pinned observed contributors before triage."""
    from contextlib import ExitStack
    import rasterio
    from rasterio.io import MemoryFile
    from scripts.maritime_ultimate_release import read_maxar, read_skysat
    from scripts.maritime_cloud_inventory import digest_bytes
    from src.maritime_recipes import canonical_hr
    if earthview_verification not in ('original-assembly','pinned-context'):
        raise ValueError('Unsupported EarthView verification mode')
    kind = row.get('_triage_source_kind', 'maxar')
    with ExitStack() as stack:
        if kind in ('maxar', 'earthview'):
            memory = stack.enter_context(MemoryFile(store.get(row['native_context'])))
            ds = stack.enter_context(memory.open())
            stored = store.get(row['hr'])
            if kind == 'maxar':
                context, _, _, _ = read_maxar(ds, row, stored)
                rgb = ds.read()
            else:
                from src.maritime_earthview_source import canonical_from_native, verify_original_assembly
                from scripts.maritime_earthview_intake import apply_toa_mapping
                if earthview_verification == 'pinned-context':
                    from src.maritime_earthview_source import validate_source
                    validate_source(row)  # All16 original-source pins/placement/hash evidence remain mandatory.
                    reproduced = None
                else:
                    reproduced = verify_original_assembly(store, row)
                context, _, _, _ = canonical_from_native(ds, row, stored)
                if reproduced is not None and not np.array_equal(reproduced, context):
                    raise ValueError('EarthView original assembly HR mismatch')
                native = ds.read()
                rgb = apply_toa_mapping(native, row['rgb_mapping']) if row['input_product'] == 'TOA' else native
        else:
            raw = kind == 'skysat_raw_window'
            ref = row['source_object'] if raw else row['mother']
            ds = stack.enter_context(rasterio.open(store.cached_source(ref)))
            mask = None
            if row['mask'].get('object'):
                mask = stack.enter_context(rasterio.open(store.cached_source(row['mask']['object'])))
            rgb, valid = read_skysat(ds, row, mask, raw)
            context, _ = canonical_hr(rgb, valid)
            for field, pixels in [('decoded_native_context_sha256',rgb),
                                  ('decoded_useful_core_sha256',rgb[:,50:-50,50:-50])]:
                if row.get(field) and digest_bytes(pixels.tobytes()) != row[field]:
                    raise ValueError('SkySat decoded original pixel hash mismatch')
    store.trim_cache()
    return rgb, context


def process_candidate(task):
    row, config_sha = task
    from scripts.maritime_cloud_inventory import digest_bytes, json_bytes
    state = _WORKER
    started = time.perf_counter()
    rgb, context = load_original_context(state['store'], row, earthview_verification=state['earthview_verification'])
    metrics = quality_metrics(context[:,16:-16,16:-16])
    bgr, geometry = prepare_triage_image(rgb, row, mode=state['mode'])
    del rgb, context
    if state['predictor'] is None:
        state['predictor'] = initialize_detector(state['weights'], state['confidence'], state['device'],state['cpu_threads'])
    predictor = state['predictor']
    detections, timings = infer_tiles(bgr, predictor)
    summary = summarize_ship_detections(detections, pixel_spacing_m=geometry['pixel_spacing_m'],
        core_bounds=geometry['core_bounds_pixels'], confidence_threshold=state['confidence'],
        iou_threshold=state['iou'])
    kind = row.get('_triage_source_kind', 'maxar')
    record = dict(schema='maritime-detector-triage-record-v1', sample_id=row['sample_id'],
        vm_instance_id=state['instance'], configuration_sha256=config_sha,
        candidate_sha256=digest_bytes(json_bytes(row)),
        source_provenance=row['source'] if kind == 'maxar' else row,
        original_mask_provenance=row.get('original_mask_provenance',row.get('mask')),
        quality=metrics, inference_geometry=geometry, detector_timings=timings,
        detector_runtime=predictor.runtime, detections=summary, ranking=rank_scene(summary),
        evidence_status='model_predicted', decision='review_pending',
        elapsed_seconds=time.perf_counter()-started, source_pixels_modified=False,
        inference_rgb_sha256=digest_bytes(bgr.tobytes()))
    if kind == 'earthview':
        replayed=state['earthview_verification']=='original-assembly'
        record.update(earthview_verification=state['earthview_verification'],
            original_assembly_replayed=replayed,original_assembly_deferred_to_final_release=not replayed)
    if kind in ('maxar','earthview'):
        record.update(native_context=row['native_context'],hr=row['hr'])
    else:
        record.update(source_kind=kind, source_object=row.get('source_object',row.get('mother')))
    return record


def run(args):
    from scripts.maritime_cloud_inventory import (create_only, digest_bytes, file_sha256,
        json_bytes, parse_gs, require_cloud_vm, verify_output)
    instance = require_cloud_vm()  # Guard before any imagery access or cloud writes.
    from google.cloud import storage
    if not 0 <= args.confidence <= 1 or not 0 < args.iou_threshold <= 1:
        raise ValueError('Invalid detector confidence or duplicate IoU threshold')
    workers = validate_workers(getattr(args, 'workers', 1))
    cpu_threads = validate_cpu_threads(getattr(args,'cpu_threads',1))
    device = validate_device(getattr(args, 'device', 'cpu'),workers)
    earthview_verification=getattr(args,'earthview_verification','original-assembly')
    if earthview_verification not in ('original-assembly','pinned-context'):
        raise ValueError('Unsupported EarthView verification mode')
    weights = verify_weights(args.weights, args.weights_sha256)
    scratch = Path(args.scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    client = storage.Client(project=args.project)
    if args.manifest.startswith('gs://'):
        bucket_name, name = parse_gs(args.manifest)
        blob = client.bucket(bucket_name).blob(name)
        blob.reload()
        manifest_bytes = blob.download_as_bytes(if_generation_match=int(blob.generation), checksum='crc32c')
        manifest_pin = dict(uri=args.manifest, generation=str(blob.generation), bytes=len(manifest_bytes),
                            crc32c=blob.crc32c, sha256=digest_bytes(manifest_bytes))
    else:
        manifest_bytes = Path(args.manifest).read_bytes()
        manifest_pin = dict(local_vm_input_sha256=digest_bytes(manifest_bytes), bytes=len(manifest_bytes))
    source_order=getattr(args,'source_order','identity')
    rows = candidate_rows(json.loads(manifest_bytes.decode('utf-8-sig')), args.limit, source_order=source_order)
    task_grouping=getattr(args,'task_grouping','candidate')
    groups=candidate_task_groups(rows,task_grouping,source_order)
    configuration = dict(schema='maritime-detector-triage-config-v1', input_manifest=manifest_pin,
        candidate_selection_sha256=digest_bytes(json_bytes(rows)), limit=args.limit,
        weights=weights, detector='YOLOv8s-OBB', ultralytics_version=ULTRALYTICS_VERSION,
        detector_license='AGPL-3.0; research use does not waive license obligations',
        code_sha256=file_sha256(Path(__file__)), triage_mode=args.triage_mode,
        confidence_threshold=args.confidence, duplicate_iou_threshold=args.iou_threshold,
        nominal_tile_size=1024, nominal_overlap=256, cpu_threads=cpu_threads, device='cpu',
        decision='review_triage_only; no sample acceptance or dataset READY')
    if cpu_threads != 1:
        configuration['schema']='maritime-detector-triage-config-v2'
    if device != 'cpu':
        configuration.update(schema='maritime-detector-triage-config-v2',device=device,
            gpu_scope='model inference only; scientific raster recipes unchanged',
            inference_precision='float32; Ultralytics half=False default')
    if earthview_verification != 'original-assembly':
        configuration.update(schema='maritime-detector-triage-config-v2',
            earthview_verification=earthview_verification,original_assembly_deferred_to_final_release=True)
    if source_order != 'identity':
        configuration['source_order'] = source_order
    if workers != 1:
        configuration['parallel_source_workers'] = workers
    if task_grouping != 'candidate':
        configuration.update(schema='maritime-detector-triage-config-v2',task_grouping=task_grouping,
            worker_source_cache_target_bytes=2000000000,aggregate_source_cache_target_bytes=workers*2000000000,
            source_cache_caveat='LRU target; single oversized original is retained for verification')
    config_sha = digest_bytes(json_bytes(configuration))
    bucket_name, prefix = parse_gs(args.output_prefix)
    bucket = client.bucket(bucket_name)
    def publish(name, payload):
        uri = f'gs://{bucket_name}/{prefix}/{name}'
        return dict(uri=uri, **create_only(bucket.blob(f'{prefix}/{name}'), payload))
    def existing(name):
        blob = bucket.blob(f'{prefix}/{name}')
        if not blob.exists():
            return None
        blob.reload()
        payload = blob.download_as_bytes(if_generation_match=int(blob.generation), checksum='crc32c')
        ref = dict(uri=f'gs://{bucket_name}/{prefix}/{name}', generation=str(blob.generation),
            bytes=len(payload), crc32c=blob.crc32c, sha256=digest_bytes(payload))
        return json.loads(payload), ref
    config_ref = publish('configuration.json', json_bytes(configuration))
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    refs = []
    pending = []
    # Spawn isolates PyTorch/YOLO and GCS clients, and avoids unsafe shared models.
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
            initializer=initialize_worker, initargs=(args.project,args.output_prefix,args.scratch,
                args.weights,args.confidence,args.triage_mode,args.iou_threshold,instance,device,earthview_verification,cpu_threads)) as pool:
        next_index = 0
        while next_index < len(groups) or pending:
            while next_index < len(groups) and len(pending) < workers:
                entries=[];missing=[]
                for row in groups[next_index]:
                    name = 'records/' + hashlib.sha256(row['sample_id'].encode()).hexdigest() + '.json'
                    cached = existing(name)
                    if cached:
                        record, ref = cached
                        if record['configuration_sha256'] != config_sha or record['candidate_sha256'] != digest_bytes(json_bytes(row)):
                            raise ValueError('Immutable triage checkpoint identity mismatch')
                    else:
                        record,ref=None,None;missing.append(row)
                    entries.append((row,name,record,ref))
                future=pool.submit(process_candidate_group,(missing,config_sha)) if missing else None
                pending.append((entries,future));next_index += 1
            entries,future = pending.pop(0)
            calculated=iter(future.result() if future is not None else [])
            for row,name,record,ref in entries:
                if record is None:
                    record=next(calculated);ref=publish(name,json_bytes(record))
                refs.append(ref)
                checkpoint = dict(configuration=config_ref, processed_count=len(refs),
                    records=refs if workers == 1 else [ref], decision='review_triage_only')
                if workers != 1:
                    checkpoint.update(schema='maritime-detector-triage-checkpoint-v2',
                        records_scope='latest_processed_record; final manifest contains complete ordered list')
                publish(f'checkpoints/{len(refs):06d}.json', json_bytes(checkpoint))
                print(json.dumps(dict(processed=len(refs), sample_id=row['sample_id'],
                    ship_counts=record['detections']['ship_counts'], harbor_count=record['detections']['harbor_count'])), flush=True)
    result = dict(schema='maritime-detector-triage-manifest-v1', configuration=config_ref,
        processed_count=len(refs), selected_count=len(rows), records=refs,
        evidence_status='model_predicted', decision='review_triage_complete; no sample acceptance')
    ref = publish('triage-manifest.json', json_bytes(result))
    print(json.dumps(dict(triage_manifest=ref)), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, help='VM-local JSON or GCS candidate-manifest URI')
    parser.add_argument('--weights', required=True, help='Verified weights file on the VM')
    parser.add_argument('--weights-sha256', required=True)
    parser.add_argument('--project', default=GCP_PROJECT)
    parser.add_argument('--limit', type=int, default=8)
    parser.add_argument('--cpu-threads',type=int,default=1,help='Torch intra-op threads per isolated worker, 1-8')
    parser.add_argument('--workers', type=int, default=1, help='Independent model/source workers, 1-8')
    parser.add_argument('--earthview-verification',choices=['original-assembly','pinned-context'],
                        default='original-assembly',help='Triage-only replay policy; final release always replays original assembly')
    parser.add_argument('--device',choices=['cpu','cuda:0'],default='cpu',
                        help='Explicit inference device; GPU requires workers=1, never falls back')
    parser.add_argument('--task-grouping',choices=['candidate','original-object'],default='candidate',
                        help='Optional one-original task affinity; requires original-object ordering')
    parser.add_argument('--source-order',choices=['identity','original-object'],default='identity',
                        help='Order before candidate limit; group original SkySat objects for cache reuse')
    parser.add_argument('--triage-mode', choices=['one-metre', 'native'], default='one-metre')
    parser.add_argument('--confidence', type=float, default=0.25)
    parser.add_argument('--iou-threshold', type=float, default=0.5)
    parser.add_argument('--output-prefix', required=True)
    parser.add_argument('--scratch', required=True)
    return run(parser.parse_args())


if __name__ == '__main__':
    main()
