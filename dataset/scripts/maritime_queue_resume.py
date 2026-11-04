"""Operational checkpoint validation and immutable closure reuse; no pixel processing."""


def validate_checkpoint(saved, batch_count, *, expected_config=None, migration_boundary=0):
    next_batch = saved['next_batch']
    if (type(next_batch) is not int or not migration_boundary <= next_batch <= batch_count
            or saved['batch_count'] != batch_count
            or len(saved['launches']) != next_batch
            or len(saved['closed']) + len(saved['active']) != next_batch
            or (expected_config is not None and saved.get('config') != expected_config)):
        raise ValueError('Checkpoint schedule/configuration mismatch')
    launches = saved['launches']
    if [row['batch'] for row in launches] != list(range(1, next_batch + 1)):
        raise ValueError('Checkpoint launch sequence mismatch')
    if len({row['container_id'] for row in launches}) != next_batch:
        raise ValueError('Checkpoint repeats a container identity')
    for index, identity in saved['active'].items():
        if not str(index).isdigit() or not 0 <= int(index) < next_batch:
            raise ValueError('Checkpoint active index outside launched schedule')
        if launches[int(index)]['container_id'] != identity:
            raise ValueError('Checkpoint active container identity mismatch')
    return saved


def publish_immutable(blob, raw):
    if not blob.exists():
        try:
            blob.upload_from_string(raw, if_generation_match=0, checksum='crc32c')
        except Exception as error:
            if getattr(error, 'code', None) != 412:
                raise
    blob.reload()
    actual = blob.download_as_bytes(if_generation_match=int(blob.generation), checksum='crc32c')
    if actual != raw:
        raise ValueError('Existing immutable controller evidence differs: ' + blob.name)
