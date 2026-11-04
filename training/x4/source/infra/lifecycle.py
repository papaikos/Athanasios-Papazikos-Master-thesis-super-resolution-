"""Immutable lifecycle contracts shared by guest and external controller."""
from datetime import datetime, timedelta, timezone
import re

BOUNDARIES = {'soft_cutoff_utc':114, 'finalization_target_utc':117, 'cleanup_cutoff_utc':118, 'native_delete_at_utc':119, 'user_maximum_utc':120}

def utc(value):
    value = datetime.fromisoformat(value.replace('Z', '+00:00')) if isinstance(value, str) else value
    if value.tzinfo is None:
        raise ValueError('UTC timestamp must be timezone aware')
    return value.astimezone(timezone.utc)

def stamp(value):
    return utc(value).isoformat().replace('+00:00', 'Z')

def make_control(run_id, t0, **fields):
    result = dict(fields, run_id=run_id, t0_utc=stamp(t0))
    result.update({key:stamp(utc(t0)+timedelta(hours=hours)) for key,hours in BOUNDARIES.items()})
    return result

def validate_control(control, registration=None):
    expected = make_control(control['run_id'], control['t0_utc'])
    for key in ['run_id','t0_utc',*BOUNDARIES]:
        if control[key] != expected[key]:
            raise ValueError('Immutable deadline mismatch: '+key)
    if registration is not None:
        if registration['control'] != control:
            raise ValueError('Control differs from immutable registration')
        for key,value in control.items():
            if registration.get(key)!=value: raise ValueError('Flat registration field differs: '+key)
        for resource in [registration['instance'],*registration['disks']]:
            if not re.fullmatch(r'[a-z][a-z0-9-]{0,62}',resource['name']) or not str(resource['id']).isdigit():
                raise ValueError('Invalid resource name or numeric ID')
        if len({d['name'] for d in registration['disks']})!=len(registration['disks']):
            raise ValueError('Duplicate disk registrations')

def finalization_deadline(control, entered):
    return stamp(min(utc(entered)+timedelta(hours=2),utc(control['cleanup_cutoff_utc'])))
