"""Evidence-preserving shortening of an existing reference observation window."""
import functools
import hashlib
import json
from pathlib import Path


def is_occurrence(spec):
    if spec=={'definition':'Striker exhibits physical motion in the observation window.',
              'actors':['t19_striker']}:
        return True
    if (spec.get('actors')==['t03v5_upper_edge_can','t03v5_lower_edge_phone','t03v5_lower_transfer_ramp']
            and spec.get('definition')=='The original can contacts the original lower phone; subsequently the same phone passes the lower support edge, loses support and begins moving down. Floor impact and a 20cm displacement are separate outcomes. Contact timing is an observed prerequisite, not proof of physical causality.'):
        return True
    if spec.get('time_semantics')=='window_end_state':
        return False
    # Final/terminal measurements need a fresh end-state evaluation.
    if any('final' in c.get('quantity','') for c in spec.get('metric_rule',{}).get('clauses',[])):
        return False
    components=spec.get('component_definitions',{})
    if components:
        return all(is_occurrence(c) for c in components.values())
    return spec.get('time_semantics') in ('first_confirmed_occurrence','ever_event_transition')


@functools.lru_cache(maxsize=12)
def _records(path,sha,size,mtime_ns):
    raw=Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=sha:
        raise ValueError('Changed reference projection parent')
    return [json.loads(line) for line in raw.decode('utf-8-sig').splitlines() if line]


def records(path,sha):
    stat=Path(path).stat()
    return _records(path,sha,stat.st_size,stat.st_mtime_ns)


def witness_time(row):
    if row.get('first_time_s') is not None:return row['first_time_s']
    m=row.get('measurement',{})
    if m.get('known_by_s') is not None:return m['known_by_s']
    if m.get('known_by_video_time_s') is not None:return m['known_by_video_time_s']
    if (m.get('strict_clearance_m',0)>0 and m.get('first_contact_source_frame') is not None
            and m.get('initial_marker_x_m',float('inf'))<m.get('unchanged_pass_plane_x_m',-float('inf'))):
        return m['first_contact_source_frame']/row['source_reference_fps']
    cert=row.get('reference_certificate') or {}
    path=cert.get('review_ledger_path')
    if path:
        raw=Path(path).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=cert.get('review_ledger_sha256'):
            raise ValueError('Changed source review ledger')
        decision=next(d for d in json.loads(raw) if d['decision_id']==cert['decision_id'])
        if decision.get('known_by_frame') is not None:
            return decision['known_by_frame']/decision['source_fps']
    return None


def validate(row,digest):
    from .phyresponse_core100 import index_native_truth,REFERENCE_FIELDS
    c=row['reference_certificate']
    if digest(c)!=row.get('reference_evidence_sha256'):
        raise ValueError('Invalid reference projection certificate hash')
    parents=records(c['parent_reference_path'],c['parent_reference_sha256'])
    parent=next((r for r in parents if [r[f] for f in REFERENCE_FIELDS]==c['parent_key']),None)
    if parent is None or parent.get('reference_certificate',{}).get('schema')=='phyresponse-window-projection-v1':
        raise ValueError('Invalid reference projection ancestry')
    index_native_truth([parent])
    tasks=records(c['parent_tasks_path'],c['parent_tasks_sha256'])
    t=next((t for t in tasks if t['id']==c['parent_task_id']),None)
    if t is None:raise ValueError('Missing original projection task')
    spec=t['event_contract']['events'][row['event']]
    end=row['window_end_s']
    if (list(row[f] for f in REFERENCE_FIELDS)!=c.get('new_key')
            or digest(c.get('window'))!=row['window_sha256']
            or c['window']['start_s']!=row['window_start_s']
            or c['window']['end_s']!=end
            or c['window']['fps']!=row['window_fps']
            or not is_occurrence(spec) or digest(spec)!=c['event_spec_sha256']
            or parent['status']!='verified' or row['status']!='verified'
            or row['reference_label']!=parent['reference_label']
            or row['case_id']!=parent['case_id'] or row['event']!=parent['event']
            or row['reference_video_sha256']!=parent['reference_video_sha256']
            or row['state_sha256']!=parent['state_sha256']
            or row['window_start_s']!=parent['window_start_s']
            or not row['window_start_s']<end<parent['window_end_s']
            or c['parent_event_contract_sha256']!=t['event_contract_sha256']
            or t['event_contract_sha256']!=parent['event_contract_sha256']
            or digest(t['window'])!=parent['window_sha256']):
        raise ValueError('Invalid reference projection semantics/binding')
    if row['reference_label']==1:
        time=witness_time(parent)
        if time is None or not row['window_start_s']<=time<=end:
            raise ValueError('Positive event not witnessed inside shortened window')
