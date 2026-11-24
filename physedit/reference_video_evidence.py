"""Validate the fixed Core100 source-video review, without inventing state files."""
import functools
import gzip
import hashlib
import json
import math
from pathlib import Path

SCHEMA = 'phyresponse-source-video-review-v1'
LEDGER = Path('review/phyresponse_gold4000_20261106/reference_event_truth_20261110/event_visual_review_20261111/VISUAL_DECISIONS.json')
LEDGER_SHA256 = '814bd8bc4a5719fab259df4df9e9e2bf708eb73a44d647195babf395f5942c14'
CLOSURE_LEDGER = Path('review/phyresponse_gold4000_20261106/reference_event_truth_20261110/evidence_closure_20261111/VISUAL_DECISIONS.json')
CLOSURE_LEDGER_SHA256 = 'd62bb2991d80e24abf139d00f3cd9ba5a43f9f38ec9962ac9bedb326558e1740'
CONTACT_CLOSURE_LEDGER = Path('review/phyresponse_gold4000_20261106/reference_event_truth_20261110/contact_closure_v4_20261111/VISUAL_DECISIONS.json')
CONTACT_CLOSURE_LEDGER_SHA256 = '41d1aa40eb9338d7b6ac8fac6dc945c97deb066abb7a3595ead56bd9d94a2165'
FIELDS = ('case_id', 'event', 'reference_video_sha256', 'event_contract_sha256', 'window_sha256')
NATIVE_CONTACT_SCHEMA = 'phyresponse-native-source-contact-review-v1'
NATIVE_CONTACT_LEDGER = Path('review/phyresponse_gold4000_20261106/reference_event_truth_20261110/network_recovery_20261111/T12_CONTACT_DECISIONS.json')
NATIVE_CONTACT_LEDGER_SHA256 = '04710d6bed2b1675ef242666ef48c4c9a9df166abf5c59b8e6e0465fb3c4bd8d'

@functools.lru_cache(maxsize=16384)
def _file_hash(path, size, mtime_ns):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def file_hash(path):
    p = Path(path)
    st = p.stat()
    return _file_hash(str(p.resolve()), st.st_size, st.st_mtime_ns)


@functools.lru_cache(maxsize=4)
def _native_contact_state(path, sha256):
    raw=Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=sha256:
        raise ValueError('Invalid native contact source state bytes')
    return json.loads(gzip.decompress(raw) if raw[:2]==b'\x1f\x8b' else raw)


def validate_native_source_contact_reference(row, digest_json):
    """Accept only the pinned positive T12 source-video/native-response review."""
    cert=row.get('reference_certificate',{})
    fail=lambda reason:ValueError('Invalid native source-contact certificate: '+reason)
    if (cert.get('schema')!=NATIVE_CONTACT_SCHEMA
            or Path(cert.get('review_ledger_path','')).resolve()!=NATIVE_CONTACT_LEDGER.resolve()
            or cert.get('review_ledger_sha256')!=NATIVE_CONTACT_LEDGER_SHA256
            or file_hash(NATIVE_CONTACT_LEDGER)!=NATIVE_CONTACT_LEDGER_SHA256
            or digest_json(cert)!=row.get('reference_evidence_sha256')):
        raise fail('review ledger binding')
    records=json.loads(NATIVE_CONTACT_LEDGER.read_text(encoding='utf8'))
    decisions={d['decision_id']:d for d in records}
    if len(decisions)!=len(records):raise fail('duplicate decision')
    decision=decisions.get(cert.get('decision_id'))
    bound=(*FIELDS,'status','reference_label','window_start_s','window_end_s','state_sha256')
    if (decision is None or any(row.get(k)!=cert.get(k) or row.get(k)!=decision.get(k) for k in bound)
            or row['case_id'] not in ('T12_C01','T12_C02','T12_C03','T12_C05')
            or row['event']!='stapler_contacts_tissue_box'
            or row['status']!='verified' or type(row['reference_label']) is not int or row['reference_label']!=1
            or row.get('state_materialized') is not True
            or row.get('window_fps')!=decision['window_fps']
            or decision.get('human_gold') is not False
            or decision.get('reviewer_kind')!='assistant_visual_with_same_source_native_response'
            or cert.get('event_spec_sha256')!=digest_json(decision['event_spec'])
            or cert.get('event_spec_sha256')!=decision['event_spec_sha256']
            or decision['event_spec'].get('actors')!=['t12_stapler','t12_tissue_box']):
        raise fail('decision/contract/label binding')
    if file_hash(decision['reference_video_path'])!=row['reference_video_sha256']:
        raise fail('source video bytes')
    for evidence in (decision['identity_config_snapshot'],decision['native_response_support']):
        if file_hash(evidence['path'])!=evidence['sha256']:raise fail('source identity/response evidence')
    state=_native_contact_state(decision['native_state_path'],row['state_sha256'])
    frame=decision['known_by_frame'];fps=decision['source_fps']
    numbers=[f['frame'] for f in decision['frames']]
    if (fps!=30 or decision['source_frames']!=len(state['frames']) or numbers!=list(range(frame-3,frame+4))
            or frame/fps>row['window_end_s'] or frame/fps<row['window_start_s']):
        raise fail('positive witness outside observation window')
    for evidence in decision['frames']:
        if (file_hash(evidence['path'])!=evidence['sha256']
                or not math.isclose(evidence['time_s'],evidence['frame']/fps,abs_tol=1e-9)):
            raise fail('source frame bytes/time')
    before={o['name']:o for o in state['frames'][frame-1]['objects']}
    after={o['name']:o for o in state['frames'][frame]['objects']}
    if (after['t12_tissue_box']['velocity'][0]<=before['t12_tissue_box']['velocity'][0]
            or after['t12_stapler']['velocity'][0]>=before['t12_stapler']['velocity'][0]
            or any(after[a].get('motion',{}).get('manual_pose') is not False for a in ('t12_stapler','t12_tissue_box'))):
        raise fail('same-source impact corroboration')

@functools.lru_cache(maxsize=4)
def _records(path,digest):
    rows = json.loads(Path(path).read_text(encoding='utf-8'))
    index = {r['decision_id']: r for r in rows}
    if len(index) != len(rows):
        raise ValueError('Duplicate source-video review decision')
    return index

def validate_source_video_reference(row, digest_json):
    cert = row.get('reference_certificate', {})
    fail = lambda reason: ValueError('Invalid source-video reference certificate: ' + reason)
    ledger=Path(cert.get('review_ledger_path','')).resolve()
    permitted={LEDGER.resolve():LEDGER_SHA256,CLOSURE_LEDGER.resolve():CLOSURE_LEDGER_SHA256,
               CONTACT_CLOSURE_LEDGER.resolve():CONTACT_CLOSURE_LEDGER_SHA256}
    pinned=permitted.get(ledger)
    if (cert.get('schema') != SCHEMA or pinned is None
            or cert.get('review_ledger_sha256') != pinned
            or file_hash(ledger) != pinned
            or digest_json(cert) != row.get('reference_evidence_sha256')):
        raise fail('review ledger binding')
    decision = _records(str(ledger),pinned).get(cert.get('decision_id'))
    bound = (*FIELDS, 'status', 'reference_label', 'window_start_s', 'window_end_s')
    if (decision is None or any(row.get(k) != cert.get(k) or row.get(k) != decision.get(k) for k in bound)
            or cert.get('event_spec_sha256') != digest_json(decision['event_spec'])
            or cert.get('event_spec_sha256') != decision['event_spec_sha256']
            or row.get('window_fps') != decision['window_fps']
            or decision.get('reviewer_kind') != 'assistant_visual'
            or decision.get('human_gold') is not False
            or decision.get('independent_blind_adjudication') is not False):
        raise fail('decision/contract/label binding')
    if row.get('state_sha256') is not None:
        native=decision.get('native_state_support',{})
        if (ledger not in (CLOSURE_LEDGER.resolve(), CONTACT_CLOSURE_LEDGER.resolve()) or row.get('state_materialized') is not True
                or row['state_sha256']!=native.get('state_sha256')
                or file_hash(native.get('path',''))!=native.get('storage_sha256')):
            raise fail('native support state binding')
    elif row.get('state_materialized') is not False:
        raise fail('unmaterialized state must be explicit')
    if file_hash(decision['reference_video_path']) != row['reference_video_sha256']:
        raise fail('source video bytes')
    identity = decision['identity_config_snapshot']
    if file_hash(identity['path']) != identity['sha256']:
        raise fail('original actor identity evidence')
    frames = decision['frames']
    numbers = [f['frame'] for f in frames]
    if (numbers != decision['reviewed_source_frames'] or numbers != sorted(set(numbers))
            or not numbers or numbers[0] < 0 or numbers[-1] >= decision['source_frames']):
        raise fail('source frame indices')
    for f in frames:
        if (file_hash(ledger.parent / f['path']) != f['sha256']
                or not math.isclose(f['time_s'], f['frame'] / decision['source_fps'], abs_tol=1e-9)):
            raise fail('source frame bytes/time')
    mode = decision['mode']
    label, end, fps = row['reference_label'], row['window_end_s'], decision['source_fps']
    last_time = (decision['source_frames'] - 1) / fps
    if row['status'] == 'unresolved':
        if label is not None or mode != 'unresolved' or not decision.get('reason'):
            raise fail('unresolved evidence cannot supply a label')
        return
    if row['status'] != 'verified' or type(label) is not int or label not in (0, 1):
        raise fail('binary observation')
    if mode == 'positive_witness':
        by = decision['known_by_frame']
        if label != 1 or by not in numbers or by / fps > end:
            raise fail('positive witness outside window')
        if row['event'] == 'complete_required_contact_chain':
            required = decision['event_spec']['required_contact_pairs']
            if ([x['actors'] for x in decision['components']] != required
                    or any(x['label'] != 1 for x in decision['components'])):
                raise fail('incomplete required contact chain')
        elif row['event'] == 'source_motion':
            support = decision['native_qa_support']
            if file_hash(support['path']) != support['sha256']:
                raise fail('original motion metadata')
        else:
            raise fail('unsupported positive observation')
    elif mode == 'full_window_separation':
        t12_original_review = (ledger == CONTACT_CLOSURE_LEDGER.resolve()
                               and row['case_id'] == 'T12_C04'
                               and row['event'] == 'stapler_contacts_tissue_box'
                               and decision['event_spec'].get('actors') == ['t12_stapler', 't12_tissue_box']
                               and bool(decision.get('native_state_support')))
        if (label != 0 or end > last_time
                or numbers != list(range(math.ceil(end * fps) + 1))
                or (row['event'] not in ('complete_required_contact_chain', 'upper_nonpulled_book_ever_contacts_table')
                    and not t12_original_review)):
            raise fail('incomplete negative window evidence')
    elif mode == 'terminal_frame':
        if (end > last_time or row['event'] != 'upper_nonpulled_book_on_table_at_end'
                or not {math.floor(end * fps + 1e-8), math.ceil(end * fps)}.issubset(numbers)):
            raise fail('missing terminal native frame')
    elif mode == 'full_window_event_nonoccurrence':
        if (ledger!=CLOSURE_LEDGER.resolve() or label!=0 or end>last_time
                or numbers!=list(range(math.ceil(end*fps-1e-9)+1))
                or row['event'] not in ('t11_driver_eraser_edge_fall_begins','phone_edge_fall_after_can_contact')
                or not decision.get('native_state_support')):
            raise fail('incomplete source-event nonoccurrence review')
    else:
        raise fail('unsupported observation mode')
