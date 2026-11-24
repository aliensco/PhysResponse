"""Validate a visible-event revision against its immutable source and review ledger."""
import hashlib
import json
from functools import lru_cache
from pathlib import Path

FIELDS=('case_id','event','reference_video_sha256','event_contract_sha256','window_sha256')

@lru_cache(maxsize=8)
def checked_rows(path, expected_sha):
    raw=Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=expected_sha:
        raise ValueError('Visible reference source changed')
    return [json.loads(line) for line in raw.decode('utf-8-sig').splitlines() if line]

def validate(row,digest):
    cert=row['reference_certificate']
    if digest(cert)!=row.get('reference_evidence_sha256'):
        raise ValueError('Visible reference certificate digest mismatch')
    if any(cert.get(k)!=row.get(k) for k in (*FIELDS,'reference_label','state_sha256')):
        raise ValueError('Visible reference certificate binding mismatch')
    parents=checked_rows(cert['parent_path'],cert['parent_sha256'])
    matches=[p for p in parents if all(p.get(k)==v for k,v in cert['parent_key'].items())]
    if len(matches)!=1:raise ValueError('Visible reference parent is not unique')
    parent=matches[0]
    if any(parent.get(k)!=row.get(k) for k in ('case_id','reference_video_sha256','window_sha256','state_sha256')):
        raise ValueError('Visible reference media, window or state changed')
    if cert['mode']=='unchanged_event_specification':
        if cert['parent_event_spec_sha256']!=cert['event_spec_sha256'] or parent['reference_label']!=row['reference_label']:
            raise ValueError('Unchanged visible reference specification or value differs')
        # Validate the original certificate with its original contract binding.
        from .phyresponse_core100 import index_native_truth
        index_native_truth([parent])
    elif cert['mode']=='reviewed_visible_event':
        audits=checked_rows(cert['review_path'],cert['review_sha256'])
        matches=[a for a in audits if all(a.get(k)==row.get(k) for k in FIELDS)]
        if len(matches)!=1 or matches[0]['result']!=row['reference_label'] or not matches[0].get('proof'):
            raise ValueError('Visible reference review missing or inconsistent')
    else:raise ValueError('Unknown visible reference revision mode')
