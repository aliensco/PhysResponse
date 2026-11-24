"""Opt-in C evidence audit v1; never changes legacy benchmark admission.

This validates records, not physical identifiability. Callers must bind the
candidate family, source video and evaluator configuration independently.
"""
import math

VERSION = 'c-reference-evidence-audit-v1'
READABLE = {'point', 'interval'}
UNRESOLVED = {'upper_bound', 'lower_bound', 'unidentifiable'}


def reference_gate(reference, formatted, expected_sha256):
    """Fail closed on missing/mismatched calibration, without rewriting inputs."""
    reasons = []
    for label, row in (('native', reference), ('formatted', formatted)):
        if row is None:
            reasons.append(label + ':missing_record')
            continue
        if row.get('reference_video_sha256') != expected_sha256:
            reasons.append(label + ':reference_identity_mismatch')
        if row.get('kind') not in READABLE | UNRESOLVED:
            reasons.append(label + ':not_explained_or_no_observation')
        if row.get('truth_in_band') is not True:
            reasons.append(label + ':truth_not_certified_in_band')
    if reference and formatted:
        for field in ('clip', 'param', 'evaluator_sha256', 'context_sha256'):
            if not reference.get(field) or reference.get(field) != formatted.get(field):
                reasons.append(field + ':missing_or_mismatch')
    return dict(version=VERSION, admitted=not reasons, reasons=reasons,
                official_protocol_changed=False)


def export_reading(row, admitted):
    """Keep failure/bound rows; no best-fit value becomes a valid point by itself."""
    kind = row.get('kind', row.get('status', 'missing'))
    best, truth = row.get('best'), row.get('truth')
    numeric = lambda x: isinstance(x, (int, float)) and math.isfinite(x) and x > 0
    valid = bool(admitted and kind in READABLE and numeric(best) and numeric(truth))
    return dict(original_kind=kind, admitted=admitted, valid_reading=valid,
                estimate_type=kind, diagnostic_best=best, band=row.get('band'),
                absolute_error=abs(best - truth) if valid else None,
                relative_error=abs(best / truth - 1) if valid else None,
                uncertainty_scope='conditional_candidate_family_only')
