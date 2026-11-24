"""Auxiliary critical-boundary metrics.

These functions deliberately return an explicit status instead of inventing a
value when the required event-chain contract, same-side pair, or response grid
was never registered.  They are independent of NearAcc/PairAcc aggregation.
"""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path


def _mean(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def load_eventexact_truth(path):
    """Load only registered full-event truth; pending rows are not contracts."""
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    return {row['clip_id']: row for row in data
            if row.get('reference_truth_status') == 'registered'}


def event_exact_output(reference_events, generated_events, required_order=None,
                       event_times_s=None):
    """Return 1/0/None for one output against a full event contract.

    None means the output cannot be judged (a required event or an order
    timestamp is missing).  It is not converted to a model failure here.
    """
    if not reference_events or not generated_events:
        return None
    for event, expected in reference_events.items():
        actual = generated_events.get(event)
        if actual not in (0, 1):
            return None
        if actual != expected:
            return 0
    order = required_order or []
    if not order:
        return 1
    times = event_times_s or {}
    for before, after in order:
        # Order is required only when both events are declared to occur.
        if generated_events.get(before) == 1 and generated_events.get(after) == 1:
            left, right = times.get(before), times.get(after)
            if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
                return None
            if left > right:
                return 0
    return 1


def response_curve(rows, points):
    """Per-reference-point positive/negative/unknown response frequencies.

    ``p1`` is the fraction of registered outputs whose target event is positive,
    ``p0`` negative and ``p_unknown`` not judgeable/missing.  These are generated
    response frequencies, not reference probabilities.
    """
    by_clip = defaultdict(list)
    for row in rows:
        by_clip[row.get('clip_id')].append(row)
    result = []
    for point in points:
        clip_id = point.get('clip_id')
        own = by_clip.get(clip_id, [])
        values = [row.get('target_prediction') for row in own]
        k = len(values)
        positive = sum(value == 1 for value in values)
        negative = sum(value == 0 for value in values)
        unknown = k - positive - negative
        result.append(dict(
            clip_id=clip_id, scene_id=point.get('scene_id'),
            label=point.get('label'), bin=point.get('bin'),
            theta=point.get('theta'),
            normalized_absolute_interval=point.get('normalized_absolute_interval'),
            signed_interval=point.get('signed_interval'),
            K=k,
            p1=positive / k if k else None,
            p0=negative / k if k else None,
            p_unknown=unknown / k if k else None,
            accuracy=point.get('accuracy'),
            accuracy_lower_bound=point.get('accuracy_lower_bound')))
    return result


def score_pairs(pairs, points, rows):
    """Score already-registered pairs from point accuracies and output rows.

    Used for same-side controls when the registry supplies clip pairs but not
    precomputed scores.  Label equality is checked by the caller's registry.
    """
    points_by_clip = {point['clip_id']: point for point in points}
    rows_by_clip = defaultdict(list)
    for row in rows:
        rows_by_clip[row.get('clip_id')].append(row)
    result = []
    for pair in pairs:
        clips = pair.get('clips') or []
        if len(clips) != 2 or any(clip not in points_by_clip for clip in clips):
            result.append(dict(pair, score=None, reference_admitted=False))
            continue
        left, right = points_by_clip[clips[0]], points_by_clip[clips[1]]
        admitted = bool(left.get('critical_rgb_distance_admitted', True) and
                        right.get('critical_rgb_distance_admitted', True) and
                        left.get('label') == right.get('label'))
        value = None
        if admitted and left.get('accuracy') is not None and right.get('accuracy') is not None:
            if pair.get('pairing_mode') == 'independent':
                value = left['accuracy'] * right['accuracy']
            else:
                left_rows = {row['seed']: row for row in rows_by_clip.get(clips[0], [])}
                right_rows = {row['seed']: row for row in rows_by_clip.get(clips[1], [])}
                products = []
                for seed in sorted(set(left_rows) & set(right_rows)):
                    lrow, rrow = left_rows[seed], right_rows[seed]
                    if lrow.get('target_correct') is None or rrow.get('target_correct') is None:
                        products = []
                        break
                    products.append(lrow['target_correct'] * rrow['target_correct'])
                value = _mean(products) if products else None
        result.append(dict(pair, score=value, reference_admitted=admitted))
    return result


def pair_summary(pairs, *, name='PairAcc'):
    """Aggregate already-computed pair scores without deleting pending pairs."""
    registered = list(pairs)
    eligible = [pair for pair in registered if pair.get('reference_admitted', True)]
    values = [pair.get('score') for pair in eligible]
    known = [value for value in values if value is not None]
    full = bool(values) and len(known) == len(values)
    return dict(
        name=name,
        registered=len(registered),
        eligible=len(eligible),
        known=len(known),
        known_only=_mean(known),
        score=_mean(values) if full else None,
        coverage=len(known) / len(values) if values else None,
        status='no_registered_pairs' if not registered else
               'complete' if full else 'pending_endpoints')


def same_side_accuracy(pairs, points=None, rows=None):
    """SameSideAcc over registered same-label control pairs.

    The current Core100 registry contains cross-boundary pairs only, so this
    returns an explicit no-registration status instead of reusing PairAcc.
    """
    if points is not None:
        pairs = score_pairs(pairs, points, rows or [])
    value = pair_summary(pairs, name='SameSideAcc')
    if not pairs:
        value['status'] = 'no_registered_same_side_pairs'
    return value


def event_exact_summary(rows):
    """EventExact over rows whose full event-chain contract is registered."""
    eligible = [row for row in rows if row.get('full_event_chain_registered')]
    if not eligible:
        return dict(value=None, known_only=None, eligible_points=0,
                    known_points=0, coverage=0.0,
                    status='full_event_chain_contract_not_registered')
    final = [row for row in eligible if row.get('decision_final') and
             row.get('event_exact_full') is not None]
    values = [row['event_exact_full'] for row in final]
    return dict(
        value=_mean(values) if len(final) == len(eligible) else None,
        known_only=_mean(values), eligible_points=len(eligible),
        known_points=len(final),
        coverage=len(final) / len(eligible) if eligible else None,
        status='complete' if len(final) == len(eligible) else 'pending_event_labels')


def _delta_matches(reference, generated):
    """Return 1/0/None for the event-change vector of one output pair."""
    if not reference or not generated:
        return None
    for event in reference:
        if generated.get(event) not in (0, 1):
            return None
    return int(all(generated[event] - reference[event] == 0
                   for event in reference))


def _generated_delta(left_row, right_row):
    """Generated ΔE vector, or None when any required generated bit is missing."""
    reference = left_row.get('reference_events') or {}
    if not reference:
        return None
    left_generated = left_row.get('generated_events') or {}
    right_generated = right_row.get('generated_events') or {}
    for event in reference:
        if left_generated.get(event) not in (0, 1) or right_generated.get(event) not in (0, 1):
            return None
    return {event: right_generated[event] - left_generated[event] for event in reference}


def chain_pair_exact_summary(pairs, rows):
    """ChainPairExact and diagnostic event-change (ΔE) matching.

    A pair is eligible only when both endpoints carry a registered full-chain
    contract.  Coupled pairs are joined by seed; independent pairs use the
    product of endpoint EventExact rates.
    """
    by_clip = defaultdict(list)
    for row in rows:
        by_clip[row.get('clip_id')].append(row)
    evaluated = []
    for pair in pairs:
        clips = pair.get('clips') or []
        if len(clips) != 2:
            evaluated.append(dict(pair, chain_pair_exact=None, delta_match=None,
                                  status='pair_must_have_two_endpoints'))
            continue
        left, right = clips
        left_rows, right_rows = by_clip.get(left, []), by_clip.get(right, [])
        both = left_rows + right_rows
        if not both or not all(row.get('full_event_chain_registered') for row in both):
            evaluated.append(dict(pair, chain_pair_exact=None, delta_match=None,
                                  status='full_event_chain_contract_not_registered'))
            continue
        mode = pair.get('pairing_mode', 'independent')
        scores, deltas = [], []
        if mode == 'coupled_seed':
            left_by = {row['seed']: row for row in left_rows}
            right_by = {row['seed']: row for row in right_rows}
            for seed in sorted(set(left_by) & set(right_by)):
                lrow, rrow = left_by[seed], right_by[seed]
                if lrow.get('event_exact_full') is not None and rrow.get('event_exact_full') is not None:
                    scores.append(lrow['event_exact_full'] * rrow['event_exact_full'])
                # ΔE is defined per event vector; require both generated vectors.
                gen = _generated_delta(lrow, rrow)
                if gen is not None:
                    ref = {event: rrow['reference_events'].get(event, 0) - lrow['reference_events'].get(event, 0)
                           for event in lrow.get('reference_events', {})}
                    deltas.append(_delta_matches(ref, gen))
        else:
            for lrow in left_rows:
                for rrow in right_rows:
                    if lrow.get('event_exact_full') is not None and rrow.get('event_exact_full') is not None:
                        scores.append(lrow['event_exact_full'] * rrow['event_exact_full'])
                    gen = _generated_delta(lrow, rrow)
                    if gen is not None:
                        ref = {event: rrow['reference_events'].get(event, 0) - lrow['reference_events'].get(event, 0)
                               for event in lrow.get('reference_events', {})}
                        deltas.append(_delta_matches(ref, gen))
        known_scores = [value for value in scores if value is not None]
        known_deltas = [value for value in deltas if value is not None]
        evaluated.append(dict(
            pair,
            chain_pair_exact=_mean(scores) if scores and len(known_scores) == len(scores) else None,
            chain_pair_exact_known_only=_mean(known_scores),
            delta_match=_mean(known_deltas) if known_deltas else None,
            delta_known_outputs=len(known_deltas),
            status='complete' if scores and len(known_scores) == len(scores) else 'pending_event_labels'))
    eligible = [item for item in evaluated if item.get('reference_admitted', True) and
                item.get('status') not in ('full_event_chain_contract_not_registered',
                                           'pair_must_have_two_endpoints')]
    values = [item['chain_pair_exact'] for item in eligible]
    known = [value for value in values if value is not None]
    delta_values = [item['delta_match'] for item in eligible if item['delta_match'] is not None]
    return dict(
        pairs=evaluated,
        chain_pair_exact=_mean(values) if values and len(known) == len(values) else None,
        chain_pair_exact_known_only=_mean(known),
        chain_pair_exact_coverage=len(known) / len(values) if values else None,
        delta_match=_mean(delta_values) if delta_values else None,
        delta_match_pairs=len(delta_values),
        status='complete' if values and len(known) == len(values) else
               'no_full_event_chain_pairs' if not eligible else 'pending_event_labels')


def boundary_error_estimate(curve, reference_interval=None, min_points=3):
    """Estimate a model threshold from p1(theta) by linear interpolation.

    A point estimate requires at least three distinct theta points; a two-point
    bracket is reported as insufficient rather than silently treated as a
    boundary estimate.
    """
    clean = []
    for point in curve:
        theta, p1 = point.get('theta'), point.get('p1')
        if isinstance(theta, (int, float)) and isinstance(p1, (int, float)):
            clean.append((float(theta), float(p1), point.get('clip_id')))
    thetas = sorted({theta for theta, _, _ in clean})
    if len(thetas) < min_points:
        return dict(value=None, theta_hat=None, status='insufficient_response_grid',
                    distinct_theta=len(thetas), points=len(clean),
                    reference_interval=reference_interval)
    # Collapse repeated theta values to their mean p1.
    grouped = defaultdict(list)
    for theta, p1, _ in clean:
        grouped[theta].append(p1)
    values = sorted((theta, _mean(ps)) for theta, ps in grouped.items())
    crossings = []
    for (left_theta, left_p1), (right_theta, right_p1) in zip(values, values[1:]):
        if left_p1 == .5:
            crossings.append(left_theta)
        elif (left_p1 - .5) * (right_p1 - .5) < 0:
            fraction = (.5 - left_p1) / (right_p1 - left_p1)
            crossings.append(left_theta + fraction * (right_theta - left_theta))
    crossings = sorted(set(round(value, 12) for value in crossings))
    if not crossings:
        status = 'no_observed_crossing'
    elif len(crossings) > 1:
        status = 'multiple_observed_crossings'
    else:
        status = 'ok'
    theta_hat = crossings[0] if status == 'ok' else None
    error_lower = error_upper = None
    if theta_hat is not None and reference_interval:
        low, high = map(float, reference_interval)
        error_lower = 0.0 if low <= theta_hat <= high else min(abs(theta_hat - low), abs(theta_hat - high))
        error_upper = max(abs(theta_hat - low), abs(theta_hat - high))
    return dict(value=theta_hat, theta_hat=theta_hat, status=status,
                distinct_theta=len(thetas), points=len(clean),
                crossings=crossings, reference_interval=reference_interval,
                error_lower=error_lower, error_upper=error_upper)


def boundary_error_summary(curve, pairs, reference_intervals=None):
    """Per-pair BoundaryError diagnostics; no grid means no point estimate."""
    reference_intervals = reference_intervals or {}
    attempts = []
    for pair in pairs:
        clips = pair.get('clips') or []
        own = [point for point in curve if point.get('clip_id') in clips]
        interval = reference_intervals.get(pair.get('pair_id'))
        attempts.append(dict(pair_id=pair.get('pair_id'), clips=clips,
                             **boundary_error_estimate(own, interval)))
    known = [item for item in attempts if item.get('theta_hat') is not None]
    return dict(pairs=attempts, value=known[0]['theta_hat'] if len(known) == 1 else None,
                known_pairs=len(known), registered_pairs=len(attempts),
                status='ok' if len(known) == len(attempts) == 1 else
                       'insufficient_response_grid' if not known else
                       'multiple_or_incomplete_pair_estimates')
