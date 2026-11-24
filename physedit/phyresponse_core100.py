"""Scoring helpers for the 4,000-video PhyResponse Core100 release.

This module deliberately does not treat human yes/no/uncertain as correctness:
the reference event truth must be supplied separately and bound to the same
clip/event/window.  With no truth file every accuracy field is blocked rather
than fabricated.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASKS = ROOT / 'review/phyresponse_gold4000_20261106/configured/tasks.jsonl'
DEFAULT_HUMAN = ROOT / 'review/phyresponse_gold4000_20261106/export_20261110/human_final.jsonl'
TRUTH_DIR = DEFAULT_TASKS.parents[1] / 'reference_event_truth_20261110'
DEFAULT_TRUTH = TRUTH_DIR / 'network_recovery_20261111/REFERENCE_EVENTS_STATE.jsonl'
DEFAULT_REFERENCE_SUPPLEMENT = TRUTH_DIR / 'event_visual_review_20261111/REFERENCE_SUPPLEMENT_ACTIVE.jsonl'
DEFAULT_OBSERVATIONS = TRUTH_DIR / 'BOUND_PRIMARY_OBSERVATIONS.jsonl'
DEFAULT_BOUNDARY = ROOT / 'benchmarks/phyresponse_core100_v1_20261110/BOUNDARY_REGISTRY.json'
DEFAULT_PAIRS = ROOT / 'benchmarks/phyresponse_core100_v1_20261110/PAIR_REGISTRY.json'
DEFAULT_LOGIC = TRUTH_DIR / 'evidence_completion_20261111/BOOLEAN_RULES.json'


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def target_events(task):
    ids = task.get('event_ids') or []
    if ids:
        return list(ids)
    return list(((task.get('event_contract') or {}).get('events') or {}).keys())


def human_label(human, event):
    if not human or human.get('measurability') != 'measurable':
        return None
    value = (human.get('events') or {}).get(event)
    return {'yes': 1, 'no': 0}.get(value)


def join_tasks_and_human(tasks, human_rows):
    by_id = {row.get('task_id'): row for row in human_rows}
    matched, missing = [], []
    for task in tasks:
        row = by_id.get(task.get('id'))
        if not row:
            missing.append(task)
            continue
        payload = row.get('payload') or {}
        if payload.get('video_sha256') != task.get('video_sha256'):
            missing.append(task)
            continue
        matched.append((task, payload))
    return matched, missing


def target_accuracy(matched, truth):
    """Per-task target accuracy; None means truth or human label missing."""
    rows = []
    for task, human in matched:
        for event in target_events(task):
            reference = (truth.get(task['clip_id']) or {}).get(event)
            label = human_label(human, event)
            correct = (int(label == reference)
                       if reference in (0, 1) and label in (0, 1) else None)
            rows.append(dict(
                task_id=task.get('id'), clip_id=task.get('clip_id'),
                model=task.get('model'), prompt_type=task.get('prompt_type'),
                seed=task.get('seed'), event=event,
                human_label=label, reference_label=reference,
                correct=correct, measurability=human.get('measurability'),
                unknown_reason=('human_' + str(human.get('measurability'))
                                if label is None else
                                'reference_truth_missing' if reference not in (0, 1) else None)))
    return rows


def _summary(rows):
    binary = [row for row in rows if row['correct'] is not None]
    return dict(
        task_events=len(rows), binary=binary.__len__(),
        coverage=len(binary) / len(rows) if rows else None,
        accuracy=sum(row['correct'] for row in binary) / len(binary) if binary else None,
        conservative_accuracy=(sum(row['correct'] for row in binary) / len(rows)
                               if rows else None),
        unknown=len(rows) - len(binary))


def by_model_prompt(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row['model'], row['prompt_type'])].append(row)
    return [dict(model=model, prompt_type=prompt, **_summary(rows))
            for (model, prompt), rows in sorted(groups.items())]


def near_accuracy(rows, boundary):
    """Point-balanced NearAcc; each clip is averaged over tasks before classes."""
    by_point = defaultdict(list)
    for row in rows:
        by_point[(row['model'], row['prompt_type'], row['clip_id'], row['event'])].append(row)
    points = []
    for (model, prompt, clip, event), own in by_point.items():
        info = (boundary.get(clip) or {}).get(event) or {}
        if info.get('bin') != 'near':
            continue
        binary = [row['correct'] for row in own if row['correct'] is not None]
        if not binary:
            continue
        points.append(dict(model=model, prompt_type=prompt, clip_id=clip, event=event,
                           reference_label=info.get('reference_label'),
                           accuracy=sum(binary) / len(binary)))
    result = []
    for model in sorted({point['model'] for point in points}):
        for prompt in sorted({point['prompt_type'] for point in points}):
            own = [point for point in points
                   if point['model'] == model and point['prompt_type'] == prompt]
            sides = [[point['accuracy'] for point in own
                      if point['reference_label'] == label] for label in (0, 1)]
            result.append(dict(model=model, prompt_type=prompt,
                               near_points=len(own),
                               counts={'negative': len(sides[0]), 'positive': len(sides[1])},
                               balanced_accuracy=(sum(sum(side) / len(side) for side in sides) / 2
                                                  if all(sides) else None)))
    return result


def pair_accuracy(rows, pairs):
    """PairAcc on same model/prompt/seed endpoints; no cross-prompt pairing."""
    by_key = defaultdict(dict)
    for row in rows:
        by_key[(row['model'], row['prompt_type'], row['seed'])][(row['clip_id'], row['event'])] = row['correct']
    result = []
    for pair in pairs:
        clips = pair.get('clips') or []
        event = pair.get('event')
        if len(clips) != 2 or not event:
            continue
        for (model, prompt, seed), values in sorted(by_key.items()):
            left = values.get((clips[0], event))
            right = values.get((clips[1], event))
            result.append(dict(pair_id=pair.get('pair_id'), model=model, prompt_type=prompt,
                               seed=seed, left=left, right=right,
                               correct=(int(left) * int(right))
                               if left in (0, 1) and right in (0, 1) else None))
    known = [row['correct'] for row in result if row['correct'] is not None]
    return dict(rows=result, eligible=len(result), known=len(known),
                accuracy=sum(known) / len(known) if known else None,
                coverage=len(known) / len(result) if result else None)


def load_truth(path):
    rows = read_jsonl(path)
    truth = defaultdict(dict)
    for row in rows:
        if row.get('status', 'registered') != 'registered':
            continue
        for event, value in (row.get('events') or {}).items():
            if value in (0, 1):
                truth[row['clip_id']][event] = value
    return dict(truth)


def load_boundary(path):
    if not Path(path).exists():
        return {}
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if isinstance(data, list):
        data = {row['clip_id']: row for row in data if row.get('clip_id')}
    return data


def digest_json(value):
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False,
                         separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def reference_key(task, event):
    return (task['clip_id'], event, task['reference_video_sha256'],
            task['event_contract_sha256'], digest_json(task['window']))


def index_native_truth(records):
    index = {}
    for row in records:
        key = tuple(row[field] for field in ('case_id', 'event', 'reference_video_sha256',
                                           'event_contract_sha256', 'window_sha256'))
        if key in index:
            raise ValueError(f'Duplicate reference event window: {key}')
        if row['status'] == 'verified':
            if type(row['reference_label']) is not int or row['reference_label'] not in (0, 1):
                raise ValueError(f'Invalid verified reference label: {key}')
            certificate = row.get('reference_certificate')
            if certificate and certificate.get('schema') == 'phyresponse-visible-reference-revision-v1':
                from .visible_reference_revision import validate
                validate(row, digest_json)
            elif certificate and certificate.get('schema') == 'phyresponse-window-projection-v1':
                from .reference_window_projection import validate
                validate(row, digest_json)
            elif certificate and certificate.get('schema') == 'phyresponse-visible-support-reference-v9':
                from .visible_t10_reference import validate
                validate(row, digest_json)
            elif certificate and certificate.get('schema') == 'phyresponse-published-group-oracle-v1':
                from .source_group_reference import validate
                validate(row, digest_json)
            elif certificate and certificate.get('schema') == 'phyresponse-source-video-review-v1':
                from physedit.reference_video_evidence import validate_source_video_reference
                validate_source_video_reference(row, digest_json)
            elif certificate and certificate.get('schema') == 'phyresponse-native-source-contact-review-v1':
                from physedit.reference_video_evidence import validate_native_source_contact_reference
                validate_native_source_contact_reference(row, digest_json)
            elif certificate is not None:
                bound = (*REFERENCE_FIELDS, 'reference_label')
                if (certificate.get('schema') != 'phyresponse-retained-native-contact-v1'
                        or certificate.get('status') != 'verified'
                        or any(certificate.get(f) != row.get(f) for f in bound)
                        or digest_json(certificate) != row.get('reference_evidence_sha256')
                        or not certificate.get('qa_sha256')
                        or not certificate.get('production_receipt_sha256')
                        or not certificate.get('original_qa_source_sha256')
                        or not certificate.get('expected_original_state_sha256')
                        or certificate.get('state_materialized') is not False):
                    raise ValueError(f'Invalid retained native reference certificate: {key}')
                first = certificate.get('native_first_contact_physics_time_s')
                fps, frames = certificate.get('source_fps'), certificate.get('source_frames')
                start, end = certificate.get('window_start_s'), certificate.get('window_end_s')
                if (fps != 30 or frames != 210 or start != 0 or end != row.get('window_end_s')
                        or not isinstance(end, (int, float)) or not math.isfinite(end)
                        or certificate.get('original_contact_predicate') != 'norm(contact.impulse_Ns) > 1e-8 N*s'
                        or certificate['original_qa_source_sha256'] != 'c16cbe1e699772371d982e3a845f1715c645f6e1c485ab139db8403f2eda19a6'
                        or certificate.get('original_renderer_sha256') != 'feb6633cfb7c963440f9f87edaa573f754ceafb1f324dc4f3ea04ce0e4d2958a'
                        or (row['reference_label'] == 0 and (first is not None or end > (frames - 1) / fps))
                        or (row['reference_label'] == 1 and
                            (type(first) not in (int, float) or not math.isfinite(first) or first < 0
                             or end < (math.floor(first * fps) + 1) / fps))):
                    raise ValueError(f'Retained reference lacks complete evidence for its window: {key}')
            if not row.get('evidence') or (not row.get('state_sha256') and certificate is None):
                raise ValueError(f'Verified reference lacks state or qualified evidence: {key}')
        elif row.get('reference_label') is not None:
            raise ValueError(f'Unverified reference has a label: {key}')
        elif (row.get('reference_certificate') or {}).get('schema') == 'phyresponse-source-video-review-v1':
            from physedit.reference_video_evidence import validate_source_video_reference
            validate_source_video_reference(row, digest_json)
        index[key] = row
    return index


def merge_native_reference_supplement(primary, supplement):
    """Replace unresolved windows only after validating the supplemental evidence."""
    result = dict(primary)
    for key, row in index_native_truth(supplement).items():
        if (key not in result or result[key]['status'] == 'verified'
                or row['status'] not in ('verified', 'unresolved')
                or (row['status'] == 'unresolved' and
                    row.get('reference_certificate', {}).get('schema') != 'phyresponse-source-video-review-v1')):
            raise ValueError(f'Reference supplement cannot replace this window: {key}')
        if not row.get('reference_certificate'):
            raise ValueError(f'Reference supplement requires a bound certificate: {key}')
        result[key] = row
    return result


def native_reference_coverage(tasks, truth):
    expected = {reference_key(task, event) for task in tasks for event in target_events(task)}
    for task in tasks:
        policy=task.get('event_contract',{}).get('terminal_sampling_policy')
        policy_sha=task.get('event_contract',{}).get('terminal_sampling_policy_sha256')
        if policy is not None and (digest_json(policy)!=policy_sha
                or policy.get('schema')!='phyresponse-native-terminal-sampling-v3'
                or policy.get('rule')!='last_native_sample_inside_window_v3'
                or policy.get('interpolation') is not False):
            raise ValueError('Terminal sampling policy hash or rule mismatch')
        for event in target_events(task):
            record=truth.get(reference_key(task,event),{})
            cert=(record.get('reference_certificate') or {})
            if (policy is not None and record.get('status')=='verified'
                    and record.get('terminal_sampling_policy_sha256')!=policy_sha):
                raise ValueError('Reference does not bind the active terminal sampling policy')
            if (cert.get('schema') in ('phyresponse-visible-reference-revision-v1','phyresponse-window-projection-v1','phyresponse-published-group-oracle-v1','phyresponse-visible-support-reference-v9')
                    and cert.get('event_spec_sha256')!=digest_json(task['event_contract']['events'][event])):
                raise ValueError('Projected reference event definition differs from active task')
    statuses = defaultdict(int)
    by_case = defaultdict(list)
    for key in expected:
        record = truth.get(key)
        status = record['status'] if record else 'missing_reference_record'
        statuses[status] += 1
        by_case[key[0]].append(status)
    return dict(cases=len(by_case), complete_cases=sum(set(status) == {'verified'} for status in by_case.values()),
                expected_event_windows=len(expected), statuses=dict(sorted(statuses.items())))


REFERENCE_FIELDS = ('case_id', 'event', 'reference_video_sha256',
                    'event_contract_sha256', 'window_sha256')


def load_native_registries(boundary_path, pair_path, tasks, truth):
    """Validate explicit admission records against active task and truth identities."""
    expected = {reference_key(task, event): task['event_contract']['events'][event]
                for task in tasks for event in target_events(task)}

    def load(path, schema):
        if path is None:
            return []
        data = json.loads(path.read_text(encoding='utf-8-sig'))
        if data.get('schema') != schema or not isinstance(data.get('records'), list):
            raise ValueError(f'Invalid native registry schema: {path}')
        return data['records']

    points = {}
    for row in load(boundary_path, 'phyresponse-native-boundaries-v1'):
        key = tuple(row[field] for field in REFERENCE_FIELDS)
        reference = truth.get(key)
        if (key in points or key not in expected or not reference
                or reference['status'] != 'verified' or row.get('status') != 'admitted'
                or type(row.get('reference_label')) is not int
                or row['reference_label'] != reference['reference_label']
                or row.get('state_sha256') != reference['state_sha256']
                or row.get('reference_evidence_sha256') != reference.get('reference_evidence_sha256')
                or not row.get('context_sha256') or not row.get('axis') or not row.get('evidence')):
            raise ValueError(f'Unbound or unadmitted boundary: {key}')
        if (reference.get('reference_certificate') and
                reference['reference_certificate'].get('event_spec_sha256') != digest_json(expected[key])):
            raise ValueError(f'Retained boundary event definition mismatch: {key}')
        lo, hi = row['interval']
        theta, span = row['theta'], row['normalization_span']
        if (not all(type(x) in (int, float) and math.isfinite(x) for x in (lo, hi, theta, span))
                or lo > hi or span <= 0):
            raise ValueError(f'Invalid numerical interval: {key}')
        lower = max(lo - theta, theta - hi, 0) / span
        if row.get('calibration_certificate'):
            from .archived_calibration import validate_certificate
            validate_certificate(row)
        upper = max(abs(theta - lo), abs(theta - hi)) / span
        if upper == 0:
            bucket = 'at_boundary'
        elif upper <= .05:
            bucket = 'near'
        elif lower > .05 and upper <= .15:
            bucket = 'mid'
        elif lower > .15:
            bucket = 'far'
        else:
            bucket = 'uncertain'
        points[key] = dict(row, bin=bucket, normalized_distance=[lower, upper])

    pairs = {}
    for row in load(pair_path, 'phyresponse-native-pairs-v1'):
        pair_id = row['pair_id']
        left, right = [tuple(ref[field] for field in REFERENCE_FIELDS) for ref in row['endpoints']]
        if (not pair_id or pair_id in pairs or left[0] == right[0] or left[1] != right[1]
                or left[4] != right[4] or left not in points or right not in points
                or row.get('status') != 'admitted' or row.get('mode') != 'coupled_seed'
                or not row.get('evidence')):
            raise ValueError(f'Unbound or unadmitted pair: {pair_id}')
        a, b = points[left], points[right]
        shared = ('context_sha256', 'axis', 'interval', 'normalization_span')
        lo, hi = a['interval']
        low_point, high_point = sorted((a['theta'], b['theta']))
        worst_imbalance = max(abs(2 * x - low_point - high_point) for x in (lo, hi)) / a['normalization_span']
        if (any(a[field] != b[field] for field in shared)
                or expected[left] != expected[right]
                or a['reference_label'] == b['reference_label']
                or low_point >= high_point or not low_point <= lo <= hi <= high_point
                or worst_imbalance > .02 + 1e-12):
            raise ValueError(f'Pair fails the cross-boundary protocol: {pair_id}')
        pairs[pair_id] = (left, right)
    return points, pairs


def score_native_boundary(tasks, rows, points, pairs):
    from .native_critical import evaluate
    metrics, pair_result, _ = evaluate(tasks, rows, points, pairs, reference_key)
    return metrics, pair_result


def load_native_equivalences(path, tasks, observations):
    if path is None:
        return {}
    data = json.loads(path.read_text(encoding='utf-8-sig'))
    if data.get('schema') != 'phyresponse-observation-equivalence-v1':
        raise ValueError('Invalid observation equivalence schema')
    by_task = {task['id']: task for task in tasks}
    by_observation = {row['id']: row for row in observations}
    result = {}
    for row in data['records']:
        key = row['task_id'], row['event']
        task = by_task.get(key[0])
        observed = by_observation.get(key[0])
        if task is None or observed is None or key in result:
            raise ValueError(f'Unbound observation equivalence: {key}')
        question_id = row.get('question_id', 'obs_' + key[1])
        question = observed.get('questions', {}).get('events', {}).get(question_id, {})
        spec = task['event_contract']['events'].get(key[1], {})
        implication = row.get('kind') == 'necessary_component_negative'
        if implication:
            component = row.get('component_id')
            definition = spec.get('component_definitions', {}).get(component)
            permitted = {
                ('T01', 'full_chain', 'can2_mug_contact', 'can_2_mug_contact'):
                    'cb1145a9c810b41e83f48f03b4df4119ba59222315f1b2cb7a9638708d76ec9c',
                ('T14', 'terminal_contact_after_full_exit', 'marker_eraser_contact', 'contacted_eraser'):
                    'b331152cc13b9883435efb66bc830fabc53419b011da1acf05d39a01bfd399dc',
            }
            pin = permitted.get((task['clip_id'].split('_')[0], key[1], question_id, component))
            if (pin is None or digest_json(question) != pin
                    or component not in spec.get('source_conjunction', {}).get('all', [])
                    or not definition or row.get('component_spec_sha256') != digest_json(definition)
                    or row.get('direction') != 'negative_only'):
                raise ValueError(f'Invalid necessary-component implication: {key}')
        if (row.get('status') != 'verified' or not row.get('evidence')
                or row.get('event_contract_sha256') != task['event_contract_sha256']
                or row.get('observation_contract_sha256') != observed.get('observation_contract_sha256')
                or row.get('window_sha256') != digest_json(task['window'])
                or row.get('question_sha256') != digest_json(question)
                or row.get('source_event_definition_sha256') != digest_json(spec)
                or question.get('relation') != 'source_component'
                or question.get('original_event_ids') != [key[1]]
                or (not implication and set(question.get('actors', [])) != set(spec.get('actors', [])))
                or question.get('prerequisites')):
            raise ValueError(f'Invalid observation equivalence: {key}')
        result[key] = row
    return result


def load_historical_human_contracts(path, tasks):
    """Bind narrowly reviewed full-video T06 questions to their exact source."""
    if path is None:
        return {}
    from .native_observation_logic import digest, profile
    data = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if data.get('schema') != 'phyresponse-exact-historical-observation-contracts-v1':
        raise ValueError('Invalid historical observation contract registry')
    source = Path(data['source_path'])
    if hashlib.sha256(source.read_bytes()).hexdigest() != data['source_sha256']:
        raise ValueError('Historical observation source hash mismatch')
    wanted = {r['id'] for r in data['records']}
    source_rows = {r['id']: r for r in read_jsonl(source) if r['id'] in wanted}
    by_task = {t['id']: t for t in tasks}
    result = {}
    expected_definition = '原登记的每一对相邻书本均确认发生接触，共9对：第1与第2本，依次直到第9与第10本。不额外添加倾角、位移或倒下数量阈值。'
    for record in data['records']:
        task = by_task.get(record['id'])
        if task is None or record['id'] in result or source_rows.get(record['id']) != record:
            raise ValueError('Unbound historical observation source record')
        event = 'complete_required_contact_chain'
        contract = record['observation_contract']
        question = contract.get('events', {}).get('obs_' + event, {})
        window = task['window']
        if (not task['clip_id'].startswith('T06_') or
                any(task.get(k) != record.get(k) for k in ('video_sha256', 'prompt_sha256', 'event_contract_sha256')) or
                task['event_contract'] != record.get('event_contract') or
                contract.get('original_contract_sha256') != task['event_contract_sha256'] or
                digest(contract) != record.get('observation_contract_sha256') or
                question.get('relation') != 'visible_original_event' or
                question.get('original_event_id') != event or question.get('definition') != expected_definition or
                profile(task, event) != {'contacts': [f'adjacent_contact_{i}' for i in range(9)], 'complete': True} or
                window.get('start_frame') != 0 or window.get('end_frame') != task['frames'] - 1 or
                window.get('fps') != task['fps']):
            raise ValueError('Historical full-video observation binding mismatch')
        result[record['id']] = record
    return result


def score_native_observations(tasks, human_rows, observation_rows, truth, equivalences=None, logic_rules=None,
                              historical_contracts=None):
    """Compare bound direct labels or a certified, logically decisive component."""
    equivalences = equivalences or {}
    logic_rules = logic_rules or {}
    historical_contracts = historical_contracts or {}
    observations = {row['id']: row for row in observation_rows}
    humans = {row['task_id']: row for row in human_rows if row.get('stage') == 'primary'}
    rows = []
    for task in tasks:
        task_id = task['id']
        observed = observations.get(task_id, {})
        human = humans.get(task_id, {}).get('payload', {})
        binding_fields = ('video_sha256', 'prompt_sha256', 'event_contract_sha256')
        bound = (all(task[field] == observed.get(field) == human.get(field) for field in binding_fields)
                 and observed.get('window') == task['window']
                 and bool(observed.get('observation_contract_sha256'))
                 and observed['observation_contract_sha256'] == human.get('observation_contract_sha256'))
        questions = observed.get('questions', {}).get('events', {})
        for event in target_events(task):
            question_id = 'obs_' + event
            question = questions.get(question_id, {})
            equivalent = (question.get('relation') == 'visible_original_event'
                          and event in question.get('original_event_ids', []))
            rule = equivalences.get((task_id, event), {})
            implication = rule.get('kind') == 'necessary_component_negative'
            equivalent = equivalent or (bool(rule) and not implication)
            record = truth.get(reference_key(task, event), {})
            reference = record.get('reference_label') if record.get('status') == 'verified' else None
            # Certified equivalences may refer to a saved source-graph question
            # whose ID is not the scorer's default obs_<event> name.
            label_question_id = rule.get('question_id', question_id) if rule and not implication else question_id
            label = human_label(human, label_question_id) if bound and equivalent else None
            label_basis = 'direct_event_observation'
            if bound and implication:
                component_label = human_label(human, rule['question_id'])
                # A AND B is false if A is explicitly false. A=true or unknown
                # does not determine B, and must never produce an event label.
                if component_label == 0:
                    label = 0
                    equivalent = True
                    label_basis = 'necessary_component_explicit_no'
            logic_evidence=[]
            logic=logic_rules.get((task_id,event))
            if bound and logic and label is None:
                from .native_observation_logic import infer
                derived,logic_evidence=infer(logic['profile'],human)
                if derived is not None:
                    label=derived
                    equivalent=True
                    label_basis='frozen_boolean_components'
            historical = historical_contracts.get(task_id)
            historical_bound = bool(historical and event == 'complete_required_contact_chain' and
                all(task[field] == observed.get(field) == human.get(field) == historical.get(field)
                    for field in binding_fields) and observed.get('window') == task['window'] and
                human.get('observation_contract_sha256') == historical['observation_contract_sha256'])
            if not bound and historical_bound:
                label = human_label(human, 'obs_' + event)
                if label is not None:
                    equivalent = True
                    label_basis = 'exact_historical_full_video_observation'
            if not human:
                reason = 'human_annotation_missing'
            elif not bound and not historical_bound:
                reason = 'human_binding_mismatch'
            elif human.get('measurability') == 'unmeasurable':
                reason = 'human_unmeasurable_or_uncertain'
            elif (label is None and logic and any(
                    human.get('events', {}).get(q) == 'uncertain'
                    for q in [*logic['profile']['contacts'],
                              *([logic['profile']['order']] if logic['profile'].get('order') else [])])):
                reason = 'human_unmeasurable_or_uncertain'
            elif not equivalent:
                reason = 'no_direct_visible_equivalent'
            elif reference is None:
                reason = 'reference_unverified'
            elif label is None:
                reason = 'human_unmeasurable_or_uncertain'
            else:
                reason = None
            rows.append(dict(task_id=task_id, clip_id=task['clip_id'], event=event,
                             model=task['model'], prompt_type=task['prompt_type'],
                             reference_label=reference, human_label=label,
                             generated_event_label_basis=label_basis if label is not None else None,
                             derived_component_evidence=logic_evidence,
                             correct=int(label == reference) if reason is None else None,
                             unknown_reason=reason))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks', type=Path, default=DEFAULT_TASKS)
    parser.add_argument('--human', type=Path, default=DEFAULT_HUMAN)
    parser.add_argument('--truth', type=Path, default=DEFAULT_TRUTH)
    parser.add_argument('--reference-supplement', type=Path,
                        help='Qualified original metadata or bound source-video review for unresolved windows')
    parser.add_argument('--observations', type=Path, default=DEFAULT_OBSERVATIONS)
    parser.add_argument('--boundary', type=Path)
    parser.add_argument('--pairs', type=Path)
    parser.add_argument('--missing-slots', type=Path, help='Explicit missing generation registry; auto-loaded only for default release')
    parser.add_argument('--equivalence', type=Path)
    parser.add_argument('--logic', type=Path, help='Hash-bound Boolean composition of original human components')
    parser.add_argument('--historical-observations', type=Path, help='Verified exact historical human observation contracts')
    parser.add_argument('--cohort', type=Path, help='Hash-bound complete original case roster and event admission')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--report-dir', type=Path,
                        help='Export all four model/prompt metrics, cohort scope, reference exclusions and review queues')
    parser.add_argument('--allow-partial', action='store_true',
                        help='Score verified native event windows and report incomplete reference coverage')
    # A registered protocol version is the default CLI entry. Explicit input
    # arguments retain reproducible historical/custom runs and test fixtures.
    profile_path=ROOT/'benchmarks/phyresponse_core100_v1_20261110/ACTIVE_SCORING.json'
    input_flags={'--tasks','--human','--truth','--observations','--boundary','--pairs',
                 '--reference-supplement','--equivalence','--logic','--missing-slots','--cohort','--historical-observations'}
    if profile_path.is_file() and not any(a.split('=')[0] in input_flags for a in sys.argv[1:]):
        profile=json.loads(profile_path.read_text(encoding='utf-8'))
        if profile.get('schema')!='phyresponse-active-scoring-profile-v1':
            raise ValueError('Invalid active PhyResponse scoring profile')
        defaults={}
        for flag,record in profile['inputs'].items():
            if '--'+flag not in input_flags:raise ValueError('Unknown active scoring input')
            path=ROOT/record['path']
            if hashlib.sha256(path.read_bytes()).hexdigest()!=record['sha256']:
                raise ValueError('Active scoring input changed: '+str(path))
            defaults[flag.replace('-','_')]=path
        defaults['allow_partial']=profile.get('score_verified_windows',False)
        parser.set_defaults(**defaults)
    args = parser.parse_args()
    # Abort rather than publish a mixed snapshot if a concurrent evidence job writes inputs.
    candidates = [args.tasks, args.human, args.truth, args.observations, args.cohort,
                  args.boundary or DEFAULT_BOUNDARY, args.pairs or DEFAULT_PAIRS,
                  args.reference_supplement or DEFAULT_REFERENCE_SUPPLEMENT,
                  args.equivalence or TRUTH_DIR / 'network_recovery_20261111/OBSERVATION_RULES.json',
                  args.missing_slots or DEFAULT_TASKS.parents[1] / 'missing_videos.json']
    logic_path=args.logic
    if (logic_path is None and args.tasks.resolve()==DEFAULT_TASKS.resolve()
            and args.observations.resolve()==DEFAULT_OBSERVATIONS.resolve() and DEFAULT_LOGIC.is_file()):
        logic_path=DEFAULT_LOGIC
    if logic_path is not None:candidates.append(logic_path)
    historical_source_path = None
    if args.historical_observations is not None:
        candidates.append(args.historical_observations)
        historical_source_path = Path(json.loads(args.historical_observations.read_text(encoding='utf-8-sig'))['source_path'])
        candidates.append(historical_source_path)
    initial_hashes = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in candidates if p is not None and p.is_file()}
    tasks = read_jsonl(args.tasks)
    if not tasks:
        parser.error('A non-empty PhyResponse task manifest is required')
    truth = index_native_truth(read_jsonl(args.truth))
    supplement_path = args.reference_supplement
    if (supplement_path is None and args.truth.resolve() == DEFAULT_TRUTH
            and args.tasks.resolve() == DEFAULT_TASKS and DEFAULT_REFERENCE_SUPPLEMENT.is_file()):
        supplement_path = DEFAULT_REFERENCE_SUPPLEMENT
    supplemental_records = read_jsonl(supplement_path) if supplement_path else []
    truth = merge_native_reference_supplement(truth, supplemental_records)
    boundary_path = args.boundary or (DEFAULT_BOUNDARY if DEFAULT_BOUNDARY.is_file() else None)
    pair_path = args.pairs or (DEFAULT_PAIRS if DEFAULT_PAIRS.is_file() else None)
    points, pairs = load_native_registries(boundary_path, pair_path, tasks, truth)
    coverage = native_reference_coverage(tasks, truth)
    complete = coverage['complete_cases'] == coverage['cases'] and coverage['expected_event_windows'] > 0
    observations = read_jsonl(args.observations)
    equivalence_path = args.equivalence
    default_equivalence = TRUTH_DIR / 'network_recovery_20261111/OBSERVATION_RULES.json'
    if (equivalence_path is None and args.tasks.resolve() == DEFAULT_TASKS
            and args.observations.resolve() == DEFAULT_OBSERVATIONS and default_equivalence.is_file()):
        equivalence_path = default_equivalence
    equivalences = load_native_equivalences(equivalence_path, tasks, observations)
    from .native_observation_logic import load_rules
    logic_rules=load_rules(logic_path,tasks,observations)
    historical_contracts=load_historical_human_contracts(args.historical_observations,tasks)
    scoring_enabled = complete or args.allow_partial
    # Unrelated incomplete reference windows do not block an independently
    # admitted Near/Mid/Far point. Wide event scoring retains its explicit gate.
    boundary_scoring_enabled = bool(points)
    rows = (score_native_observations(tasks, read_jsonl(args.human), observations, truth, equivalences,logic_rules,historical_contracts)
            if scoring_enabled or boundary_scoring_enabled else None)
    from .native_critical import evaluate
    missing_path = args.missing_slots
    if missing_path is None and args.tasks.resolve() == DEFAULT_TASKS.resolve():
        missing_path = DEFAULT_TASKS.parents[1] / 'missing_videos.json'
    missing = json.loads(missing_path.read_text(encoding='utf-8-sig')) if missing_path else []
    metrics, pair_score, critical = evaluate(tasks, rows or [], points, pairs, reference_key, missing)

    report = dict(
        tasks=len(tasks), reference_coverage=coverage,
        scoring_scope=('complete_reference' if complete else
                       'verified_native_event_windows' if args.allow_partial else
                       'admitted_native_boundary_points' if boundary_scoring_enabled else 'blocked'),
        target_accuracy=(dict(overall={k:v for k,v in _summary(rows).items() if k != 'conservative_accuracy'},
                              by_model_prompt=[{k:v for k,v in r.items() if k != 'conservative_accuracy'}
                                               for r in by_model_prompt(rows)]) if scoring_enabled else None),
        task_events=rows if scoring_enabled else None,
        boundary_scoring_scope='admitted_native_points' if boundary_scoring_enabled else None,
        near_accuracy=metrics.get('near') or None, mid_accuracy=metrics.get('mid') or None,
        far_accuracy=metrics.get('far') or None, pair_accuracy=pair_score if pairs else None,
        formal_boundary_admitted_event_windows=len(points), formal_pairs_admitted=len(pairs),
        reviewed_observation_equivalences=len(equivalences),
        reviewed_boolean_component_rules=len(logic_rules),
        retained_native_reference_windows=sum((r.get('reference_certificate') or {}).get('schema') == 'phyresponse-retained-native-contact-v1' for r in supplemental_records),
        source_video_reference_windows_verified=sum(r['status'] == 'verified' and r.get('reference_basis') == 'assistant_source_video_review' for r in supplemental_records),
        source_video_review_unresolved_windows=sum(r['status'] == 'unresolved' and r.get('reference_basis') == 'assistant_source_video_review' for r in supplemental_records),
        status=('partial_reference_scoring' if args.allow_partial and not complete else
                'admitted_boundary_scoring_reference_incomplete' if boundary_scoring_enabled and not complete else
                'blocked_reference_truth_incomplete' if not complete else
                'ready' if points and pairs else 'reference_complete_boundary_pending'))
    report.update(critical)
    from .cohort_protocol import validate as validate_cohort, event_vectors
    if args.cohort:
        report['cohort_registration'] = validate_cohort(args.cohort,tasks,truth,missing,reference_key)
        report['status'] = 'full_cohort_registered_partial_critical_coverage'
    report['event_exact'] = event_vectors(tasks,rows or [],missing)
    if scoring_enabled:
        report['target_accuracy'] = critical['all_case_event_accuracy']
    report['schema'] = 'phyresponse-native-critical-v2'
    inputs = dict(tasks=args.tasks, human=args.human, truth=args.truth,
                  observations=args.observations, boundary=boundary_path, pairs=pair_path,
                  missing_slots=missing_path, supplement=supplement_path, equivalence=equivalence_path)
    inputs['logic']=logic_path
    inputs['historical_observations']=args.historical_observations
    inputs['historical_observations_source']=historical_source_path
    inputs['cohort']=args.cohort
    report['input_sha256'] = {name: dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                              for name,path in inputs.items() if path is not None}
    for name,path in inputs.items():
        if path is not None and initial_hashes.get(str(path.resolve())) != report['input_sha256'][name]['sha256']:
            raise RuntimeError(f'Input changed during scoring: {path}')
    report['input_snapshot_verified'] = True
    from .native_metric_report import build, export
    report['formal_metric_report'] = build(report, tasks, truth, points, pairs)
    from .prompt_effects import compare
    report['prompt_effects'] = compare(report)
    if args.report_dir:
        export(report, args.report_dir)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
