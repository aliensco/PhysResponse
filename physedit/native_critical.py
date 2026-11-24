"""Native-release critical metrics. No legacy distances or labels are read."""
from collections import Counter, defaultdict
import hashlib
import json
from .critical_metrics import boundary_error_estimate


def response_grid_diagnostics(curves, event_specs=None):
    """Compare fully observed native grids without imputing unknown attempts."""
    groups = defaultdict(list)
    for row in curves:
        interval = row.get('reference_boundary_interval')
        spec = event_specs.get(tuple(row['reference_key'])) if event_specs is not None else 'unchecked'
        if (row.get('context_sha256') and row.get('axis') and
                row.get('theta') is not None and interval and
                row.get('normalization_span') and spec is not None):
            key = (row['context_sha256'], row['event'], row['window_sha256'],
                   row['axis'], tuple(interval), row['normalization_span'],
                   row['model'], row['prompt_type'], spec)
            groups[key].append(row)
    results = []
    for key, rows in sorted(groups.items()):
        if len({r['theta'] for r in rows}) < 3:
            continue
        rows = sorted(rows, key=lambda r: r['theta'])
        known = [r for r in rows if r['expected'] > 0 and r['final'] == r['expected']
                 and r.get('p_unknown', 0) == 0]
        estimate = boundary_error_estimate(known, list(key[4]))
        ref_cross = [(a, b) for a, b in zip(rows, rows[1:])
                     if a['reference_label'] != b['reference_label']]
        model_cross = [(a, b) for a, b in zip(known, known[1:])
                       if (a['p1'] - .5) * (b['p1'] - .5) < 0]
        direction = 'unavailable'
        if len(ref_cross) == len(model_cross) == 1:
            a, b = ref_cross[0]; c, d = model_cross[0]
            direction = 'same' if (b['reference_label'] - a['reference_label']) * (d['p1'] - c['p1']) > 0 else 'opposite'
        status = estimate['status']
        if status == 'ok':
            status = ('single_observation_diagnostic' if min(r['final'] for r in known) == 1 else 'observed_grid_diagnostic') if direction == 'same' else 'direction_mismatch' if direction == 'opposite' else 'direction_unavailable'
        accepted = status in ('single_observation_diagnostic', 'observed_grid_diagnostic')
        results.append(dict(context_sha256=key[0], event=key[1], window_sha256=key[2],
                            axis=key[3], reference_interval=list(key[4]), normalization_span=key[5],
                            model=key[6], prompt_type=key[7],
                            reference_keys=[r['reference_key'] for r in rows],
                            theta=[r['theta'] for r in rows],
                            observed_counts=[r['final'] for r in rows],
                            expected_counts=[r['expected'] for r in rows],
                            known_theta=len({r['theta'] for r in known}), status=status,
                            direction=direction,
                            theta_hat=estimate['theta_hat'] if accepted else None,
                            error_lower=estimate['error_lower'] if accepted else None,
                            error_upper=estimate['error_upper'] if accepted else None))
    return dict(accuracy=None, status='diagnostic_grids_only' if results else 'native_response_grid_not_registered',
                registered_grids=len(results), groups=results,
                scope='Exploratory observed grids; partial and single-sample grids do not establish a formal aggregate.')


def summary(values):
    known = [v for v in values if v is not None]
    rate = sum(known) / len(known) if known else None
    return dict(expected=len(values), final=len(known), task_events=len(values), binary=len(known), unknown=len(values)-len(known),
                coverage=len(known)/len(values) if values else None,
                accuracy=rate if values and len(known)==len(values) else None,
                known_only_accuracy=rate,
                status='complete' if values and len(known)==len(values) else
                       'incomplete' if values else 'no_registered_points')


def evaluate(tasks, rows, points, pairs, reference_key, missing=()):
    by_id = {t['id']: t for t in tasks}
    event_specs = {}
    for task in tasks:
        for event, definition in task['event_contract']['events'].items():
            key = reference_key(task, event)
            digest = hashlib.sha256(json.dumps(definition, sort_keys=True, ensure_ascii=False,
                                                 separators=(',', ':')).encode('utf-8')).hexdigest()
            if key in event_specs and event_specs[key] != digest:
                raise ValueError('Conflicting definitions for the same native event reference')
            event_specs[key] = digest
    observed = {}
    for row in rows:
        key = (row['task_id'], row['event'])
        if key in observed:
            raise ValueError('Duplicate task/event observation')
        observed[key] = row
    slots = {}
    group_windows = defaultdict(set)
    for task in tasks:
        slot = (task['clip_id'], task['model'], task['prompt_type'], task['seed'])
        if slot in slots:
            raise ValueError('Duplicate registered generation slot')
        slots[slot] = task
        for event in task.get('event_ids') or task['event_contract']['events']:
            group_windows[(task['model'], task['prompt_type'], task['seed'])].add(reference_key(task,event)[4])
    missing_slots = set()
    for row in missing:
        slot = tuple(row[k] for k in ('clip_id','model','prompt_type','seed'))
        if slot in slots or slot in missing_slots:
            raise ValueError('Duplicate or conflicting missing generation slot')
        missing_slots.add(slot)
    roster = set(slots) | missing_slots
    # All-case event ledger uses current contracts, never legacy case labels.
    case_events = defaultdict(set)
    for task in tasks:
        case_events[task['clip_id']].update(task.get('event_ids') or task['event_contract']['events'])
    event_values, event_reasons = [], Counter()
    slot_statuses = Counter()
    event_groups = defaultdict(list)
    slot_ledger = []
    for slot in sorted(roster):
        task = slots.get(slot)
        if task is None:
            slot_statuses['generation_missing'] += 1
            for event in case_events[slot[0]]:
                event_values.append(None)
                event_groups[(slot[1],slot[2])].append(None)
                event_reasons['generation_missing'] += 1
            slot_ledger.append(dict(clip_id=slot[0],model=slot[1],prompt_type=slot[2],seed=slot[3],
                                    status='generation_missing',task_id=None))
            continue
        own = [observed.get((task['id'],event), {})
               for event in task.get('event_ids') or task['event_contract']['events']]
        for row in own:
            event_values.append(row.get('correct'))
            event_groups[(slot[1],slot[2])].append(row.get('correct'))
            if row.get('correct') is None:
                event_reasons[row.get('unknown_reason') or 'observation_missing'] += 1
        if all(r.get('correct') is not None for r in own):
            status='final'
        elif any(r.get('unknown_reason') == 'human_unmeasurable_or_uncertain' for r in own):
            status='unjudgeable'
        else:
            status='pending_evidence_or_annotation'
        slot_statuses[status] += 1
        slot_ledger.append(dict(clip_id=slot[0],model=slot[1],prompt_type=slot[2],seed=slot[3],
                                task_id=task['id'],status=status,
                                event_summary=summary([r.get('correct') for r in own]),
                                unknown_reasons=dict(Counter(r.get('unknown_reason') or 'observation_missing' for r in own if r.get('correct') is None))))
    groups = sorted({s[1:] for s in roster})
    attempts = []
    for model,prompt,seed in groups:
        for key, point in sorted(points.items()):
            slot = (key[0],model,prompt,seed)
            if slot not in roster:
                continue
            task = slots.get(slot)
            if task:
                if key[1] not in (task.get('event_ids') or task['event_contract']['events']):
                    continue
                if reference_key(task,key[1]) != key:
                    continue
                row = observed.get((task['id'],key[1]), {})
                reason = row.get('unknown_reason') or (None if row.get('correct') is not None else 'observation_missing')
            else:
                # Window applicability only, never invent a missing clip's contract/label.
                if key[4] not in group_windows[(model,prompt,seed)]:
                    continue
                row, reason = {}, 'generation_missing'
            attempts.append(dict(key=key,model=model,prompt_type=prompt,seed=seed,
                                 clip_id=key[0],event=key[1],bin=point['bin'],
                                 reference_label=point['reference_label'],
                                 correct=row.get('correct'),human_label=row.get('human_label'),
                                 unknown_reason=reason))

    metrics = {b:[] for b in ('near','mid','far','uncertain','at_boundary')}
    curves = []
    for model,prompt in sorted({(g[0],g[1]) for g in groups}):
        cohort = [a for a in attempts if (a['model'],a['prompt_type'])==(model,prompt)]
        for bucket in metrics:
            own = [a for a in cohort if a['bin']==bucket]
            if not own:
                continue
            by_point = defaultdict(list)
            for a in own:
                by_point[a['key']].append(a)
            sides, known_sides = [[],[]], [[],[]]
            for key, aa in by_point.items():
                s=summary([a['correct'] for a in aa]); label=points[key]['reference_label']
                if s['accuracy'] is not None: sides[label].append(s['accuracy'])
                if s['known_only_accuracy'] is not None: known_sides[label].append(s['known_only_accuracy'])
            balanced=lambda ss: sum(sum(x)/len(x) for x in ss)/2 if all(ss) else None
            s=summary([a['correct'] for a in own])
            s.update(model=model,prompt_type=prompt,registered_points=len(by_point),
                     aggregation='reference_class_balanced_point_macro',
                     evaluated_points=sum(map(len,known_sides)),scored_attempt_points=sum(map(len,sides)),
                     pending_attempts=s['unknown'],
                     unjudgeable_attempts_preserved_unknown=sum(a['unknown_reason']=='human_unmeasurable_or_uncertain' for a in own),
                     accuracy=balanced(sides) if s['unknown']==0 else None,
                     known_only_accuracy=balanced(known_sides),
                     class_points={'negative':sum(points[k]['reference_label']==0 for k in by_point),
                                   'positive':sum(points[k]['reference_label']==1 for k in by_point)},
                     unknown_reasons=dict(Counter(a['unknown_reason'] for a in own if a['correct'] is None)))
            if s['unknown']==0 and not all(sides): s['status']='missing_reference_class'
            metrics[bucket].append(s)
        by_point=defaultdict(list)
        for a in cohort: by_point[a['key']].append(a)
        for key, aa in sorted(by_point.items()):
            p=points[key]; n=len(aa); counts=Counter(a['human_label'] if a['human_label'] in (0,1) else 'unknown' for a in aa)
            theta=p.get('theta'); interval=p.get('interval'); span=p.get('normalization_span')
            curves.append(dict(model=model,prompt_type=prompt,reference_key=list(key),
                clip_id=key[0],event=key[1],window_sha256=key[4],bin=p['bin'],
                context_sha256=p.get('context_sha256'),axis=p.get('axis'),theta=theta,
                normalization_span=span,reference_boundary_interval=interval,
                signed_distance_interval=[(theta-interval[1])/span,(theta-interval[0])/span] if interval and span else None,
                signed_distance_raw_interval=[theta-interval[1],theta-interval[0]] if interval else None,
                signed_distance_interval_unit='normalized_by_frozen_span',
                absolute_distance_interval=p.get('normalized_distance'),
                reference_label=p['reference_label'],K=n,p1=counts[1]/n,p0=counts[0]/n,
                p_unknown=counts['unknown']/n,**summary([a['correct'] for a in aa])))

    lookup={(a['model'],a['prompt_type'],a['seed'],a['key']):a for a in attempts}
    pair_rows=[]
    for pair_id,(left,right) in sorted(pairs.items()):
        for model,prompt,seed in groups:
            a=lookup.get((model,prompt,seed,left)); b=lookup.get((model,prompt,seed,right))
            if a is None and b is None: continue
            av=a['correct'] if a else None; bv=b['correct'] if b else None
            # Pair success is A AND B. A certified incorrect endpoint proves
            # pair failure without asserting anything about its unknown mate.
            pair_correct=0 if av==0 or bv==0 else 1 if av==1 and bv==1 else None
            pair_rows.append(dict(pair_id=pair_id,model=model,prompt_type=prompt,seed=seed,
                correct=pair_correct,endpoint_correctness=[av,bv],
                decision='known_endpoint_mismatch' if av==0 or bv==0 else 'both_endpoints_correct' if pair_correct==1 else 'incomplete_endpoint_evidence',
                endpoint_unknown_reasons=[x['unknown_reason'] if x else 'endpoint_not_bound' for x in (a,b)]))
    pair_overall=summary([r['correct'] for r in pair_rows])
    pair_overall.update(task_events=pair_overall['expected'],binary=pair_overall['final'])
    pair_groups=[]
    for model,prompt in sorted({(r['model'],r['prompt_type']) for r in pair_rows}):
        rr=[r for r in pair_rows if (r['model'],r['prompt_type'])==(model,prompt)]
        pair_groups.append(dict(model=model,prompt_type=prompt,**summary([r['correct'] for r in rr])))
    pair_by_model = [
        dict(model=model, **summary([r['correct'] for r in pair_rows if r['model'] == model]))
        for model in sorted({r['model'] for r in pair_rows})
    ]
    pair_result=dict(metric_name='CoupledPairAcc',mode='coupled_seed',overall=pair_overall,
                     registered_pair_windows=len(pairs),
                     independent_case_pairs=len({tuple(sorted((a[0],b[0]))) for a,b in pairs.values()}),
                     by_model=pair_by_model,by_model_prompt=pair_groups,rows=pair_rows)
    strata=[]
    for bucket in metrics:
        own=[a for a in attempts if a['bin']==bucket]
        strata.append(dict(bin=bucket,registered_event_windows=sum(p['bin']==bucket for p in points.values()),
                           aggregation='event_attempt_micro_not_balanced_NearAcc',
                           unique_cases=len({a['clip_id'] for a in own}),
                           **summary([a['correct'] for a in own])))
    effects=[]
    for model,prompt in sorted({(g[0],g[1]) for g in groups}):
        find=lambda b:next((r for r in metrics[b] if (r['model'],r['prompt_type'])==(model,prompt)),{})
        near,far=find('near'),find('far')
        valid=near.get('accuracy') is not None and far.get('accuracy') is not None
        effects.append(dict(model=model,prompt_type=prompt,
                            near_minus_far=near['accuracy']-far['accuracy'] if valid else None,
                            status='descriptive_unmatched_context' if valid else 'incomplete_or_missing_stratum'))
    extras=dict(distance_strata=strata,response_curve=curves,distance_effect=effects,
                case_distance_ledger=[dict(case_id=c,
                    admitted_windows=[dict(reference_key=list(k),bin=p['bin'],
                        normalized_distance=p.get('normalized_distance')) for k,p in sorted(points.items()) if k[0]==c],
                    status='admitted_windows_only' if any(k[0]==c for k in points) else 'boundary_evidence_not_admitted',
                    distance=None if not any(k[0]==c for k in points) else 'see_window_intervals')
                    for c in sorted({s[0] for s in roster})],
                all_case_event_accuracy=dict(overall=summary(event_values),
                    aggregation='registered_event_attempt_micro_including_missing_generations',
                    by_model_prompt=[dict(model=m,prompt_type=p,**summary(v)) for (m,p),v in sorted(event_groups.items())]),
                generation_slot_ledger=slot_ledger,
                distance_effect_claim='Descriptive only: contexts are not matched; no causal difficulty claim.',
                boundary_attempts=attempts,
                denominator_ledger=dict(expected_generation_slots=len(roster),actual_generation_slots=len(slots),
                    missing_generation_slots=len(missing_slots),missing_slots=[dict(zip(('clip_id','model','prompt_type','seed'),s)) for s in sorted(missing_slots)],
                    unique_cases=len({s[0] for s in roster}),slot_statuses=dict(slot_statuses),
                    all_registered_event_attempts=summary(event_values),
                    all_event_unknown_reasons=dict(event_reasons),
                    quality_failure=None,generation_failure=None,
                    failure_status='No independent failure adjudication supplied; missing media and unjudgeable remain unknown.',
                    admitted_event_windows=len(points),unique_boundary_cases=len({k[0] for k in points}),
                    boundary_attempts=summary([a['correct'] for a in attempts]),
                    unknown_reasons=dict(Counter(a['unknown_reason'] for a in attempts if a['correct'] is None)),
                    missing_slots_without_bound_window=sum(not any(a['clip_id']==s[0] and a['model']==s[1] and a['prompt_type']==s[2] and a['seed']==s[3] for a in attempts) for s in missing_slots)),
                independent_pair_accuracy=dict(accuracy=None,status='no_native_independent_pairs_registered'),
                same_side_accuracy=dict(accuracy=None,status='no_native_same_side_pairs_registered'),
                event_exact=dict(accuracy=None,status='native_full_chain_observations_not_registered'),
                chain_pair_exact=dict(accuracy=None,status='native_full_chain_pairs_not_registered'),
                boundary_error=response_grid_diagnostics(curves, event_specs))
    return metrics,pair_result,extras
