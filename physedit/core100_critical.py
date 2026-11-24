"""Versioned critical evaluation for all 100 revised references.

Source numerical intervals and RGB scoring admission are separate. Legacy
inputs, annotations and live registries are never edited by this module.
"""
import argparse
import collections
import copy
import hashlib
import json
import math
from pathlib import Path

from . import critical_metrics

ROOT=Path(__file__).resolve().parents[1]
POINTER=ROOT/'catalog/active_core100_critical_protocol.json'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def mean(values):return sum(values)/len(values)if values else None

def distance_strata(points):
    """Point-balanced distance strata; NearAcc never requires a pair."""
    from .boundary import distance_summary
    rows=[]
    for point in points:
        row=dict(point)
        row['scene_id']=point.get('scene_id') or point.get('clip_id','').split('_')[0]
        row['source_group_id']=point.get('source_group_id') or row['scene_id']
        rows.append(row)
    return distance_summary(rows, admitted_key='critical_rgb_distance_admitted')

def scene_scores(cases, points, pairs):
    points_by_clip = {point['clip_id']: point for point in points}
    scenes = collections.defaultdict(list)
    for case in cases:
        scenes[case['scene_id']].append(case)

    results = []
    for scene, scene_cases in sorted(scenes.items()):
        scene_points = [points_by_clip[case['clip_id']] for case in scene_cases]
        scene_clips = {point['clip_id'] for point in scene_points}
        near_families = collections.defaultdict(list)
        for case, point in zip(scene_cases, scene_points):
            if point['bin'] != 'near' or not point['critical_rgb_distance_admitted']:
                continue
            unit = primary(case)
            context = case.get('fixed_context_sha256') or case['clip_id']
            event_rule = case.get('judge_protocol', {}).get('events', {}).get(unit['event'])
            profile = case.get('numerical_reference_profile') or {}
            family = (context, unit['axis'], unit['event'], case['horizon_s'], profile.get('id'),
                      json.dumps(event_rule, sort_keys=True),
                      unit['normalization_span'], tuple(map(tuple, unit['intervals'])))
            near_families[family].append(point)
        paired_near = [family for family in near_families.values()
                       if {point['label'] for point in family} == {0, 1}]
        near_family_scores = [next(bucket['balanced_accuracy'] for bucket in distance_strata(family)
                                    if bucket['bin'] == 'near') for family in paired_near]
        near_stats = next(bucket for bucket in distance_strata(scene_points) if bucket['bin'] == 'near')
        pair_metrics = {}
        for mode in ('independent', 'coupled_seed'):
            scores = [pair['score'] for pair in pairs if pair['pairing_mode'] == mode
                      and pair['reference_admitted'] and scene_clips.issuperset(pair['clips'])]
            pair_metrics[mode] = dict(eligible_pairs=len(scores),
                                      finalized_pairs=sum(value is not None for value in scores),
                                      accuracy=mean(scores) if None not in scores else None)
        results.append(dict(
            scene_id=scene, registered_points=len(scene_points),
            distance_admitted_points=sum(p['critical_rgb_distance_admitted'] for p in scene_points),
            finalized_points=sum(p['accuracy'] is not None for p in scene_points),
            distance_strata=distance_strata(scene_points),
            near_registered_points=sum(map(len, near_families.values())),
            near_two_sided_families=len(paired_near),
            near_accuracy=near_stats['balanced_accuracy'],
            nearacc_definition='point_balanced_unpaired_v2',
            near_accuracy_family_macro=mean(near_family_scores) if near_family_scores and None not in near_family_scores else None,
            near_accuracy_lower_bound=near_stats['balanced_accuracy_lower_bound'],
            near_accuracy_sides=near_stats['side_accuracy'],
            near_reference_points=near_stats['reference_points'],
            near_reference_counts=near_stats['reference_counts'],
            near_finalized_points=near_stats['finalized_points'],
            near_scene_coverage=len(near_stats['scenes']),
            pair_metrics=pair_metrics))
    return results

def load(registry_path=None,expected_sha256=None):
    pointer=read(POINTER)
    path=Path(registry_path).resolve()if registry_path else(POINTER.parent/pointer['registry']).resolve()
    expected=expected_sha256 if registry_path else pointer['registry_sha256']
    if not expected or not path.is_relative_to(ROOT)or sha(path)!=expected:
        raise ValueError('critical_registry_changed')
    data=read(path)
    from .core100_rebalanced import load as load_roster
    roster_pointer=data.get('roster_pointer')
    if roster_pointer:
        rp=(ROOT/roster_pointer['path']).resolve()
        if not rp.is_relative_to(ROOT)or sha(rp)!=roster_pointer['sha256']:
            raise ValueError('critical_roster_pointer_changed')
        _,m,tasks,_=load_roster(rp)
    else:_,m,tasks,_=load_roster()
    conditions={r['clip_id']:r for r in tasks}
    if data['roster_release']!=m['version']:raise ValueError('critical_roster_version_changed')
    rows=data['cases'];wanted={f'T{i:02}_C{j:02}'for i in range(1,21)for j in range(1,6)}
    if len(rows)!=100 or {r['clip_id']for r in rows}!=wanted:raise ValueError('critical_roster_changed')
    from .core100_bindings import interval_distance
    for row in rows:
        if row['reference_video_sha256']!=conditions[row['clip_id']]['reference']['sha256']:
            raise ValueError('critical_reference_changed')
        if 'judge_protocol'in row and set(row['judge_protocol']['events'])!=set(row['reference_events']):
            raise ValueError('critical_judge_event_definitions_changed')
        for unit in row['critical_units']:
            if unit['label']!=row['reference_events'][unit['event']]:raise ValueError('critical_reference_label_changed')
            if not unit['intervals']:
                if unit.get('numerical_evidence_status')!='matching_reference_version_boundary_missing' or unit['normalized_absolute_interval']is not None or unit['bin']is not None or unit['rgb_distance_admitted']:
                    raise ValueError('critical_unresolved_boundary_contradiction')
                continue
            ds=[interval_distance(unit['theta'],b,unit['normalization_span'])for b in unit['intervals']]
            expected=[min(d['normalized_absolute_interval'][i]for d in ds)for i in [0,1]]
            if unit['normalized_absolute_interval']!=expected:raise ValueError('critical_distance_changed')
            if unit['rgb_distance_admitted']and unit['rgb_blockers']:raise ValueError('critical_admission_contradiction')
    for name,digest in data['source_hashes'].items():
        source=(ROOT/name).resolve()
        if not source.is_relative_to(ROOT)or sha(source)!=digest:raise ValueError('critical_source_snapshot_changed')
    for c in data['input_bindings']:
        if sha(ROOT/c['condition_image'])!=c['condition_sha256']:raise ValueError('critical_condition_changed')
        if c['prompt_variants_en']!=conditions[c['clip_id']]['prompt_variants_en']:
            raise ValueError('critical_frozen_prompt_changed')
    return data,path

def primary(row):
    candidates=[u for u in row['critical_units']if not u['supplemental_axis']]
    if not candidates:raise ValueError('original_critical_family_missing:'+row['clip_id'])
    # Preserve existing admitted event selection when it belongs to the
    # original axis. Otherwise select its original first registered boundary,
    # never the event/axis with the smallest distance or best model score.
    return next((u for u in candidates if u['event']in row['admitted_rgb_events']),candidates[0])

def replay_disclosure(row):
    closure=row.get('original_GPU_terminal_event_closure')or row.get('original_event_stage_closure',{})
    if 'source_comparison_scope'not in closure:return None
    extra=closure.get('additional_repeat_attempts')
    result=dict(scope=closure['source_comparison_scope'],
        normalization_domain=row['normalization_domain'],
        observed_interval_extends_published_domain=any(b.get('outside_original_parameter_domain',False)for b in row['boundary_records']),
        source_full_window_position_error_m=closure['source_full_window_position_error_m'],
        exact_full_trajectory_equivalence_claimed=False,
        additional_repeats_successful=extra.get('additional_repeats_successful')if extra else None,
        additional_repeat_limitation=extra.get('additional_repeat_limitation')if extra else None,
        applicability=closure.get('applicability','Original terminal binary event and observed local interval only; not full-chain or trajectory/inversion validation.'))
    if 'onset_offset_frames'in closure:result['onset_offset_frames']=closure['onset_offset_frames']
    return result


def bind_answer(answer,*,reference_video_sha256,horizon_s,axis=None,registry=None):
    data,path=(registry,None)if registry is not None else load()
    cid=answer.get('clip_id',answer.get('point_id'));row=next(r for r in data['cases']if r['clip_id']==cid)
    if reference_video_sha256!=row['reference_video_sha256']:raise ValueError('critical_reference_identity_mismatch')
    if abs(horizon_s-row['horizon_s'])>1e-8:raise ValueError('critical_observation_window_mismatch')
    candidates=[u for u in row['critical_units']if u['event']==answer['target_event']and(axis is None or u['axis']==axis)]
    if len(candidates)!=1:raise ValueError('critical_event_axis_selection_required')
    unit=candidates[0]
    if answer['events'].get(unit['event'])!=unit['label']:raise ValueError('critical_event_truth_conflict')
    if 'theta'in answer and answer['theta']!=unit['theta']:raise ValueError('critical_parameter_conflict')
    result=copy.deepcopy(answer)
    result['critical_reference_id']=unit['reference_id']
    result['critical_rgb_distance_admitted']=unit['rgb_distance_admitted']
    result['numerical_distance_reference']=copy.deepcopy(unit)
    if row.get('numerical_reference_profile'):
        result['numerical_reference_profile']=copy.deepcopy(row['numerical_reference_profile'])
    if replay_disclosure(row):result['reference_replay_disclosure']=replay_disclosure(row)
    source=str(path or POINTER)
    if not unit['rgb_distance_admitted']:
        reason=('Numerical interval retained; RGB event/axis/window evidence pending.'if unit['intervals']else
            'Matching reference-version interval not recovered; another production version is not transferred.')
        result['boundary']=dict(status='unavailable',reason=reason,source=source)
    elif len(unit['intervals'])==1:
        result['boundary']=dict(status='verified',bracket=unit['intervals'][0],normalization_span=unit['normalization_span'],
            source=source,verification_scope='registered local simulator interval; no exact or unique real-world threshold claim')
        if row.get('numerical_reference_profile'):
            result['boundary']['verification_scope']='author-approved CPU64 refined-profile observed interval; original-to-refined boundary shift unquantified; not an original rendering search value'
    else:
        result['boundary']=dict(status='observed',intervals=[dict(bracket=b)for b in unit['intervals']],
            normalization_span=unit['normalization_span'],source=source)
    return result

def inversion_registration(path,data):
    """Bind an external validated inversion protocol; never infer eligibility."""
    if path is None:return {},None
    path=Path(path);registration=read(path);rows={r['clip_id']:r for r in data['cases']}
    if registration.get('roster_release')!=data['roster_release']:
        raise ValueError('critical_inversion_roster_changed')
    result={}
    for r in registration['references']:
        cid=r['clip_id'];ref=rows.get(cid)
        if cid in result or ref is None:raise ValueError('critical_inversion_reference_ids_changed')
        if r.get('reference_video_sha256')!=ref['reference_video_sha256']or r.get('horizon_s')!=ref['horizon_s']:
            raise ValueError('critical_inversion_reference_or_window_changed')
        if r.get('status')not in ['eligible','not_applicable','pending_reference_validation']or not r.get('reason'):
            raise ValueError('critical_inversion_status_unregistered')
        if r['status']=='eligible':
            if not r.get('protocol_version')or not r.get('validation_evidence'):
                raise ValueError('critical_inversion_validation_missing')
            window=r.get('measurement_window_s')
            if not isinstance(window,list)or len(window)!=2 or not all(type(v)in(int,float)and math.isfinite(v)for v in window)or not 0<=window[0]<window[1]<=ref['horizon_s']:
                raise ValueError('critical_inversion_measurement_window_changed')
        result[cid]=dict(r,reference_registration_sha256=sha(path))
    return result,sha(path)


def prepare(out,model,*,fps,variant=1,seeds=(0,),inversion_registry=None,
            registry_path=None,expected_sha256=None):
    data,path=load(registry_path,expected_sha256)
    if variant not in [1,2,3]or not math.isfinite(fps)or fps<=0:raise ValueError('invalid_critical_generation_config')
    if not seeds or len(set(seeds))!=len(seeds)or any(type(s)is not int or not 0<=s<100000 for s in seeds):
        raise ValueError('invalid_critical_seed_registration')
    inversions,inversion_hash=inversion_registration(inversion_registry,data)
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
    by_id={r['clip_id']:r for r in data['input_bindings']};inputs=[];keys=[]
    for index,row in enumerate(data['cases']):
        cid=row['clip_id'];c=by_id[cid];unit=primary(row)
        pairs=[p for p in data['pairs']if cid in p['clips']]
        if len(pairs)>1:raise ValueError('overlapping_critical_pair_registration')
        pair=pairs[0]if pairs else None
        context=('pair:'+pair['pair_id']if pair else row.get('fixed_context_sha256')or 'point:'+cid)
        for seed0 in seeds:
            seed=seed0+(100000*index if pair and pair['pairing_mode']=='independent'else 0)
            sid=cid+'_p'+str(variant)+'_s'+str(seed)
            inputs.append(dict(sample_id=sid,clip_id=cid,point_id=cid,seed=seed,
                fps=fps,num_frames=4*math.ceil(row['horizon_s']*fps/4)+1,
                evaluation_duration_s=row['horizon_s'],condition_frame_index=c['condition_frame_index'],
                condition_image=str(ROOT/c['condition_image']),condition_sha256=c['condition_sha256'],
                source_width=c['source_width'],source_height=c['source_height'],
                prompt=c['prompt_variants_en'][variant-1],prompt_sha256=c['prompt_variant_sha256'][variant-1],prompt_variant=variant,
                reference_video_sha256=row['reference_video_sha256'],reference_release_version=data['roster_release']))
            key=dict(sample_id=sid,clip_id=cid,point_id=cid,scene_id=cid[:3],source_group_id=context,
                family_id=unit['axis'],theta=unit['theta'],target_event=unit['event'],events=copy.deepcopy(row['reference_events']),
                reference_video_sha256=row['reference_video_sha256'],reference_sample_id=row['sample_id'],
                reference_state_sha256=row.get('source_state_sha256'),required_order=[],
                pair_ids=[pair['pair_id']]if pair else [],pairing_mode=pair['pairing_mode']if pair else'independent',
                target_reference_admitted=unit['event']in row['admitted_rgb_events'],
                inversion=dict(status='pending_reference_validation',reason='External inversion registration is retained separately; this critical run does not replace its applicability decisions.'),
                full_EventExact_applicable=False)
            if cid in inversions:key['inversion']=copy.deepcopy(inversions[cid])
            key['gravity']=row['declared_gravity_mps2']
            keys.append(bind_answer(key,reference_video_sha256=row['reference_video_sha256'],horizon_s=row['horizon_s'],axis=unit['axis'],registry=data))
    from .boundary import write_rows,plan
    write_rows(out/'inputs.jsonl',inputs);write_rows(out/'answers.jsonl',keys)
    cases={r['clip_id']:r for r in data['cases']}
    from .judge_contract import VERSION as JOB_SCHEMA
    protocol=dict(schema='core100-critical-judge-protocol-v3',event_job_schema=JOB_SCHEMA,roster_release=data['roster_release'],
        critical_registry_sha256=sha(path),confidence_threshold=.9,
        confidence_policy='Uncalibrated judge confidence routes prelabels only; no automatic final labels.',
        samples={k['sample_id']:copy.deepcopy(cases[k['clip_id']]['judge_protocol'])for k in keys})
    (out/'judge_protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf8')
    spec=dict(model=model,inputs=str(out/'inputs.jsonl'),answers=str(out/'answers.jsonl'),
        reference_release_version=data['roster_release'],critical_registry=str(path),critical_registry_sha256=sha(path),
        full_EventExact_applicable=False,event_job_schema=JOB_SCHEMA)
    if inversion_registry:
        destination=out/'inversion_registration.json';destination.write_bytes(Path(inversion_registry).read_bytes())
        spec.update(inversion_registration=str(destination),inversion_registration_sha256=inversion_hash)
    (out/'spec.json').write_text(json.dumps(spec,indent=2),encoding='utf8')
    result=plan(spec)
    points=[]
    for case in data['cases']:
        unit=primary(case)
        points.append(dict(clip_id=case['clip_id'],label=unit['label'],bin=unit['bin'],
            critical_rgb_distance_admitted=unit['rgb_distance_admitted'],accuracy=None))
    admitted={point['clip_id'] for point in points if point['critical_rgb_distance_admitted']}
    pairs=[dict(pair,score=None,reference_admitted=admitted.issuperset(pair['clips'])) for pair in data['pairs']]
    result['scene_reference_coverage']=scene_scores(data['cases'],points,pairs)
    (out/'PRECHECK.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    return result

def score(run_dir,**kwargs):
    from . import boundary
    run_dir=Path(run_dir);registration=read(run_dir/'registered.json')['spec']
    data,path=load(registration['critical_registry'],registration['critical_registry_sha256'])
    if registration.get('inversion_registration')and sha(registration['inversion_registration'])!=registration.get('inversion_registration_sha256'):
        raise ValueError('critical_inversion_registration_changed')
    keys=boundary.read_rows(run_dir/'answers.jsonl');inputs=boundary.read_rows(run_dir/'inputs.jsonl')
    if {r['clip_id']for r in inputs}!={r['clip_id']for r in data['cases']}:raise ValueError('critical_run_requires_complete100_roster')
    conditions={r['clip_id']:r for r in data['input_bindings']};by_case={r['clip_id']:r for r in data['cases']}
    units={u['reference_id']:u for r in data['cases']for u in r['critical_units']}
    key_by_id={k['sample_id']:k for k in keys}
    for row in inputs:
        c=conditions[row['clip_id']];k=key_by_id[row['sample_id']];ref=by_case[row['clip_id']]
        if row['reference_release_version']!=data['roster_release']or row['reference_video_sha256']!=ref['reference_video_sha256']:
            raise ValueError('critical_run_reference_changed')
        if row['prompt']!=c['prompt_variants_en'][row['prompt_variant']-1]or row['condition_sha256']!=c['condition_sha256']:
            raise ValueError('critical_run_frozen_input_changed')
        if row['evaluation_duration_s']!=ref['horizon_s']:raise ValueError('critical_run_window_changed')
        unit=units.get(k.get('critical_reference_id'))
        if unit!=k.get('numerical_distance_reference')or unit!=primary(ref):raise ValueError('critical_run_target_registration_changed')
        if k['target_event']!=unit['event']or k['theta']!=unit['theta']or k['critical_rgb_distance_admitted']!=unit['rgb_distance_admitted']:
            raise ValueError('critical_run_target_registration_changed')
        rebound=bind_answer(k,reference_video_sha256=ref['reference_video_sha256'],horizon_s=ref['horizon_s'],axis=unit['axis'],registry=data)
        if k['boundary']!=rebound['boundary']:raise ValueError('critical_run_boundary_changed')
        if k['events']!=ref['reference_events']:raise ValueError('critical_run_reference_truth_changed')
        if k.get('numerical_reference_profile')!=ref.get('numerical_reference_profile'):
            raise ValueError('critical_run_numerical_reference_profile_changed')
        if k.get('reference_replay_disclosure')!=replay_disclosure(ref):
            raise ValueError('critical_run_reference_replay_disclosure_changed')
    result=boundary.score(run_dir,**kwargs)
    annotations=boundary.read_rows(kwargs['annotations'])if kwargs.get('annotations')else[]
    annotation_by_id={r['sample_id']:r for r in annotations}
    truth_path=ROOT/'benchmarks/core100_original_reference_metrics_20261110/EVENTEXACT_REFERENCE_TRUTH.json'
    eventexact_truth=critical_metrics.load_eventexact_truth(truth_path)if truth_path.exists()else{}
    for row in result['rows']:
        k=key_by_id[row['sample_id']];ann=annotation_by_id.get(row['sample_id'],{})
        row['clip_id']=k['clip_id'];row['critical_reference_id']=k['critical_reference_id']
        row['critical_rgb_distance_admitted']=k['critical_rgb_distance_admitted']
        row['numerical_reference_profile_id']=k.get('numerical_reference_profile',{}).get('id','case_registered_reference')
        row['decision_final']=(row['generation_status']=='generation_failure'or row['screening_status']=='confirmed_failure'or
            (row['screening_status']=='pass'and row['annotation_status']=='bound'and ann.get('review_status')=='final'))
        row['prelabel_target_correct_diagnostic']=row['target_correct']
        if not row['decision_final']:row['target_correct']=0
        truth=eventexact_truth.get(k['clip_id'])
        row['generated_events']=dict(ann.get('events',{}))if row['annotation_status']=='bound'else{}
        if truth:
            reference_events=truth.get('event_labels')or{}
            generated=row['generated_events']
            row['full_event_chain_registered']=True
            row['eventexact_truth_authority']=truth.get('authority')
            row['reference_events']=dict(reference_events)
            row['event_exact_full']=critical_metrics.event_exact_output(
                reference_events,generated,truth.get('required_order'),ann.get('event_times_s'))
            row['event_bits_exact_full']=(int(all(generated.get(event)==value for event,value in reference_events.items()))
                if reference_events and all(generated.get(event)in(0,1)for event in reference_events)else None)
        else:
            row['full_event_chain_registered']=False
            row['event_exact_full']=None
            row['event_bits_exact_full']=None
            row['reference_events']=dict(k['events'])
        row['target_vector_match_diagnostic']=row.pop('event_exact');row['event_exact']=None
    points=[]
    for cid in conditions:
        own=[r for r in result['rows']if r['clip_id']==cid];unit=primary(by_case[cid])
        theta=unit.get('theta');span=unit.get('normalization_span');intervals=unit.get('intervals')or[]
        signed=[theta-intervals[0][1],theta-intervals[0][0]]if theta is not None and intervals else None
        points.append(dict(clip_id=cid,label=unit['label'],accuracy_lower_bound=mean([r['target_correct']for r in own]),
            accuracy=mean([r['target_correct']for r in own])if all(r['decision_final']for r in own)else None,
            bin=unit['bin'],theta=theta,
            normalized_absolute_interval=unit.get('normalized_absolute_interval'),
            signed_interval=signed,
            normalized_signed_interval=[signed[0]/span,signed[1]/span]if signed and span else None,
            critical_rgb_distance_admitted=unit['rgb_distance_admitted'],
            numerical_reference_profile_id=by_case[cid].get('numerical_reference_profile',{}).get('id','case_registered_reference'),
            target_reference_admitted=unit['event']in by_case[cid]['admitted_rgb_events'],registered=len(own)))
    near_stats=next(bucket for bucket in distance_strata(points) if bucket['bin']=='near')
    result['legacy_aggregate_diagnostics']={name:result.get(name)for name in ['balanced_accuracy','near_accuracy','event_exact','independent_pair_accuracy','coupled_pair_accuracy','task_success_rate','event_measurement']}
    result['near_accuracy']=near_stats['balanced_accuracy']
    result['nearacc_definition']='point_balanced_unpaired_v2'
    result['near_accuracy_lower_bound']=near_stats['balanced_accuracy_lower_bound']
    result['near_accuracy_sides']=near_stats['side_accuracy']
    result['near_reference_points']=near_stats['reference_points']
    result['near_reference_counts']=near_stats['reference_counts']
    result['near_finalized_points']=near_stats['finalized_points']
    result['near_scene_coverage']=len(near_stats['scenes'])
    result['near_registered_reference_points']=near_stats['reference_points']
    result['response_curve']=critical_metrics.response_curve(result['rows'],points)
    result['EventExact_reference_registered']=sum(case['clip_id'] in eventexact_truth for case in data['cases'])
    result['EventExact_reference_registered_ids']=sorted(case['clip_id'] for case in data['cases'] if case['clip_id'] in eventexact_truth)
    result['EventExact_reference_coverage']=result['EventExact_reference_registered']/100
    event_exact=critical_metrics.event_exact_summary(result['rows'])
    result['event_exact']=event_exact['value']
    result['EventExact_status']=event_exact['status']
    result['EventExact_known_only']=event_exact['known_only']
    result['EventExact_coverage']=event_exact['coverage']
    result['full_EventExact_applicable']=bool(event_exact['eligible_points'])
    result['boundary_error']=None
    result['boundary_error_status']='No model response-grid threshold estimation registered in this run; pairs alone are not a point estimate.'
    result['target_reference_coverage']=sum(primary(r)['event']in r['admitted_rgb_events']for r in data['cases'])/100
    result['critical_reference_coverage']=sum(p['critical_rgb_distance_admitted']for p in points)/100
    result['reference_registration_summary']=data['summary'];result['points']=points
    result['distance_curve']=points
    result['distance_strata']=distance_strata(points)
    from .boundary import distance_effect
    result['distance_effect']=distance_effect(result['distance_strata'])
    profiles=sorted({p['numerical_reference_profile_id']for p in points})
    result['distance_strata_by_numerical_profile']=[dict(profile_id=profile,
        strata=distance_strata([p for p in points if p['numerical_reference_profile_id']==profile]))for profile in profiles]
    result['numerical_reference_profile_coverage']={profile:sum(p['critical_rgb_distance_admitted']and p['numerical_reference_profile_id']==profile for p in points)for profile in profiles}
    result['refined_reference_configurations_and_uncertainty']={r['clip_id']:r['numerical_reference_profile']for r in data['cases']if r.get('numerical_reference_profile')}
    result['reference_replay_disclosures']={r['clip_id']:replay_disclosure(r)for r in data['cases']if replay_disclosure(r)}
    result['reference_profile_scope']='case_registered_reference retains each case own provenance and does not assert a homogeneous engine/profile. Approved refined cases are separately tagged, stratified and disclosed.'
    result['distance_effect_claim']='Descriptive strata only. This roster does not by itself register a matched-context near/mid/far response experiment.'
    # Pending human decisions are not final model errors. Keep a transparent
    # lower bound, while final paired scores require finalized endpoints.
    final_pairs=[]
    for pair in data['pairs']:
        ps=[next(p for p in points if p['clip_id']==cid)for cid in pair['clips']]
        own=[r for r in result['rows']if r['clip_id']in pair['clips']]
        value=None
        if all(p['critical_rgb_distance_admitted']and p['accuracy']is not None for p in ps):
            if pair['pairing_mode']=='independent':value=ps[0]['accuracy']*ps[1]['accuracy']
            else:
                seeds={r['seed']for r in own};value=mean([math.prod(r['target_correct']for r in own if r['seed']==seed)for seed in seeds])
        final_pairs.append(dict(pair,score=value,reference_admitted=all(p['critical_rgb_distance_admitted']for p in ps)))
    result['pairs']=final_pairs
    same_side=critical_metrics.same_side_accuracy(data.get('same_side_pairs', []), points, result['rows'])
    result['same_side_accuracy']=same_side
    chain=critical_metrics.chain_pair_exact_summary(result['pairs'], result['rows'])
    result['chain_pair_exact']=chain
    reference_intervals={}
    for pair in result['pairs']:
        unit=primary(by_case[pair['clips'][0]])
        intervals=unit.get('intervals') or []
        if intervals:
            reference_intervals[pair['pair_id']]=intervals[0]
    result['boundary_error_report']=critical_metrics.boundary_error_summary(
        result['response_curve'], result['pairs'], reference_intervals)
    result['boundary_error_status']=result['boundary_error_report']['status']
    result['scene_scores']=scene_scores(data['cases'], points, final_pairs)
    for mode,name in [('independent','independent_pair_accuracy'),('coupled_seed','coupled_pair_accuracy')]:
        own_pairs=[p for p in final_pairs if p['pairing_mode']==mode]
        eligible=[p for p in own_pairs if p['reference_admitted']]
        values=[p['score']for p in eligible]
        known=[v for v in values if v is not None]
        result[name]=mean(values)if values and all(v is not None for v in values)else None
        result[name+'_known_only']=mean(known)if known else None
        result[name+'_known_pairs']=len(known)
        result[name+'_known_coverage']=len(known)/len(values)if values else None
        result[name+'_complete']=bool(values)and len(known)==len(values)
        result[name+'_reference_coverage']=len(eligible)/len(own_pairs)if own_pairs else None
        result[name+'_eligible_pairs']=len(eligible)
    result['independent_pair_success']=result['independent_pair_accuracy']
    admitted=[p for p in points if p['target_reference_admitted']]
    sides=[[p for p in admitted if p['label']==y]for y in [0,1]]
    result['balanced_accuracy']=mean([mean([p['accuracy']for p in side])for side in sides])if all(sides)and all(p['accuracy']is not None for p in admitted)else None
    result['ordinary_event_reference_points']=len(admitted)
    result['task_success_rate']=None
    result['evaluation_scope']='versioned Core100 critical protocol; original task/vector diagnostics retained separately'
    measured=[r for r in result['rows']if r['decision_final']and r['target_assessable']]
    result['event_measurement']=dict(registered=len(keys),final_decision_count=sum(r['decision_final']for r in result['rows']),
        target_assessable=len(measured),target_coverage=len(measured)/len(keys),
        reference_coverage=result['target_reference_coverage'],
        full_chain_coverage=None,full_EventExact_applicable=False,
        pending_decisions=sum(not r['decision_final']for r in result['rows']),
        aggregation='point-macro label-balanced accuracy; per-output measurement coverage; failures retained')
    for family in result['families']:
        family['target_vector_match_diagnostic']=family.pop('event_exact');family['event_exact']=None
    return result

def judge(run_dir,protocol_path,out_dir,*,quality_file=None,event_file=None,reviews_file=None,execute=False):
    """Run the two-stage router and score human-confirmed target labels.

    The generic router predates review_status. Convert only explicitly
    human-confirmed target labels to final and retain per-event review states.
    """
    from . import boundary,boundary_pipeline
    run_dir=Path(run_dir);out_dir=Path(out_dir)
    registration=read(run_dir/'registered.json')['spec']
    data,_=load(registration['critical_registry'],registration['critical_registry_sha256'])
    protocol=read(protocol_path);cases={r['clip_id']:r for r in data['cases']}
    keys=boundary.read_rows(run_dir/'answers.jsonl')
    if protocol.get('critical_registry_sha256')!=registration['critical_registry_sha256']:
        raise ValueError('critical_judge_registry_changed')
    if registration.get('event_job_schema')and protocol.get('event_job_schema')!=registration['event_job_schema']:
        raise ValueError('critical_judge_observation_contract_changed')
    for key in keys:
        expected=cases[key['clip_id']]['judge_protocol']
        actual=protocol['samples'][key['sample_id']]
        if any(actual.get(k)!=expected[k]for k in ['target','events','uncertainty_policy']):
            raise ValueError('critical_judge_protocol_changed')
    receipt=boundary_pipeline.run_pipeline(run_dir,protocol_path,out_dir,quality_file,event_file,reviews_file,execute)
    annotations=boundary.read_rows(out_dir/'annotations.jsonl')
    routes={r['sample_id']:r for r in boundary.read_rows(out_dir/'routing.jsonl')}
    answers={k['sample_id']:k for k in keys}
    for ann in annotations:
        route=routes[ann['sample_id']]['events']
        ann['event_review_status']={e:'final'if r=='human_confirmed'else'pending'for e,r in route.items()}
        ann['review_status']='final'if route[answers[ann['sample_id']]['target_event']]=='human_confirmed'else'pending'
    boundary.write_rows(out_dir/'critical_annotations.jsonl',annotations)
    (out_dir/'scores.json').rename(out_dir/'generic_scores_diagnostic.json')
    value=score(run_dir,annotations=out_dir/'critical_annotations.jsonl',screening=out_dir/'screening.jsonl')
    (out_dir/'scores.json').write_text(json.dumps(value,indent=2),encoding='utf8')
    receipt.update(critical_registry_sha256=registration['critical_registry_sha256'],
        score_policy='versioned critical score; human-confirmed target final, machine prelabels pending',
        final_target_labels=sum(a['review_status']=='final'for a in annotations))
    (out_dir/'critical_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf8')
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    a=sub.add_parser('status')
    a.add_argument('--registry',type=Path);a.add_argument('--expected-registry-sha256')
    a=sub.add_parser('prepare');a.add_argument('--out',type=Path,required=True);a.add_argument('--model',required=True)
    a.add_argument('--fps',type=float,required=True);a.add_argument('--variant',type=int,default=1);a.add_argument('--seeds',nargs='+',type=int,default=[0])
    a.add_argument('--inversion-registration',type=Path)
    a.add_argument('--registry',type=Path);a.add_argument('--expected-registry-sha256')
    a=sub.add_parser('score');a.add_argument('--run-dir',type=Path,required=True);a.add_argument('--annotations',type=Path)
    a.add_argument('--screening',type=Path);a.add_argument('--gravity',type=Path);a.add_argument('--out',type=Path,required=True)
    a=sub.add_parser('judge');a.add_argument('--run-dir',type=Path,required=True)
    a.add_argument('--protocol',type=Path,required=True);a.add_argument('--out-dir',type=Path,required=True)
    for name in ['quality-predictions','event-predictions','reviews']:a.add_argument('--'+name,type=Path)
    a.add_argument('--execute',action='store_true')
    args=p.parse_args()
    if args.action=='status':value=load(args.registry,args.expected_registry_sha256)[0]['summary']
    elif args.action=='prepare':value=prepare(args.out,args.model,fps=args.fps,variant=args.variant,seeds=args.seeds,inversion_registry=args.inversion_registration,
        registry_path=args.registry,expected_sha256=args.expected_registry_sha256)
    elif args.action=='judge':value=judge(args.run_dir,args.protocol,args.out_dir,quality_file=args.quality_predictions,
        event_file=args.event_predictions,reviews_file=args.reviews,execute=args.execute)
    else:
        value=score(args.run_dir,annotations=args.annotations,screening=args.screening,gravity=args.gravity)
        with args.out.open('x',encoding='utf8')as f:json.dump(value,f,indent=2)
        value={k:v for k,v in value.items()if k not in ['rows','families','points']}
    print(json.dumps(value,ensure_ascii=False))

if __name__=='__main__':main()
