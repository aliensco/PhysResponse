"""A complete reporting surface for the four admitted native critical metrics.

Reference exclusions never become generated failures. Registered attempts with
missing human observations remain in their own metric denominators.
"""
import csv
import json
from collections import Counter
from pathlib import Path


def build(report, tasks, truth, points, pairs):
    groups=sorted({(r['model'],r['prompt_type']) for r in report['generation_slot_ledger']})
    point_keys=set(points)
    pair_keys={key for endpoints in pairs.values() for key in endpoints}
    pending=[(key,row) for key,row in truth.items() if row['status']!='verified']
    if any(key in point_keys or key in pair_keys for key,_ in pending):
        raise ValueError('Unverified reference entered an admitted critical denominator')
    specs=[('NearAcc','near_accuracy','near'),('MidAcc','mid_accuracy','mid'),('FarAcc','far_accuracy','far')]
    metrics={}
    for name,field,bucket in specs:
        rows={(r['model'],r['prompt_type']):r for r in report.get(field) or []}
        metrics[name]=[]
        for model,prompt in groups:
            row=rows.get((model,prompt),{})
            metrics[name].append(dict(metric=name,model=model,prompt_type=prompt,
                accuracy=row.get('accuracy'),known_only_accuracy=row.get('known_only_accuracy'),
                status=row.get('status','no_registered_points'),expected=row.get('expected',0),
                final=row.get('final',0),unknown=row.get('unknown',0),coverage=row.get('coverage'),
                registered_points=row.get('registered_points',0),class_points=row.get('class_points',{'negative':0,'positive':0}),
                unknown_reasons=row.get('unknown_reasons',{}),aggregation='reference_class_balanced_point_macro'))
    pair_report=report.get('pair_accuracy') or {}
    rows={(r['model'],r['prompt_type']):r for r in pair_report.get('by_model_prompt',[])}
    metrics['PairAcc']=[]
    for model,prompt in groups:
        row=rows.get((model,prompt),{})
        metrics['PairAcc'].append(dict(metric='PairAcc',model=model,prompt_type=prompt,
            accuracy=row.get('accuracy'),known_only_accuracy=row.get('known_only_accuracy'),
            status=row.get('status','no_registered_pairs'),expected=row.get('expected',0),final=row.get('final',0),
            unknown=row.get('unknown',0),coverage=row.get('coverage'),
            registered_points=None,class_points=None,unknown_reasons={},aggregation='coupled_seed_registered_pair_accuracy'))
    case_scope=[]
    for item in report['case_distance_ledger']:
        case=item['case_id'];own=[(k,r) for k,r in truth.items() if k[0]==case]
        bins=Counter(p['bin'] for key,p in points.items() if key[0]==case)
        pair_windows=sum(any(key[0]==case for key in endpoints) for endpoints in pairs.values())
        case_scope.append(dict(case_id=case,retained_in_original_cohort=True,
            verified_reference_windows=sum(r['status']=='verified' for _,r in own),
            unverified_reference_windows=sum(r['status']!='verified' for _,r in own),
            near_windows=bins['near'],mid_windows=bins['mid'],far_windows=bins['far'],
            uncertain_windows=bins['uncertain'],pair_windows=pair_windows,
            critical_scope_status='admitted' if bins or pair_windows else 'no_admitted_boundary_or_pair',
            unresolved_reference_blocks_current_metrics=False))
    exclusions=[dict(case_id=key[0],event=key[1],reference_video_sha256=key[2],event_contract_sha256=key[3],
        window_sha256=key[4],window_start_s=row.get('window_start_s'),window_end_s=row.get('window_end_s'),
        reference_status=row['status'],reference_label=row.get('reference_label'),reason=row.get('reason'),
        decision='excluded_from_critical_admission_until_reference_verified',
        in_admitted_metric_denominator=False,blocks_other_admitted_points=False) for key,row in pending]
    metric_status=[]
    for name,rows in metrics.items():
        complete=[r for r in rows if r['accuracy'] is not None]
        reasons=Counter(r['status'] for r in rows if r['accuracy'] is None)
        metric_status.append(dict(metric=name,wired=True,model_prompt_groups=len(groups),complete_groups=len(complete),
            full_group_coverage=len(complete)==len(groups) and bool(groups),
            status='complete' if len(complete)==len(groups) and groups else 'partial' if complete else 'not_estimable',
            complete_group_mean=sum(r['accuracy'] for r in complete)/len(complete) if complete else None,
            missing_group_statuses=dict(reasons),expected_attempts=sum(r['expected'] for r in rows),
            final_attempts=sum(r['final'] for r in rows),pending_attempts=sum(r['unknown'] for r in rows)))
    return dict(schema='phyresponse-critical-metric-report-v1',integration_complete=True,
        current_near_scope=dict(case_ids=sorted({k[0] for k,p in points.items() if p['bin']=='near'}),
            scene_ids=sorted({k[0].split('_')[0] for k,p in points.items() if p['bin']=='near'}),
            event_windows=sum(p['bin']=='near' for p in points.values()),
            scope='admitted native-window subset only; complete coverage of the intended original Near design is not established'),
        reference_basis_counts=dict(Counter(r.get('reference_basis','native_state') for r in truth.values())),
        published_group_oracle_cases=sorted({k[0] for k,r in truth.items() if r.get('reference_basis','').startswith('published_physics_group')}),
        pipeline_status='completed_with_explicit_coverage',all_original_references_required_before_scoring=False,
        model_prompt_groups=len(groups),metrics=metrics,metric_status=metric_status,case_scope=case_scope,
        reference_exclusions=exclusions,excluded_reference_windows=len(exclusions),
        excluded_reference_cases=sorted({r['case_id'] for r in exclusions}),
        unrelated_reference_pending_blocks_current_metrics=False,
        full_four_metric_coverage=all(r['full_group_coverage'] for r in metric_status),
        policy='Use individually admitted native event/window references and registered distances/pairs. Unbound cases do not gate other points; unjudgeable generated attempts are retained unknown, never relabelled as failures.')


def export(report, directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    def write_csv(name,rows,fields=None):
        with (directory/name).open('w',encoding='utf-8-sig',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields or (list(rows[0]) if rows else ['metric','model','status']),extrasaction='ignore')
            writer.writeheader()
            for row in rows:
                writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in row.items()})
    metrics=report['formal_metric_report']
    write_csv('FOUR_METRICS.csv',[r for rows in metrics['metrics'].values() for r in rows])
    write_csv('METRIC_STATUS.csv',metrics['metric_status'])
    write_csv('CASE_METRIC_SCOPE.csv',metrics['case_scope'])
    write_csv('REFERENCE_EXCLUSIONS.csv',metrics['reference_exclusions'],
        ['case_id','event','reference_video_sha256','event_contract_sha256','window_sha256','window_start_s','window_end_s',
         'reference_status','reference_label','reason','decision','in_admitted_metric_denominator','blocks_other_admitted_points'])
    queue=[r for r in report['boundary_attempts'] if r['correct'] is None]
    write_csv('GENERATED_EVENT_REVIEW_QUEUE.csv',queue,
        ['clip_id','event','model','prompt_type','seed','bin','key','reference_label','human_label','unknown_reason'])
    write_csv('PAIR_REVIEW_QUEUE.csv',[r for r in (report.get('pair_accuracy') or {}).get('rows',[]) if r['correct'] is None],
        ['pair_id','model','prompt_type','seed','correct','endpoint_unknown_reasons'])
    write_csv('PER_TASK_EVENT_COMPARISON.csv',report.get('task_events') or [],
        ['task_id','clip_id','event','model','prompt_type','reference_label','human_label','generated_event_label_basis','correct','unknown_reason'])
    if 'event_exact' in report and 'rows' in report['event_exact']:
        write_csv('ALL100_EVENT_EXACT_TASKS.csv',report['event_exact']['rows'],
            ['task_id','case_id','model','prompt_type','seed','exact','expected_events','final_events','mismatched_events','unknown_events','decision','source_event_contract_sha256'])
        write_csv('ALL100_EVENT_EXACT_CASES.csv',report['event_exact']['by_case'])
        write_csv('ALL100_EVENT_EXACT_MODEL_PROMPT.csv',report['event_exact']['by_model_prompt'])
    if 'prompt_effects' in report:
        for field,name in [('per_model','M_D_PER_MODEL.csv'),('matched_attempts','M_D_MATCHED_ATTEMPTS.csv'),('matched_summary','M_D_MATCHED_SUMMARY.csv'),('complete_group_summary','M_D_COMPLETE_GROUP_SUMMARY.csv'),('case_event_comparison','M_D_CASE_EVENTS.csv')]:
            write_csv(name,report['prompt_effects'][field])
