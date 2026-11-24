"""M versus D comparisons on matched, already adjudicated native attempts."""
from collections import defaultdict


def compare(report):
    formal=report['formal_metric_report']
    per_model=[]
    for metric,rows in formal['metrics'].items():
        index={(r['model'],r['prompt_type']):r for r in rows}
        for model in sorted({r['model'] for r in rows}):
            absent=dict(accuracy=None,status='prompt_not_in_inventory',coverage=None,expected=0,final=0)
            m=index.get((model,'medium'),absent);d=index.get((model,'detailed'),absent)
            per_model.append(dict(model=model,metric=metric,M_accuracy=m['accuracy'],D_accuracy=d['accuracy'],
                M_status=m['status'],D_status=d['status'],M_coverage=m['coverage'],D_coverage=d['coverage'],
                M_expected=m['expected'],D_expected=d['expected'],M_final=m['final'],D_final=d['final']))
    matched=[]
    for bucket,name in [('near','NearAcc'),('mid','MidAcc'),('far','FarAcc')]:
        cells=defaultdict(dict)
        for row in report['boundary_attempts']:
            if row['bin']!=bucket or row['prompt_type'] not in ['medium','detailed']:continue
            key=(row['model'],row['seed'],tuple(row['key']))
            cells[key][row['prompt_type']]=row
        for (model,seed,key),cell in cells.items():
            m,d=cell.get('medium'),cell.get('detailed')
            if m is None or d is None:continue
            matched.append(dict(metric=name,model=model,seed=seed,reference_key=list(key),case_id=key[0],
                reference_label=m['reference_label'],M_correct=m['correct'],D_correct=d['correct'],
                both_adjudicated=m['correct'] is not None and d['correct'] is not None))
    cells=defaultdict(dict)
    for row in (report.get('pair_accuracy') or {}).get('rows',[]):
        if row['prompt_type'] in ['medium','detailed']:
            cells[(row['model'],row['seed'],row['pair_id'])][row['prompt_type']]=row
    for (model,seed,pair_id),cell in cells.items():
        m,d=cell.get('medium'),cell.get('detailed')
        if m is None or d is None:continue
        matched.append(dict(metric='PairAcc',model=model,seed=seed,reference_key=pair_id,case_id=None,
            reference_label=None,M_correct=m['correct'],D_correct=d['correct'],
            both_adjudicated=m['correct'] is not None and d['correct'] is not None))
    summaries=[]
    for metric in ['NearAcc','MidAcc','FarAcc','PairAcc']:
        for model in sorted({r['model'] for r in per_model})+['ALL_MODELS']:
            own=[r for r in matched if r['metric']==metric and (model=='ALL_MODELS' or r['model']==model)]
            known=[r for r in own if r['both_adjudicated']]
            if metric=='PairAcc':
                m=sum(r['M_correct'] for r in known)/len(known) if known else None
                d=sum(r['D_correct'] for r in known)/len(known) if known else None
            else:
                # Each identical reference point is averaged over retained seeds
                # before equal class weighting; models remain distinct units.
                points=defaultdict(list)
                for r in known:points[(r['model'],tuple(r['reference_key']))].append(r)
                classes=[[],[]]
                for values in points.values():
                    classes[values[0]['reference_label']].append((sum(v['M_correct'] for v in values)/len(values),
                                                                 sum(v['D_correct'] for v in values)/len(values)))
                m=sum(sum(v[0] for v in side)/len(side) for side in classes)/2 if all(classes) else None
                d=sum(sum(v[1] for v in side)/len(side) for side in classes)/2 if all(classes) else None
            summaries.append(dict(metric=metric,model=model,matched_expected=len(own),matched_final=len(known),
                matched_coverage=len(known)/len(own) if own else None,M_matched=m,D_matched=d,
                D_minus_M_pp=(d-m)*100 if m is not None and d is not None else None,
                D_wins=sum(r['D_correct']>r['M_correct'] for r in known),
                M_wins=sum(r['M_correct']>r['D_correct'] for r in known),
                ties=sum(r['M_correct']==r['D_correct'] for r in known),
                scope='identical model/seed/reference video/event contract/window; both adjudicated subset',
                status='matched_subset' if m is not None else 'missing_reference_class_or_adjudication'))
    cases=defaultdict(list)
    for r in report.get('task_events') or []:
        if r['prompt_type'] in ('medium','detailed'):
            cases[(r['clip_id'],r['model'],r['prompt_type'])].append(r)
    by_case=[]
    for (case,model,prompt),rows in sorted(cases.items()):
        known=[r for r in rows if r['correct'] is not None]
        by_case.append(dict(case_id=case,model=model,prompt_type=prompt,expected=len(rows),final=len(known),
            correct=sum(r['correct'] for r in known),coverage=len(known)/len(rows),
            known_only_event_accuracy=sum(r['correct'] for r in known)/len(known) if known else None,
            accuracy=sum(r['correct'] for r in known)/len(rows) if len(known)==len(rows) else None))
    complete_groups=[]
    for metric in ['NearAcc','MidAcc','FarAcc','PairAcc']:
        own=[r for r in per_model if r['metric']==metric and r['M_accuracy'] is not None and r['D_accuracy'] is not None]
        m=sum(r['M_accuracy'] for r in own)/len(own) if own else None
        d=sum(r['D_accuracy'] for r in own)/len(own) if own else None
        complete_groups.append(dict(metric=metric,models=len(own),model_names=[r['model'] for r in own],
            M_accuracy=m,D_accuracy=d,D_minus_M_pp=(d-m)*100 if own else None,
            scope='same models; both M and D have complete admitted metric groups; macro over models'))
    return dict(per_model=per_model,matched_attempts=matched,matched_summary=summaries,
                complete_group_summary=complete_groups,case_event_comparison=by_case)
