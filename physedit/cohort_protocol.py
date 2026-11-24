"""Complete retained-cohort admission and exact registered event-vector scoring."""
import hashlib
import json
from collections import Counter,defaultdict
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()


def validate(path,tasks,truth,missing,reference_key):
    data=json.loads(Path(path).read_text('utf8'))
    if data.get('schema')!='phyresponse-full-cohort-protocol-v10':raise ValueError('Invalid full-cohort protocol')
    for name in ('original_split','source_metadata','accepted_input_authority'):
        source=data.get(name)
        if source and hashlib.sha256(Path(source['path']).read_bytes()).hexdigest()!=source['sha256']:
            raise ValueError('Full-cohort source authority changed')
    members=data.get('members',[]);ids=[r['case_id'] for r in members]
    actual=sorted({t['clip_id'] for t in tasks}|{r['clip_id'] for r in missing})
    if len(ids)!=len(set(ids)) or len(ids)!=data['expected_cases'] or sorted(ids)!=actual:
        raise ValueError('Full-cohort member missing, duplicated or substituted')
    if len(tasks)!=data['actual_media_tasks'] or len(tasks)+len(missing)!=data['expected_generation_slots']:
        raise ValueError('Full-cohort generation inventory differs')
    if all(k in data for k in ('models','prompt_types','seeds')):
        expected={(c,m,p,s) for c in ids for m in data['models'] for p in data['prompt_types'] for s in data['seeds']}
        slots=[(r['clip_id'],r['model'],r['prompt_type'],r['seed']) for r in [*tasks,*missing]]
        if len(slots)!=len(set(slots)) or set(slots)!=expected:raise ValueError('Full-cohort model/prompt/seed roster differs')
    for member in members:
        own=[t for t in tasks if t['clip_id']==member['case_id']]
        if any(t['reference_video_sha256']!=member['reference_video_sha256'] or t['condition_sha256']!=member['condition_sha256'] for t in own):
            raise ValueError('Full-cohort source media or condition substituted')
        task_ids=sorted(t['id'] for t in own)
        keys=sorted({reference_key(t,e) for t in own for e in t['event_ids']})
        if digest(task_ids)!=member['task_ids_sha256'] or digest(keys)!=member['reference_keys_sha256']:
            raise ValueError('Full-cohort contract/window inventory differs')
        if any(k not in truth or truth[k]['status']!='verified' for k in keys):
            raise ValueError('Full-cohort reference vector is not admitted')
    authority=None
    if data.get('accepted_input_authority'):
        authority=json.loads(Path(data['accepted_input_authority']['path']).read_text('utf8'))
        if authority.get('authority')!='current_actual_phyresponse_4000_inputs':
            raise ValueError('Unsupported accepted input authority')
        bindings=[dict(case_id=m['case_id'],reference_video_sha256=m['reference_video_sha256'],condition_sha256=m['condition_sha256']) for m in members]
        if authority.get('current_case_bindings_sha256')!=digest(sorted(bindings,key=lambda r:r['case_id'])):
            raise ValueError('Accepted current input bindings changed')
    return dict(schema=data['schema'],registered_cases=len(ids),registered_scene_count=len({c.split('_')[0] for c in ids}),
        evaluation_input_authority=authority['authority'] if authority else None,
        original_version_differences_accepted=bool(authority),
        original_membership_preserved=None,original_case_id_roster_preserved=True,
        original_physical_membership_verified=False,
        source_identity_note='Case-ID coverage alone does not prove background-only replacement or preservation of original physical conditions.',
        reference_admission_complete=True,all_cases_evaluated=True,
        expected_generation_slots=data['expected_generation_slots'],actual_media_tasks=len(tasks),missing_media_slots=len(missing),
        original_split_path=data['original_split']['path'],original_split_sha256=data['original_split']['sha256'],
        critical_strata_complete=False,scope='all original cases in common evaluation; native-distance/pair eligibility is a separate field, never a case exclusion')


def event_vectors(tasks,rows,missing):
    observed={(r['task_id'],r['event']):r for r in rows}
    if len(observed)!=len(rows):raise ValueError('Duplicate event observation in exact-vector scoring')
    results=[]
    for task in tasks:
        event_ids=task.get('event_ids') or list(task['event_contract']['events'])
        own=[observed.get((task['id'],e),{}) for e in event_ids]
        wrong=[e for e,r in zip(event_ids,own) if r.get('correct')==0]
        known=sum(r.get('correct') is not None for r in own)
        # A known mismatch disproves exact equality; this does not assign any
        # label to another unobserved event. Positive requires the whole vector.
        exact=0 if wrong else 1 if own and known==len(own) else None
        results.append(dict(task_id=task['id'],case_id=task['clip_id'],model=task['model'],prompt_type=task['prompt_type'],seed=task['seed'],
            exact=exact,expected_events=len(own),final_events=known,mismatched_events=wrong,
            unknown_events=[e for e,r in zip(event_ids,own) if r.get('correct') is None],
            decision='known_event_mismatch' if wrong else 'all_registered_events_match' if exact==1 else 'incomplete_event_evidence',
            source_event_contract_sha256=task['event_contract_sha256']))
    for row in missing:
        results.append(dict(task_id=None,case_id=row['clip_id'],model=row['model'],prompt_type=row['prompt_type'],seed=row['seed'],
            exact=None,expected_events=None,final_events=0,mismatched_events=[],unknown_events=[],decision='generation_missing',source_event_contract_sha256=None))
    def summary(own):
        known=[r['exact'] for r in own if r['exact'] is not None]
        return dict(expected=len(own),final=len(known),unknown=len(own)-len(known),correct=sum(known),
            accuracy=sum(known)/len(own) if own and len(known)==len(own) else None,
            known_only_accuracy=sum(known)/len(known) if known else None,coverage=len(known)/len(own) if own else None,
            status='complete' if own and len(known)==len(own) else 'partial')
    by_group=defaultdict(list);by_case=defaultdict(list)
    for r in results:by_group[(r['model'],r['prompt_type'])].append(r);by_case[r['case_id']].append(r)
    return dict(overall=summary(results),by_model_prompt=[dict(model=m,prompt_type=p,**summary(own)) for (m,p),own in sorted(by_group.items())],
        by_case=[dict(case_id=c,**summary(own)) for c,own in sorted(by_case.items())],rows=results,
        aggregation='exact equality of registered target event vector; decisive known mismatch=0, every event matched=1, otherwise unknown',
        not_a_claim_all_auxiliary_source_graph_nodes_are_annotated=True)
