"""Reviewed Boolean implications over frozen original human observations.

These are derived labels, not new human annotations or numeric measurements.
Only enumerated necessary components are used. Missing quantitative conjuncts
can never become true merely because all observed contacts are true.
"""
import hashlib
import json


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()


def profile(task, event):
    scene=task['clip_id'][:3]
    spec=task['event_contract']['events'][event]
    visible_profiles={
        ('T01','can2_mug_contact_and_visible_response'):('b399cf637f4a8ccc530fb0ffa045b84134c86182418666dc13e7ea2c5238197e',['can2_mug_contact'],None),
        ('T04','can2_mug_contact_and_visible_response'):('d2eca4c61a097e85c707e9b90bad4d5486d5e403a00805d3f970279c82902d32',['can2_mug_contact'],None),
        ('T03','phone_contact_chain_and_visible_fall'):('6f5847f82b178a99a1ec76fcda88fe982f560d9b2372d1fad868733aa40f3071',['obs_book_contacts_upper_can','obs_can_lands_on_lower_shelf','can_phone_contact','phone_floor_contact'],None),
        ('T14','marker_contact_after_visible_full_exit'):('ad99904598fa89a0c94417f296d92aab15c03224c37f4b416da90ddea997094b',['marker_eraser_contact'],None),
        ('T20','ordered_contacts_and_visible_cup_response'):('84962daebc3ec23ef0fc7bf46fd0009588c7bca46b2ba633ca3926c13e5fb177',['book_can_contact','can_box_contact','box_cup_contact'],'three_contacts_nondecreasing')}
    if (scene,event) in visible_profiles:
        pin,contacts,order=visible_profiles[scene,event]
        if digest(spec.get('definition'))!=pin:raise ValueError('Unexpected revised visible conjunction')
        result=dict(contacts=contacts,complete=False)
        if order:result['order']=order
        return result
    if scene=='T03' and event=='phone_edge_fall_after_can_contact':
        expected = ('The original can contacts the original lower phone; subsequently the same phone passes the lower support edge, loses support and begins moving down. Floor impact and a 20cm displacement are separate outcomes. Contact timing is an observed prerequisite, not proof of physical causality.')
        if (spec.get('definition') != expected or set(spec.get('actors', [])) !=
                {'t03v5_upper_edge_can', 't03v5_lower_edge_phone', 't03v5_lower_transfer_ramp'}):
            raise ValueError('Unexpected T03 visible fall contract')
        return dict(contacts=['can_phone_contact','phone_edge_fall'],
                    order='can_phone_before_fall',complete=True)
    if scene=='T03' and event=='phone_second_fall_original':
        if spec.get('source_boolean_expression')!='book_can AND can_shelf AND can_phone AND phone_floor AND phone_drop':
            raise ValueError('Unexpected T03 Boolean contract')
        return dict(contacts=['obs_book_contacts_upper_can','obs_can_lands_on_lower_shelf','can_phone_contact','phone_floor_contact'],complete=False)
    if scene=='T04' and event=='original_full_chain':
        if (not spec.get('definition','').startswith('The original t04v3_chain_can_2 contacts the original t04v3_chain_mug, AND')
                and spec.get('source_boolean_expression')!='can2_mug_contact AND (rotation OR displacement)'):
            raise ValueError('Unexpected T04 Boolean contract')
        return dict(contacts=['can2_mug_contact'],complete=False)
    if scene=='T06' and event=='complete_required_contact_chain':
        names=spec.get('source_conjunction',{}).get('all',[])
        pairs=spec.get('required_contact_pairs',[])
        if not names or len(pairs)!=len(names) or names!=[f'adjacent_contact_{i}' for i in range(len(pairs))]:
            raise ValueError('Unexpected T06 contact chain')
        return dict(contacts=names,complete=True)
    if scene=='T13' and event=='registered_two_contact_conjunction_and_order':
        if 'strictly earlier' not in spec.get('definition',''):
            raise ValueError('Unexpected T13 strict order')
        return dict(contacts=['obs_driver_to_falling_book_contact','obs_falling_book_to_drink_contact'],order='two_contacts_strict_order',complete=True)
    if scene=='T14' and event=='terminal_contact_after_full_exit':
        if spec.get('source_conjunction')!={'all':['passed_channel','contacted_eraser'],'order':[['passed_channel','contacted_eraser']]}:
            raise ValueError('Unexpected T14 conjunction')
        return dict(contacts=['marker_eraser_contact'],complete=False)
    if scene=='T20' and event=='complete_propagation_original':
        if spec.get('source_boolean_expression')!='ordered_three_contact_chain AND cup_response':
            raise ValueError('Unexpected T20 conjunction')
        return dict(contacts=['book_can_contact','can_box_contact','box_cup_contact'],
                    order='three_contacts_nondecreasing',complete=False)
    return None


def validate_profile(task,event,observation):
    p=profile(task,event)
    if p is None:raise ValueError('Unreviewed Boolean profile')
    questions=observation.get('questions',{}).get('events',{})
    for name in p['contacts']:
        q=questions.get(name,{})
        original_id=name.removeprefix('obs_') if task['clip_id'].startswith('T13') else event
        if (q.get('relation') not in ('visible_original_event','source_component')
                or original_id not in q.get('original_event_ids',[]) or q.get('prerequisites')
                or q.get('answer_style')!='event' or not q.get('definition')):
            raise ValueError('Unbound Boolean component')
        if task['clip_id'].startswith('T13') and set(q.get('actors',[]))!=set(task['event_contract']['events'][original_id]['actors']):
            raise ValueError('T13 original actor mismatch')
    if p.get('order'):
        q=questions.get(p['order'],{})
        if (q.get('relation')!='source_component' or q.get('answer_style')!='order'
                or q.get('prerequisites')!=p['contacts'] or event not in q.get('original_event_ids',[])):
            raise ValueError('Unbound order prerequisite')
    if task['clip_id'].startswith('T06'):
        pairs=task['event_contract']['events'][event]['required_contact_pairs']
        if any(set(questions[q].get('actors',[]))!=set(pair) for q,pair in zip(p['contacts'],pairs)):
            raise ValueError('Adjacent-book actor mismatch')
    return p


def infer(profile, human):
    if human.get('measurability')!='measurable':return None,[]
    values=human.get('events',{})
    negative=[q for q in profile['contacts'] if values.get(q)=='no']
    if negative:return 0,negative
    contacts_complete=all(values.get(q)=='yes' for q in profile['contacts'])
    if not contacts_complete:return None,[]
    order=profile.get('order')
    if order and values.get(order)=='no':return 0,[*profile['contacts'],order]
    if order and values.get(order)!='yes':return None,[]
    if profile['complete']:return 1,[*profile['contacts'],*([order] if order else [])]
    return None,[]


def load_rules(path,tasks,observations):
    if path is None:return {}
    data=json.loads(path.read_text(encoding='utf-8-sig'))
    if data.get('schema')!='phyresponse-native-boolean-observations-v1':raise ValueError('Invalid Boolean rule schema')
    by_task={t['id']:t for t in tasks}; by_obs={o['id']:o for o in observations}; result={}
    for r in data['records']:
        key=(r['task_id'],r['event']);t=by_task.get(key[0]);o=by_obs.get(key[0])
        if t is None or o is None or key in result:raise ValueError('Unbound Boolean rule')
        p=validate_profile(t,key[1],o)
        qs=[*p['contacts'],*([p['order']] if p.get('order') else [])]
        if (r.get('profile')!=p or r.get('status')!='verified' or not r.get('evidence')
                or r.get('event_contract_sha256')!=t['event_contract_sha256']
                or r.get('source_event_spec_sha256')!=digest(t['event_contract']['events'][key[1]])
                or r.get('observation_contract_sha256')!=o['observation_contract_sha256']
                or r.get('window_sha256')!=digest(t['window'])
                or r.get('question_sha256')!={q:digest(o['questions']['events'][q]) for q in qs}):
            raise ValueError('Boolean evidence binding mismatch')
        result[key]=r
    return result
