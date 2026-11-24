"""Versioned published-physics-group oracle, explicitly not camera execution GT."""
import copy
import gzip
import hashlib
import importlib.util
import json
import math
from fractions import Fraction
from pathlib import Path

BASE=Path('review/phyresponse_gold4000_20261106/reference_event_truth_20261110')
OUT=BASE/'group_reference_md_v8_20261111'
LEDGER=OUT/'GROUP_REFERENCE_POLICY.json'
LEDGER_SHA256='2d38f512a40ad118e212f773aef3ee2255381948700bece0735e510352f56b5f'
SCHEMA='phyresponse-published-group-oracle-v1'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state(path):
    raw=Path(path).read_bytes()
    return json.loads(gzip.decompress(raw) if raw[:2]==bytes((31,139)) else raw)


def physical_context(config,native):
    objects={x['name']:x for x in config['dynamic_objects']}
    actual={x['name']:x['spec'] for x in native['dynamic_objects']}
    if objects!=actual:raise ValueError('Group oracle body definitions differ')
    allowed={'raytracing_env_map_required','raytracing_env_map','raytracing_env_encoding','raytracing_lights','raytracing_env_euler'}
    clean=lambda d:{k:v for k,v in d.items() if k not in allowed}
    if clean(config['scene_overrides'])!=clean(native['scene_overrides']):
        raise ValueError('Group oracle physical solver/gravity differs')
    forces=lambda data:[{k:v for k,v in r.items() if k!='profile'} for r in data.get('runtime_forces') or []]
    if forces(config)!=forces(native):raise ValueError('Group oracle force schedule differs')
    exp=lambda d:{k:v for k,v in d.items() if k!='parameter'}
    if exp(config['experiment'])!=exp(native['experiment']):raise ValueError('Group oracle event/context differs')
    timing={k:config['render'][k] for k in ('fps','frames','pre_static_frames','physics_steps_per_render_frame')}
    if any(native['render'].get(k)!=v for k,v in timing.items()) or config.get('manifest')!=native.get('manifest'):
        raise ValueError('Group oracle environment or native clock differs')
    return dict(objects=objects,physical_overrides=clean(config['scene_overrides']),runtime_forces=forces(config),
                experiment=exp(config['experiment']),timing=timing,manifest=config.get('manifest'))


def static_prefix_proof(native,pairs):
    """Conservative X bounds under both legacy quaternion orders; no RGB metric."""
    hold=native['render']['pre_static_frames'];frames=native['frames'];specs={o['name']:o['spec'] for o in native['dynamic_objects']}
    def extent(obj,spec):
        q=obj['quaternion_xyzw'];norm=math.sqrt(sum(x*x for x in q))
        if not math.isclose(norm,1,abs_tol=1e-5):raise ValueError('Invalid source orientation')
        q=[v/norm for v in q];extents=[]
        for x,y,z,w in [q,[q[1],q[2],q[3],q[0]]]:
            rotation=(1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w))
            extents.append(sum(abs(a)*b/2 for a,b in zip(rotation,spec['collision_proxy_size'])))
        return max(extents)+spec.get('contact_offset',0)
    proofs=[]
    for a,b in pairs:
        oa=next(o for o in frames[0]['objects'] if o['name']==a);ob=next(o for o in frames[0]['objects'] if o['name']==b)
        if any(next(o for o in f['objects'] if o['name']==a)!=oa or next(o for o in f['objects'] if o['name']==b)!=ob for f in frames[:hold]):
            raise ValueError('Omitted prefix is not a recorded static hold')
        gap=abs(ob['position'][0]-oa['position'][0])-extent(oa,specs[a])-extent(ob,specs[b])
        if gap<=0:raise ValueError('Omitted static-prefix contact cannot be excluded')
        proofs.append(dict(actors=[a,b],minimum_conservative_x_gap_m=gap,frames=list(range(hold)),
                           orientation_rule='maximum enclosing X extent across both legacy quaternion orders'))
    return proofs


def evaluate(native,window,event,event_spec):
    spec=importlib.util.spec_from_file_location('group_oracle_event_rules',BASE/'event_rules.py')
    rules=importlib.util.module_from_spec(spec);spec.loader.exec_module(rules)
    if window['start_frame']!=0 or native['render']['fps']!=30:raise ValueError('Unsupported source window/clock')
    end=Fraction(str(window['end_frame']))/Fraction(str(window['fps']))
    proofs=[];working=native
    if event=='complete_required_contact_chain':
        proofs=static_prefix_proof(native,event_spec['required_contact_pairs'])
        working=copy.deepcopy(native)
        for frame in working['frames'][:native['render']['pre_static_frames']]:
            if 'contacts' not in frame:frame['contacts']=[]
    trace=rules.Trace(working,dict(decoded_fps=30,frame_count=len(native['frames'])),end)
    if event=='support_pen_displacement_reaches_source_threshold':
        clauses=event_spec['metric_rule']['clauses']
        if (event_spec['actors']!=['t10_support_pen'] or len(clauses)!=1
                or clauses[0]['quantity'] not in ['support_pen_max_displacement_m','pen_max_center_displacement_m']
                or clauses[0]['operator']!='>=' or clauses[0]['threshold']!=.04):
            raise ValueError('Changed pen event/threshold')
        positions=trace.positions('t10_support_pen')
        result=trace.threshold([math.dist(positions[0],p) for p in positions],'>=',.04,
                               clauses[0]['quantity'],False)
    else:result=rules.evaluate(trace,event,event_spec)
    return result,proofs


def validate(row,digest_json):
    cert=row.get('reference_certificate',{})
    if (cert.get('schema')!=SCHEMA or cert.get('policy_sha256')!=LEDGER_SHA256 or sha(LEDGER)!=LEDGER_SHA256
            or cert.get('policy_path')!=str(LEDGER) or digest_json(cert)!=row.get('reference_evidence_sha256')):
        raise ValueError('Invalid published group-oracle policy binding')
    policy=json.loads(LEDGER.read_text(encoding='utf8'));case=policy['cases'].get(row['case_id'])
    if not case or row.get('reference_basis')!='published_physics_group_oracle_v8' or row.get('selected_camera_execution_verified') is not False:
        raise ValueError('Group oracle cannot claim selected-camera execution verification')
    if row['reference_video_sha256']!=case['selected_reference_video_sha256'] or row['state_sha256']!=case['state_sha256']:
        raise ValueError('Group oracle media/state mismatch')
    key='|'.join((row['event'],row['event_contract_sha256'],row['window_sha256']))
    registered=case['windows'].get(key)
    if not registered or cert['event_spec_sha256']!=digest_json(registered['event_spec']):raise ValueError('Unregistered group event/window')
    for name in ['case_id','event','reference_video_sha256','state_sha256','event_contract_sha256','window_sha256','reference_label']:
        if cert.get(name)!=row.get(name):raise ValueError('Group reference row/certificate mismatch')
    for item in [case['state'],case['selected_config'],case['selected_video']]:
        if sha(item['path'])!=item['sha256']:raise ValueError('Group source bytes changed')
    if sha(BASE/'event_rules.py')!=policy['event_rules_sha256']:raise ValueError('Group event extractor changed')
    native=state(case['state']['path']);config=json.loads(Path(case['selected_config']['path']).read_text(encoding='utf8'))
    if digest_json(physical_context(config,native))!=case['physical_context_sha256']:raise ValueError('Group physical context changed')
    result,proofs=evaluate(native,registered['window'],row['event'],registered['event_spec'])
    if result['label']!=row['reference_label'] or result['label'] is None:raise ValueError('Group label is not reproduced by source evidence')
    if row.get('measurement')!=result or row.get('static_prefix_proof')!=proofs:raise ValueError('Group measurements changed')
