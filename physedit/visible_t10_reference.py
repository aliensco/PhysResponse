"""Pinned visible support-loss review of the exact selected T10 source videos."""
import functools
import json
from fractions import Fraction
from pathlib import Path
from .source_group_reference import BASE,sha

OUT=BASE/'visible_t10_v9_20261111'
POLICY=OUT/'VISIBLE_EVENT_POLICY.json'
POLICY_SHA='a60eb13fcc02f773f9a82b8111b92e8b2dbb50a2da9abb7e0251efa3b3c1973a'
SCHEMA='phyresponse-visible-support-reference-v9'
EVENT='folder_support_loss'
SPEC=dict(name='文件夹是否失去原支撑笔的支撑',
    definition='Within the registered observation window, the same initial support pen visibly ceases to hold up the same initial folder. Remaining stably supported by that pen is negative. Occlusion, disappearance, deformation or mere pen movement alone is insufficient; no centimetre measurement is required.',
    actors=['t10_supported_folder','t10_support_pen'],time_semantics='first_confirmed_occurrence')


def build_cases(source_policy):
    sources=json.loads(Path(source_policy).read_text('utf8'))['cases']
    index_path=BASE/'event_visual_review_20261111/FRAME_INDEX.json'
    index={r['case_id']:r for r in json.loads(index_path.read_text('utf8'))}
    cases={}
    for case in ('T10_C01','T10_C02','T10_C03','T10_C04','T10_C05'):
        source=sources[case];entry=index[case]
        assert entry['reference_video_sha256']==source['selected_reference_video_sha256']
        assert entry['fps']==30 and entry['frame_count']==210 and len(entry['frames'])==210
        frames=[]
        for i,f in enumerate(entry['frames']):
            path=index_path.parent/f['path'];assert f['frame']==i and sha(path)==f['sha256']
            frames.append(dict(frame=i,path=str(path),sha256=f['sha256']))
        label=int(case in ('T10_C01','T10_C02','T10_C03'))
        cases[case]=dict(video=source['selected_video'],config=source['selected_config'],frames=frames,fps=30,
            source_frame_index=dict(path=str(index_path),sha256=sha(index_path)),reference_label=label,
            positive_witness_frame=96 if label else None,negative_reviewed_through_frame=209 if not label else None,
            decision_basis=('Original pen is displaced out of its supporting role and original folder has descended by frame96; later support by the tape roll or another object does not restore the original pen support.' if label else
                'All 210 native frames reviewed: the same visible pen remains under and holds the same folder; motion/tilt alone is not support loss.'),
            review_kind='assistant_selected_source_video_review',human_gold=False,
            other_camera_trajectory_used_as_truth=False,first_occurrence_localized=False)
    return cases


def review_case(case,window):
    if window['start_frame']!=0:raise ValueError('Unsupported visible review start')
    end=Fraction(str(window['end_frame']))/Fraction(str(window['fps']))
    last=int(end*case['fps'])
    label=case['reference_label'];witness=case['positive_witness_frame']
    if last>209 or (label==1 and last<witness):
        return dict(label=None,reason='Window not covered by the pinned visible support-loss review')
    if label==0 and (case['negative_reviewed_through_frame']<last or [f['frame'] for f in case['frames'][:last+1]]!=list(range(last+1))):
        return dict(label=None,reason='Negative lacks full native-frame window review')
    return dict(label=label,positive_witness_frame=witness,source_last_frame=last,
        positive_witness_time_s=witness/30 if witness is not None else None,
        first_time_s=None,first_time_status='not_localized; witness is an upper bound, not an invented onset',
        reviewed_frames=[0,witness] if label else list(range(last+1)),
        basis=case['decision_basis'])


@functools.lru_cache(maxsize=8)
def verified_case(policy_sha,case_id):
    if policy_sha!=POLICY_SHA or sha(POLICY)!=POLICY_SHA:raise ValueError('Changed visible support review policy')
    case=json.loads(POLICY.read_text('utf8'))['cases'][case_id]
    for item in [case['video'],case['config'],case['source_frame_index'],*case['frames']]:
        if sha(item['path'])!=item['sha256']:raise ValueError('Changed selected-video review evidence')
    return case


def validate(row,digest_json):
    cert=row.get('reference_certificate',{})
    if (cert.get('schema')!=SCHEMA or cert.get('policy_path')!=str(POLICY)
            or digest_json(cert)!=row.get('reference_evidence_sha256')):
        raise ValueError('Unbound visible support reference')
    case=verified_case(cert.get('policy_sha256'),row['case_id'])
    if row['event']!=EVENT or row.get('state_sha256') is not None or row.get('reference_basis')!='selected_reference_video_review_v9':
        raise ValueError('Visible support cannot reuse a numeric label or substitute a state')
    for k in ('case_id','event','reference_video_sha256','state_sha256','event_contract_sha256','window_sha256','reference_label'):
        if cert.get(k)!=row.get(k):raise ValueError('Visible support certificate mismatch')
    p=json.loads(POLICY.read_text('utf8'));key='|'.join((row['case_id'],row['event_contract_sha256'],row['window_sha256']))
    registered=p['windows'].get(key)
    if not registered or registered['event_spec']!=SPEC or cert.get('event_spec_sha256')!=digest_json(SPEC):
        raise ValueError('Changed visible support event definition')
    if case['video']['sha256']!=row['reference_video_sha256']:raise ValueError('Wrong reference video')
    measured=review_case(case,registered['window'])
    if measured['label'] is None or measured['label']!=row['reference_label'] or row['measurement']!=measured:
        raise ValueError('Visible support reference outside adjudicated window')
