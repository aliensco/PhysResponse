"""Admission of reread trajectories, never historical labels or distance bins."""
import math


def window_interval(readback,event,spec_sha,window_sha):
    points=[];rejected=[]
    for probe in readback['probes']:
        check=probe.get('compatibility_check',{})
        reason=None
        if not probe.get('physical_context_matches_current') or not check.get('physics_equal'):
            reason='held_physics_not_verified'
        elif check.get('numerical_settings_differ') is not False:
            reason='numerical_method_not_identical'
        elif not probe.get('state_sha256') or not probe.get('source_config_sha256'):
            reason='unbound_raw_state_or_config'
        windows=[w for w in probe.get('windows',[]) if w['event']==event and w['event_spec_sha256']==spec_sha and w['window_sha256']==window_sha]
        if len(windows)!=1:reason=reason or 'exact_event_window_not_found'
        value=windows[0]['result'] if len(windows)==1 else {}
        if value.get('label') not in (0,1):reason=reason or 'event_unknown'
        # An absent contact row is not negative evidence. The producer must
        # explicitly attest complete telemetry, including the static prefix.
        completeness=probe.get('contact_completeness',{})
        if value.get('label')==0 and (not completeness.get('frames') or
                completeness.get('explicitly_complete_frames')!=completeness['frames'] or
                completeness.get('explicit_incomplete_frames')!=0 or
                completeness.get('frames_with_contact_list')!=completeness['frames']):
            reason=reason or 'negative_contact_telemetry_incomplete'
        if reason:
            rejected.append(dict(theta=probe['theta'],reason=reason));continue
        theta=probe['theta']
        if type(theta) not in (int,float) or not math.isfinite(theta):raise ValueError('Invalid probe parameter')
        points.append(dict(theta=theta,label=value['label'],state_sha256=probe['state_sha256'],
            config_sha256=probe['source_config_sha256'],context_sha256=probe['current_physical_context_sha256'],
            source_state_path=probe['source_state_path']))
    if not points:raise ValueError('No qualified trajectory points')
    if len({p['context_sha256'] for p in points})!=1:raise ValueError('Mixed held physical contexts')
    by_theta={}
    for p in points:
        if p['theta'] in by_theta and by_theta[p['theta']]['label']!=p['label']:
            raise ValueError('Conflicting outcomes at same parameter')
        by_theta[p['theta']]=p
    ordered=sorted(by_theta.values(),key=lambda p:p['theta'])
    transitions=[(a,b) for a,b in zip(ordered,ordered[1:]) if a['label']!=b['label']]
    if len(transitions)!=1:raise ValueError('A unique observed crossing is not established')
    a,b=transitions[0]
    return dict(interval=[a['theta'],b['theta']],endpoint_labels=[a['label'],b['label']],
        context_sha256=a['context_sha256'],points=ordered,rejected=rejected,
        global_uniqueness_claimed=False)


def validate_certificate(row):
    import hashlib,json
    from pathlib import Path
    cert=row['calibration_certificate']
    if cert.get('schema')!='phyresponse-archive-calibration-v13':raise ValueError('Invalid archive calibration certificate')
    path=Path(cert['proof_path']);raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=cert['proof_sha256']:raise ValueError('Archive calibration proof changed')
    proof=json.loads(raw)
    for source in proof['readbacks']:
        raw_source=Path(source['path']).read_bytes()
        if hashlib.sha256(raw_source).hexdigest()!=source['sha256']:raise ValueError('Archive readback changed')
    bound=('case_id','event','reference_video_sha256','event_contract_sha256','window_sha256',
           'context_sha256','axis','theta','interval','normalization_span','reference_label')
    matches=[p for p in proof['admitted_points'] if all(p.get(k)==row.get(k) for k in bound)]
    if len(matches)!=1:raise ValueError('Archive calibration does not bind active point')
