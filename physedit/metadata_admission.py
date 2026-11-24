"""Reference-only calibration candidates from explicitly comparable metadata.

This does not infer a physical outcome from a parameter. Outcomes come from
the already qualified current reference event/window table. No model outputs
or historical event labels are inputs.
"""
import copy
import math


def clean_experiment(value,axis):
    if isinstance(value,list):return [clean_experiment(x,axis) for x in value]
    if not isinstance(value,dict):return value
    result={}
    ignored={'sample_id','trajectory_id','random_seed','seed','camera_id','render_preset','counterfactual_region',
             'expected_outcomes','parameter_region','replicate_id','sampling_design_id',
             'formal_design','formal_registration','formal_sampling',
             'parameter_id','counterfactual_group_id','cross_background_physics_group_id','case'}
    for k,v in value.items():
        if k in ignored:continue
        if axis=='gravity' and k in {'gravity','gravity_id','gravity_vector','gravity_vector_mps2','gravity_magnitude_mps2','gravity_mps2','gravity_magnitude'}:
            result[k]='gravity_axis';continue
        if axis.startswith('speed:') and k in {'initial_speed_mps','book_initial_speed_mps','stapler_initial_speed_mps',
            'commanded_extraction_speed_mps','initial_support_bar_linear_momentum_kg_mps','initial_support_bar_velocity_world_mps',
            'declared_one_shot_impulse_magnitude_Ns','declared_one_shot_impulse_vector_Ns'}:
            result[k]='speed_axis';continue
        if axis.startswith('speed:') and k=='parameter' and isinstance(v,dict) and 'speed' in v.get('parameter_name',''):
            result[k]=clean_experiment({a:('speed_axis' if a in {'parameter_value','parameter_index','normalized_parameter_value'} else b) for a,b in v.items()},axis);continue
        result[k]=clean_experiment(v,axis)
    return result


def signature(config,axis):
    c=copy.deepcopy(config)
    for k in ('sample_id','dataset_version','seed','random_seed','annotations','camera_id','show_viewer'):c.pop(k,None)
    render=c.get('render',{})
    c['render']={k:render[k] for k in ('fps','frames','pre_static_frames','physics_steps_per_render_frame','physics_backend','precision') if k in render}
    override=c.get('scene_overrides',{})
    c['scene_overrides']={k:v for k,v in override.items() if not k.startswith('raytracing')}
    c['experiment']=clean_experiment(c.get('experiment',{}),axis)
    objects=c.get('dynamic_objects',[])
    if axis=='gravity':
        vector=c['scene_overrides'].get('gravity')
        if not isinstance(vector,list) or len(vector)!=3 or vector[:2]!=[0.,0.] or vector[2]>=0:raise ValueError('No explicit vertical gravity vector')
        theta=-vector[2];c['scene_overrides']['gravity']=[0.,0.,'gravity_axis']
        if 'gravity_id' in c['scene_overrides']:c['scene_overrides']['gravity_id']='gravity_axis'
    elif axis.startswith('speed:'):
        actor=axis.split(':',1)[1];obj=next((o for o in objects if o['name']==actor),None)
        body=(obj or {}).get('spec',obj or {})
        vector=body.get('motion',{}).get('initial_velocity')
        if not isinstance(vector,list) or len(vector)!=3:raise ValueError('No explicit initial velocity')
        theta=math.sqrt(sum(v*v for v in vector))
        if theta<=0:raise ValueError('Zero speed has no registered intervention direction')
        body['motion']['initial_velocity']=[round(v/theta,12) for v in vector]
        def velocity_mirrors(value):
            if isinstance(value,list):return [velocity_mirrors(v) for v in value]
            if not isinstance(value,dict):return value
            return {k:([round(v/theta,12) for v in vector] if k in {'initial_velocity_mps','stapler_initial_velocity_mps'} and v==vector else velocity_mirrors(v)) for k,v in value.items()}
        c['experiment']=velocity_mirrors(c['experiment'])
        if 't12_parameters' in c:c['t12_parameters']=velocity_mirrors(clean_experiment(c['t12_parameters'],axis))
    elif axis.startswith('friction:'):
        names=axis.split(':',1)[1].split(',')
        bodies=[o.get('spec',o) for o in objects if o['name'] in names]
        if len(bodies)!=len(names):raise ValueError('Friction actor identity missing')
        values=[o.get('friction') for o in bodies]
        if not values or any(type(v) not in (int,float) or v!=values[0] for v in values):
            raise ValueError('Shared friction axis requires identical explicit body coefficients')
        theta=values[0]
        for body in bodies:body['friction']='friction_axis'
        def mirrors(value):
            if isinstance(value,list):return [mirrors(v) for v in value]
            if not isinstance(value,dict):return value
            return {k:('friction_axis' if ('friction' in k or k in {'parameter_value','controlled_value'}) and type(v) in (int,float) and abs(v-theta)<1e-7 else mirrors(v)) for k,v in value.items()}
        c['experiment']=mirrors(c['experiment'])
        parameter=c['experiment'].get('parameter',{})
        if 'friction' in parameter.get('parameter_name',''):
            for key in ('parameter_index','normalized_parameter_value'):parameter.pop(key,None)
    elif axis.startswith('gap:'):
        parameter=axis.split(':',1)[1]
        names={'gap_ratio':('t05v2_guide_left','t05v2_guide_right'),
               'channel_throat_width_m':('t14v2_corridor_book_left','t14v2_corridor_book_right')}[parameter]
        named={o['name']:o for o in objects};left,right=[named[n] for n in names]
        lp,rp=left['pose']['position'],right['pose']['position'];lw,rw=left['size'][1],right['size'][1]
        gap=(rp[1]-rw/2)-(lp[1]+lw/2)
        exp=c['experiment'];declared=exp['parameter'][parameter]
        width=exp['parameter']['phone_nominal_width_m'] if parameter=='gap_ratio' else 1.
        theta=gap/width
        if abs(theta-declared)>1e-10:raise ValueError('Declared gap differs from explicit fixed collision geometry')
        center=(lp[1]+rp[1])/2
        lp[1]=round(center-lw/2,12);rp[1]=round(center+rw/2,12)
        for key in (parameter,'gap_width_m'):
            if key in exp['parameter']:exp['parameter'][key]='gap_axis'
        geometry=exp.get('geometry_contract',{})
        for key in ('guide_gap_width_m','channel_geometric_clearance_m','channel_pose_clearance_m'):
            if key in geometry:geometry[key]='gap_axis'
        for key,body in (('book_left_center_m',left),('book_right_center_m',right)):
            if key in geometry:geometry[key][1]=body['pose']['position'][1]
        c.pop('formal_sampling',None)
    elif axis=='command:driver_initial_speed_mps':
        exp=c['experiment'];parameter=exp['parameter'];theta=parameter['parameter_value']
        if parameter.get('initial_velocity_application_mode')!='zero' or parameter.get('trigger_type')!='finite_force_until_second_contact':
            raise ValueError('T11 force-command implementation is not explicit')
        if type(theta) not in (float,int) or not 1.8<=theta<=3.6:raise ValueError('Command outside declared authoring branch')
        fraction=(theta-1.8)/1.8;before=.18+.12*fraction;after=.6*fraction;delay=int(round(20*fraction))
        force=next(f for f in c['runtime_forces'] if f['target']=='t11_driver_eraser')
        if abs(force['force_world'][0]-before)>1e-12 or abs(force['force_world_after_contact'][0]-after)>1e-12 or force['stop_after_contact_delay_frames']!=delay:
            raise ValueError('Force command differs from registered original authoring equations')
        force['force_world'][0]='command_axis';force['force_world_after_contact'][0]='command_axis';force['stop_after_contact_delay_frames']='command_axis'
        for key in ('parameter_value','driver_force_before_first_contact_n','driver_force_after_second_contact_n','driver_force_after_second_contact_frames'):
            parameter[key]='command_axis'
    elif axis=='support_angle_deg':
        exp=c['experiment'];theta=exp['support_angle_deg'];z=exp['surface_z'];contact=exp['contact_position_normalized']
        named={o['name']:o for o in objects};folder=named['t16_folder'];book=named['t16_support_book']
        length,thickness=folder['size'][0],folder['size'][2];angle=math.radians(theta)
        expected_z=z+.0015+abs(math.sin(angle))*length/2+abs(math.cos(angle))*thickness/2
        expected_x=folder['pose']['position'][0]+(contact-.5)*math.cos(angle)*length+book['size'][2]/2+.002
        if abs(folder['pose']['position'][2]-expected_z)>1e-10:
            raise ValueError('T16 poses differ from pinned original support-angle authoring equations')
        if folder['pose']['euler_xyz']!=[0.,-theta,0.]:raise ValueError('T16 folder orientation differs from declared angle')
        # Some exact references contain a recorded support-clearance repair.
        # Preserve that fixed offset in the context; never merge it with an
        # unrepaired reference merely because their angle names match.
        actual_x=book['pose']['position'][0]
        canonical_x=actual_x+(contact-.5)*(1-math.cos(angle))*length
        folder['pose']['position'][2]=z+.0015+thickness/2;folder['pose']['euler_xyz'][1]=0.
        book['pose']['position'][0]=canonical_x
        for frame in book.get('motion',{}).get('keyframes',[]):
            if abs(frame['position'][0]-actual_x)>1e-10:raise ValueError('Support withdrawal includes an uncontrolled x change')
            frame['position'][0]=canonical_x
        exp['support_angle_deg']='support_angle_axis'
        p=exp.get('parameter',{})
        if p.get('parameter_name')=='support_angle':
            for key in ('parameter_value','parameter_index','normalized_parameter_value'):p.pop(key,None)
    else:raise ValueError('Unsupported metadata intervention axis')
    return theta,c


def axes(config):
    result=['gravity']
    friction_groups={}
    for obj in config.get('dynamic_objects',[]):
        body=obj.get('spec',obj)
        vector=body.get('motion',{}).get('initial_velocity')
        if isinstance(vector,list) and len(vector)==3 and any(vector):result.append('speed:'+obj['name'])
        mu=body.get('friction')
        if type(mu) in (int,float):
            result.append('friction:'+obj['name'])
            friction_groups.setdefault(mu,[]).append(obj['name'])
    for names in friction_groups.values():
        if len(names)>1:result.append('friction:'+','.join(sorted(names)))
    parameter=config.get('experiment',{}).get('parameter',{})
    for key in ('gap_ratio','channel_throat_width_m'):
        if key in parameter:result.append('gap:'+key)
    if parameter.get('initial_velocity_application_mode')=='zero' and parameter.get('trigger_type')=='finite_force_until_second_contact':
        result.append('command:driver_initial_speed_mps')
    if config.get('experiment',{}).get('template_id')=='T16' and 'support_angle_deg' in config.get('experiment',{}):result.append('support_angle_deg')
    return result


def normalization(items,axis,original_designs):
    # Prefer the normalized coordinates explicitly recorded in CURRENT source
    # metadata. Their affine relation recovers the declared full sweep span.
    coordinates=[]
    for item in items:
        p=item['config'].get('experiment',{}).get('parameter',{})
        if axis.startswith('speed:') and 'speed' in p.get('parameter_name','') and abs(p.get('parameter_value',math.inf)-item['theta'])<1e-9:
            z=p.get('normalized_parameter_value')
            if isinstance(z,(int,float)):coordinates.append((item['theta'],z))
    for a in coordinates:
        for b in coordinates:
            if abs(a[1]-b[1])<1e-12:continue
            span=(b[0]-a[0])/(b[1]-a[1]);lo=a[0]-a[1]*span
            if span>0 and all(abs(theta-(lo+z*span))<1e-9 for theta,z in coordinates):
                return [lo,lo+span],'current_normalized_metadata_coordinates'
    domains=[]
    if axis.startswith(('speed:','friction:','gap:','command:')) or axis=='support_angle_deg':
        for item in items:
            old=original_designs.get(item['case_id'],{});domain=old.get('normalization_domain')
            kind='speed' if axis.startswith('speed:') else 'friction' if axis.startswith('friction:') else axis if ':' not in axis else axis.split(':',1)[1]
            if kind in old.get('parameter','') and isinstance(domain,list) and len(domain)==2 and domain[1]>domain[0]:domains.append(domain)
    if domains:
        lo=min(d[0] for d in domains);hi=max(d[1] for d in domains)
        if all(lo-1e-9<=item['theta']<=hi+1e-9 for item in items):return [lo,hi],'original_parameter_design_domain_only; historical labels/distances unused'
    values=[item['theta'] for item in items]
    if len(set(values))<2:raise ValueError('No parameter span')
    return [min(values),max(values)],'observed_current_fixed_reference_metadata_range; not a solved critical value'


def crossings(points):
    grouped={}
    for p in points:
        old=grouped.get(p['theta'])
        if old is not None and old['label']!=p['label']:raise ValueError('Same parameter has conflicting current reference outcomes')
        grouped[p['theta']]=p
    ordered=sorted(grouped.values(),key=lambda p:p['theta'])
    return [(a,b) for a,b in zip(ordered,ordered[1:]) if a['label']!=b['label']]
