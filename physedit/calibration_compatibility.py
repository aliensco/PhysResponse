"""Physics compatibility for counterfactual scans, separate from source truth.

This never certifies that two rendered videos are the same execution. A visual
background can be removed from a scan context only when every external collision
channel is explicitly disabled. Actual new-reference labels remain authoritative.
"""
import copy
from .metadata_admission import signature

EXTERNAL_COLLISIONS=('static_scene_collision','layout_collision','room_wall_colliders','add_ground_plane')
VISUAL_FIELDS={'color','roughness','metallic','diffuse_texture_path','camera_focus','visual_quality_baseline',
               'visualization','visual_mesh_path','preserve_mesh_material','semantic_label','category'}


def unwrap(config):
    if isinstance(config.get('physics'),dict) and isinstance(config['physics'].get('source_config'),dict):
        return config['physics']['source_config']
    return config


def projection(config,axis):
    config=unwrap(config)
    if not config.get('dynamic_objects'):raise ValueError('Explicit actor physics is unavailable')
    _,value=signature(config,axis)
    objects=copy.deepcopy(value['dynamic_objects'])
    for wrapper in objects:
        body=wrapper.get('spec',wrapper)
        for key in VISUAL_FIELDS:body.pop(key,None)
        if 'collision_mesh_path' in body:body.pop('mesh_path',None)
    overrides=value.get('scene_overrides',{})
    external_off=all(overrides.get(key) is False for key in EXTERNAL_COLLISIONS)
    safety=overrides.get('room_boundary_safety',{})
    external_off=external_off and safety.get('enabled') is False
    numerical={k:v for k,v in overrides.items() if k not in EXTERNAL_COLLISIONS and k not in ('gravity','gravity_id','room_boundary_safety','static_visual_only_instance_ids') and not k.startswith('raytracing')}
    clock={k:v for k,v in value.get('render',{}).items() if k in ('fps','frames','pre_static_frames')}
    numerical['render_method']={k:v for k,v in value.get('render',{}).items() if k not in clock}
    result=dict(dynamic_objects=objects,runtime_forces=value.get('runtime_forces',[]),
        gravity=overrides.get('gravity'),external_collision_flags={k:overrides.get(k) for k in EXTERNAL_COLLISIONS},
        room_boundary_safety=safety,render_clock=clock,mechanisms=value.get('mechanisms',[]))
    if not external_off:
        result['manifest']=value.get('manifest')
        result['static_visual_only_instance_ids']=overrides.get('static_visual_only_instance_ids')
        for k in ('background_anchor','_background_anchor'):
            if k in value:result[k]=value[k]
    return result,numerical,external_off


def compare(current,probe,axis):
    left,left_numeric,left_off=projection(current,axis)
    right,right_numeric,right_off=projection(probe,axis)
    return dict(physics_equal=left==right,current_physics=left,probe_physics=right,
        background_excluded=left_off and right_off,numerical_settings_differ=left_numeric!=right_numeric,
        current_numerical_settings=left_numeric,probe_numerical_settings=right_numeric,
        seed_equal=current.get('random_seed',current.get('seed'))==probe.get('random_seed',probe.get('seed')),
        same_execution_claimed=False)


def current_reference_signature(config,axis):
    """Retain full normalized controls/solver metadata, mask verified visuals.

    Unlike the lean archived-scan projection, this keeps experiment fields and
    all other unknown configuration fields. It only removes documented visual
    fields; external scenery is omitted only when explicitly non-colliding.
    """
    theta,value=signature(unwrap(config),axis)
    for wrapper in value.get('dynamic_objects',[]):
        body=wrapper.get('spec',wrapper)
        for key in VISUAL_FIELDS:body.pop(key,None)
        if 'collision_mesh_path' in body:body.pop('mesh_path',None)
    overrides=value.get('scene_overrides',{})
    external_off=all(overrides.get(k) is False for k in EXTERNAL_COLLISIONS) and overrides.get('room_boundary_safety',{}).get('enabled') is False
    if external_off:
        for key in ('manifest','background_anchor','_background_anchor'):value.pop(key,None)
        overrides.pop('static_visual_only_instance_ids',None)
    return theta,value
