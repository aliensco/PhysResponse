"""Regression: valid negative outcomes survive missing inversion and generation failures."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from physedit.boundary import digest, distance, distance_effect, distance_summary, plan, run, score, write_rows


class BoundaryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        image=self.root/'image.png';image.write_bytes(b'test condition')
        inputs=[];keys=[]
        for i,y in enumerate((0,1)):
            sid=f'p{i}'
            inputs.append(dict(sample_id=sid,condition_image=str(image),condition_sha256=digest(image),
                prompt='Move at the requested velocity',seed=i,num_frames=3,fps=2,
                evaluation_duration_s=1,source_width=16,source_height=16))
            keys.append(dict(sample_id=sid,scene_id='T15',source_group_id='s',family_id='f',point_id=sid,
                theta=i,target_event='fall',events={'fall':y},gravity=9.81,pair_ids=['pair'],
                inversion={'status':'not_applicable' if i==0 else 'eligible','reason':'registered',
                           'reference_registration_sha256':'reference'}))
        write_rows(self.root/'inputs.jsonl',inputs);write_rows(self.root/'answers.jsonl',keys)
        self.spec=dict(model='test',inputs=str(self.root/'inputs.jsonl'),answers=str(self.root/'answers.jsonl'),
                       generation_command=[sys.executable,'-c','pass'])

    def tearDown(self):self.tmp.cleanup()

    def prepared(self):
        def fake_media(path,row):
            path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'cpu test media')
            return dict(status='ok',video=str(path),video_sha256=digest(path))
        with patch('physedit.boundary.media',side_effect=fake_media):run(self.spec,self.root/'run')
        return self.root/'run'

    def annotations(self,root,labels=(0,1)):
        observed=[json.loads(s) for s in (root/'observed_rows.jsonl').read_text().splitlines()]
        path=self.root/'events.jsonl'
        write_rows(path,[dict(sample_id=o['sample_id'],video_sha256=o['video_sha256'],
                 method='unit_test_fixture',events={'fall':y},evidence='synthetic regression fixture')
                 for o,y in zip(observed,labels)])
        return path

    def test_no_inversion_gate(self):
        self.assertEqual(plan(self.spec)['generation_candidates'],2)
        root=self.prepared()
        export=[json.loads(s) for s in (root/'generation_rows.jsonl').read_text().splitlines()]
        self.assertEqual(len(export),2)
        for r in export:
            self.assertFalse(set(r)&{'events','inversion','boundary','theta','reference_target'})
        result=score(root,self.annotations(root))
        self.assertEqual(result['registered'],2)
        self.assertEqual(result['balanced_accuracy'],1)
        self.assertEqual(result['independent_pair_accuracy'],1)
        self.assertIsNone(result['near_accuracy'])
        self.assertEqual(result['inversion']['statuses'],{'not_applicable':1,'missing':1})

    def test_pending_reference_cannot_start_formal_run(self):
        path=self.root/'answers.jsonl'
        rows=[json.loads(s) for s in path.read_text().splitlines()]
        rows[0]['reference_admission']={'status':'pending','reasons':['reference_rgb_observability_pending']}
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        with self.assertRaisesRegex(ValueError,'reference admission pending'):
            plan(self.spec)

    def test_missing_annotation_remains_failure(self):
        root=self.prepared();r=score(root)
        self.assertEqual(r['registered'],2);self.assertEqual(r['balanced_accuracy'],0)
        self.assertEqual(r['annotations_bound'],0)

    def test_constant_answer_does_not_pass_pair(self):
        root=self.prepared();r=score(root,self.annotations(root,(1,1)))
        self.assertEqual(r['balanced_accuracy'],.5);self.assertEqual(r['independent_pair_accuracy'],0)

    def test_at_bound_not_success(self):
        root=self.prepared();path=self.root/'gravity.jsonl'
        write_rows(path,[dict(sample_id='p1',status='at_bound',gravity=9.81)])
        result=score(root,gravity=path)
        self.assertEqual(result['inversion']['fitted'],0)
        self.assertEqual(result['inversion']['statuses']['at_bound'],1)

    def test_all_generation_failures_retained(self):
        root=self.root/'run';run(self.spec,root);r=score(root)
        self.assertEqual(r['registered'],2);self.assertEqual(r['scored_rows'],2)
        self.assertEqual(r['balanced_accuracy'],0)

    def test_unknown_prediction_rejected(self):
        root=self.prepared();p=self.root/'bad.jsonl'
        write_rows(p,[dict(sample_id='alien')])
        with self.assertRaises(ValueError):score(root,p)

    def test_changed_key_rejected(self):
        root=self.prepared()
        with (root/'answers.jsonl').open('a') as f:f.write('\n')
        with self.assertRaises(ValueError):score(root)

    def test_changed_video_invalidates_annotation(self):
        root=self.prepared();ann=self.annotations(root)
        (root/'generated'/'p1'/'resized.mp4').write_bytes(b'changed')
        self.assertEqual(score(root,ann)['balanced_accuracy'],.5)

    def test_chain_registration_requires_matching_final_contract_and_strict_order(self):
        path = self.root/'answers.jsonl'
        keys = [json.loads(line) for line in path.read_text().splitlines()]
        keys[1].update(events={'contact': 1, 'fall': 1},
                       required_order=[['contact', 'fall']], strict_event_order=True,
                       event_contract_sha256='chain-contract')
        path.unlink()
        write_rows(path, keys)
        root = self.prepared()
        annotation = self.annotations(root)
        rows = [json.loads(line) for line in annotation.read_text().splitlines()]
        candidate = rows[1]
        candidate.update(events={'contact': 1, 'fall': 1}, event_times_s={'contact': .5, 'fall': .5},
                         event_contract_sha256='chain-contract', review_status='final')
        for digest, status, second_time, expected_bound, expected_score in [
            ('old-contract', 'final', .75, 'missing_or_unbound', 0),
            ('chain-contract', 'prelabel', .75, 'missing_or_unbound', 0),
            ('chain-contract', 'final', .5, 'bound', 0),
            ('chain-contract', 'final', .75, 'bound', 1),
        ]:
            candidate.update(event_contract_sha256=digest, review_status=status)
            candidate['event_times_s']['fall'] = second_time
            annotation.unlink()
            write_rows(annotation, rows)
            row = next(r for r in score(root, annotation)['rows'] if r['sample_id'] == 'p1')
            self.assertEqual(row['annotation_status'], expected_bound)
            self.assertEqual(row['event_exact'], expected_score)

    def test_near_accuracy_is_unpaired_across_families(self):
        points = [
            dict(bin='near', label=0, accuracy=1, scene_id='T01', source_group_id='a'),
            dict(bin='near', label=0, accuracy=0, scene_id='T02', source_group_id='b'),
            dict(bin='near', label=1, accuracy=1, scene_id='T03', source_group_id='c'),
        ]
        near = next(row for row in distance_summary(points) if row['bin'] == 'near')
        self.assertEqual(near['reference_points'], 3)
        self.assertEqual(near['reference_counts'], {'negative': 2, 'positive': 1})
        self.assertEqual(near['side_accuracy'], {'negative': .5, 'positive': 1})
        self.assertEqual(near['balanced_accuracy'], .75)
        self.assertEqual(len(near['scenes']), 3)

    def test_score_nearacc_uses_cross_family_points_without_pair(self):
        image=self.root/'image2.png';image.write_bytes(b'test condition 2')
        inputs=[];keys=[]
        for i,(label,scene,family,theta) in enumerate(((0,'T01','f0',.01),(1,'T02','f1',.03))):
            sid=f'n{i}'
            inputs.append(dict(sample_id=sid,condition_image=str(image),condition_sha256=digest(image),
                prompt='Move at the requested velocity',seed=i,num_frames=3,fps=2,
                evaluation_duration_s=1,source_width=16,source_height=16))
            keys.append(dict(sample_id=sid,scene_id=scene,source_group_id=scene,family_id=family,point_id=sid,
                theta=theta,target_event='fall',events={'fall':label},gravity=9.81,
                inversion={'status':'not_applicable','reason':'registered'},
                boundary={'status':'verified','source':'test','normalization_span':1,'bracket':[0,.02]}))
        write_rows(self.root/'inputs2.jsonl',inputs);write_rows(self.root/'answers2.jsonl',keys)
        spec=dict(self.spec,inputs=str(self.root/'inputs2.jsonl'),answers=str(self.root/'answers2.jsonl'))
        def fake_media(path,row):
            path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'cpu test media 2')
            return dict(status='ok',video=str(path),video_sha256=digest(path))
        with patch('physedit.boundary.media',side_effect=fake_media):run(spec,self.root/'run2')
        result=score(self.root/'run2',self.annotations(self.root/'run2',labels=(0,1)))
        self.assertEqual(result['near_accuracy'],1)
        self.assertIsNone(result['near_accuracy_family_macro'])
        self.assertEqual(result['near_reference_counts'],{'negative':1,'positive':1})
        self.assertEqual(result['nearacc_definition'],'point_balanced_unpaired_v2')

    def test_distance_effect_reports_near_far_drop(self):
        near = dict(bin='near', reference_points=4, balanced_accuracy=.75, scenes=['T17'])
        far = dict(bin='far', reference_points=8, balanced_accuracy=.25, scenes=['T01', 'T02'])
        effect = distance_effect([near, far])
        self.assertEqual(effect['near_accuracy'], .75)
        self.assertEqual(effect['far_accuracy'], .25)
        self.assertEqual(effect['drop'], -.5)
        self.assertEqual(effect['near_points'], 4)
        self.assertEqual(effect['far_points'], 8)

    def test_near_accuracy_reports_na_without_both_classes(self):
        points = [dict(bin='near', label=1, accuracy=1, scene_id='T01', source_group_id='a')]
        near = next(row for row in distance_summary(points) if row['bin'] == 'near')
        self.assertIsNone(near['balanced_accuracy'])
        self.assertEqual(near['side_accuracy']['positive'], 1)
        self.assertIsNone(near['side_accuracy']['negative'])

    def test_distance_interval_ambiguity(self):
        k=dict(theta=.06,boundary=dict(status='verified',source='search',normalization_span=1,bracket=[0,.02]))
        self.assertIsNone(distance(k)['bin'])
        k['theta']=.03
        self.assertEqual(distance(k)['bin'],'near')

    def test_zero_lower_bound_is_not_known_exact_threshold(self):
        k=dict(theta=0,boundary=dict(status='verified',source='search',normalization_span=1,bracket=[0,.01]))
        self.assertEqual(distance(k)['bin'],'near')
        k['theta']=.005
        self.assertEqual(distance(k)['bin'],'near')
        k['boundary']['bracket']=[0,.06]
        self.assertIsNone(distance(k)['bin'])
        k['theta']=0;k['boundary']['bracket']=[0,0]
        self.assertEqual(distance(k)['bin'],'at_boundary')

    def test_existing_output_not_overwritten(self):
        root=self.prepared()
        with self.assertRaises(FileExistsError):run(self.spec,root)

    def test_short_native_clock_rejected(self):
        p=self.root/'inputs.jsonl'
        rows=[json.loads(s) for s in p.read_text().splitlines()]
        rows[0]['evaluation_duration_s']=2
        p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        with self.assertRaises(ValueError):plan(self.spec)

    def test_successful_gravity_reports_both_denominators(self):
        root=self.prepared();observed=[json.loads(s) for s in (root/'observed_rows.jsonl').read_text().splitlines()]
        path=self.root/'gravity.jsonl'
        write_rows(path,[dict(sample_id='p1',status='ok',gravity=9.81,
                             video_sha256=observed[1]['video_sha256'],reference_registration_sha256='reference')])
        g=score(root,gravity=path)['inversion']
        self.assertEqual(g['registered_coverage'],.5)
        self.assertEqual(g['eligible_fit_coverage'],1)
        self.assertEqual(g['relative_deviation_mean'],0)

    def test_input_hash_failure_preserves_other_point(self):
        path=self.root/'inputs.jsonl';rows=[json.loads(s) for s in path.read_text().splitlines()]
        rows[0]['condition_sha256']='wrong'
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        root=self.prepared()
        self.assertEqual(len((root/'generation_rows.jsonl').read_text().splitlines()),1)
        self.assertEqual(score(root)['registered'],2)

    def test_duplicate_registration_rejected(self):
        path=self.root/'inputs.jsonl';first=path.read_text().splitlines()[0]
        with path.open('a') as f:f.write(first+'\n')
        with self.assertRaises(ValueError):plan(self.spec)

    def test_shared_seeds_not_mislabelled_independent(self):
        path=self.root/'inputs.jsonl';rows=[json.loads(s) for s in path.read_text().splitlines()]
        rows[1]['seed']=rows[0]['seed']
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        with self.assertRaises(ValueError):plan(self.spec)

    def test_runner_checks_available_gpu_capacity(self):
        from physedit.runners.boundary_ltx import check_gpu
        with patch('subprocess.check_output',return_value='0, 20000, 2\n'):
            check_gpu(0)
        with patch('subprocess.check_output',return_value='0, 13000, 2\n'):
            with self.assertRaises(RuntimeError):check_gpu(0)
        with patch('subprocess.check_output',return_value='0, 20000, 50\n'):
            with self.assertRaises(RuntimeError):check_gpu(0)

    def screen_record(self, root, status='confirmed_failure', **extra):
        observed=[json.loads(s) for s in (root/'observed_rows.jsonl').read_text().splitlines()]
        record=dict(sample_id='p1',video_sha256=observed[1]['video_sha256'],status=status,
                    method='vlm_then_adjudication',reviewer='test reviewer',evidence='identity lost',
                    blocked_events=['fall'],block_inversion=True)
        record.update(extra)
        path=self.root/'screening.jsonl'
        other=dict(sample_id='p0',video_sha256=observed[0]['video_sha256'],status='pass',
                   method='vlm',evidence='target maintained')
        write_rows(path,[other,record]);return path

    def test_screen_failure_keeps_denominator_and_separates_coverage(self):
        root=self.prepared();ann=self.annotations(root)
        s=self.screen_record(root)
        r=score(root,ann,screening=s)
        self.assertEqual(r['registered'],2)
        self.assertEqual(r['balanced_accuracy'],.5)
        self.assertEqual(r['independent_pair_accuracy'],0)
        self.assertEqual(r['event_measurement']['target_coverage'],.5)
        self.assertEqual(r['event_measurement']['conditional_target_accuracy'],1)
        self.assertEqual(r['task_success_rate'],.5)
        self.assertEqual(r['inversion']['eligible_fit_coverage'],0)
        self.assertEqual(r['inversion']['statuses']['screening_confirmed_failure'],1)

    def test_earlier_event_survives_later_inversion_failure(self):
        root=self.prepared()
        s=self.screen_record(root,blocked_events=[])
        r=score(root,self.annotations(root),screening=s)
        self.assertEqual(r['balanced_accuracy'],1)
        self.assertEqual(r['event_measurement']['target_coverage'],1)
        self.assertEqual(r['task_success_rate'],.5)
        self.assertEqual(r['inversion']['fitted'],0)

    def test_suspect_is_not_confirmed_generation_failure(self):
        root=self.prepared();s=self.screen_record(root,status='suspect')
        r=score(root,self.annotations(root),screening=s)
        self.assertEqual(r['screening']['confirmed_failure_count'],0)
        self.assertEqual(r['screening']['unresolved_count'],1)
        self.assertEqual(r['event_measurement']['target_coverage'],.5)

    def test_wrong_screen_hash_rejected(self):
        root=self.prepared();s=self.screen_record(root,video_sha256='wrong')
        with self.assertRaises(ValueError):score(root,screening=s)

    def test_blocked_fit_cannot_leak_into_error(self):
        root=self.prepared();s=self.screen_record(root)
        obs=[json.loads(x) for x in (root/'observed_rows.jsonl').read_text().splitlines()]
        g=self.root/'fit.jsonl'
        write_rows(g,[dict(sample_id='p1',status='ok',gravity=9.81,
                           video_sha256=obs[1]['video_sha256'],reference_registration_sha256='reference')])
        r=score(root,self.annotations(root),g,s)
        self.assertEqual(r['inversion']['fitted'],0)
        self.assertIsNone(r['inversion']['relative_deviation_mean'])

    def test_empty_screen_keeps_all_pending(self):
        root=self.prepared();s=self.root/'empty_screen.jsonl';write_rows(s,[])
        r=score(root,self.annotations(root),screening=s)
        self.assertEqual(r['registered'],2)
        self.assertEqual(r['screening']['unresolved_count'],2)
        self.assertEqual(r['event_measurement']['target_coverage'],0)
        self.assertIsNone(r['event_measurement']['conditional_target_accuracy'])

    def test_coupled_pair_failure_retains_both_endpoints(self):
        path=self.root/'inputs.jsonl';rows=[json.loads(x) for x in path.read_text().splitlines()]
        for r in rows:r['seed']=0
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        path=self.root/'answers.jsonl';rows=[json.loads(x) for x in path.read_text().splitlines()]
        for r in rows:r['pairing_mode']='coupled_seed'
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        root=self.prepared();s=self.screen_record(root)
        r=score(root,self.annotations(root),screening=s)
        self.assertEqual(r['coupled_pair_accuracy'],0)
        self.assertEqual(r['registered'],2)

    def test_vlm_flags_route_to_review_not_deletion(self):
        from physedit.boundary_screening import prepare
        root=self.prepared()
        obs=[json.loads(s) for s in (root/'observed_rows.jsonl').read_text().splitlines()]
        path=self.root/'vlm.jsonl'
        write_rows(path,[dict(sample_id=o['sample_id'],video_sha256=o['video_sha256'],status='ok',
                             judgement=dict(quality_flags=['target_deformation'] if i else [],
                                            assessable=True,reason='visible evidence')) for i,o in enumerate(obs)])
        out=self.root/'screen'
        summary=prepare(root,path,None,out)
        self.assertEqual(summary['routing_rows'],2)
        self.assertEqual(summary['statuses'],{'pass':1,'suspect':1})
        r=score(root,self.annotations(root),screening=out/'screening.jsonl')
        self.assertEqual(r['registered'],2)
        self.assertEqual(r['event_measurement']['target_coverage'],.5)


if __name__=='__main__':unittest.main()
