"""Synthetic-only checks for the R2c conditional shortlist."""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.research_d6.p3_g0_r2c import conditional_shortlist as s
from scripts.research_d6.p3_g0_r2 import partition_eligibility as rank
from tests.research_d6.test_p3_g0_r1_manifest import fixture

def synthetic():
    rows=fixture(products=12,groups=18,views=5)
    products={r['product'] for r in rows}
    for r in rows:
        r['source_sha256']=''
        for key in ('exposure_status','allowed_research_role','permission_status','exclusion_status','exclusion_check_scope','near_duplicate_status','group_conflict_status'):r[key]='UNKNOWN'
        r['object_identity_confidence']='INFERRED'
    groupids={}
    mapping=[]
    for row in rows:
        key=(row['product'],row['opaque_group_id']); groupids.setdefault(key,'fictional-group-'+row['opaque_group_id'])
        mapping.append({'product':row['product'],'opaque_group_id':row['opaque_group_id'],'opaque_image_id':row['opaque_image_id'],
                        'source_group_key':groupids[key],'source_image_identity':'fictional/'+row['opaque_image_id']+'.jpg','view':row['view']})
    order=[]
    for i,prod in enumerate(sorted(products,key=lambda x:rank.order_key('PRODUCT',x)),1):
        gs=sorted((gid for p,gid in groupids if p==prod),key=lambda x:rank.order_key('REP_GROUP',prod+rank.SEP+x))
        for j,gid in enumerate(gs,1):order.append({'product':prod,'product_rank':str(i),'opaque_group_id':gid,'group_rank':str(j),'state':'CONDITIONAL_METADATA_POOL'})
    return rows,order,mapping,products

def frame():
    rows,order,mapping,products=synthetic()
    return s.validate_frame(rows,order,mapping,products,216,1080)

class SelectionTests(unittest.TestCase):
    def test_existing_salt_and_deterministic_sort(self):
        self.assertEqual(rank.SALT,'LP-D6-P3-G0-v0.2-20261120')
        self.assertEqual(s.choose(frame()),s.choose(frame()))

    def test_input_order_does_not_change_choice(self):
        rows,order,mapping,products=synthetic()
        f=s.validate_frame(list(reversed(rows)),order,list(reversed(mapping)),products,216,1080)
        self.assertEqual(s.choose(frame()),s.choose(f))

    def test_two_plus_four_tentative_roles_exclusive(self):
        products,groups,_=s.choose(frame())
        self.assertEqual(sum(r['conditional_role']=='TENTATIVE_D_CAL' for r in products),2)
        self.assertEqual(sum(r['conditional_role']=='TENTATIVE_D_EVAL' for r in products),4)
        self.assertEqual(len({(r['product'],r['conditional_role']) for r in groups}),6)

    def test_108_groups_216_views(self):
        _,groups,views=s.choose(frame());self.assertEqual((len(groups),len(views)),(108,216))
        self.assertEqual(sum(r['candidate_kind']=='REPRESENTATIVE_CANDIDATE' for r in groups),36)
        self.assertEqual(sum(r['candidate_kind']=='STRESS_POOL_CANDIDATE' for r in groups),72)

    def test_two_distinct_views_per_group(self):
        _,_,views=s.choose(frame());by={}
        for row in views:by.setdefault(row['opaque_group_id'],set()).add(row['view'])
        self.assertTrue(all(len(views)==2 for views in by.values()))

    def test_group_never_crosses_role(self):
        _,groups,views=s.choose(frame());roles={x['opaque_group_id']:x['conditional_role'] for x in groups}
        self.assertTrue(all(roles[x['opaque_group_id']]==x['conditional_role'] for x in views))

    def test_full_backup_order_exists_before_selection(self):
        f=frame();self.assertEqual(len(f['expected_order']),216)
        products,_,_=s.choose(f);self.assertEqual(sum(x['conditional_role']=='BACKUP_PRODUCT' for x in products),6)

    def test_no_candidate_promoted_from_unknown(self):
        products,groups,views=s.choose(frame());self.assertTrue(all(x['eligibility']=='UNKNOWN' for x in products+groups+views))

    def test_forbidden_formal_or_stress_status(self):
        products,groups,views=s.choose(frame())
        self.assertTrue(all(x['conditional_role'] not in {'D_CAL','D_EVAL'} for x in products+groups+views))
        self.assertTrue(all(x['candidate_kind']!='STRESS_SELECTED' for x in groups+views))

    def test_public_summary_has_no_private_paths_ids_or_labels(self):
        data=dict(status='READY',product_count=6,backup_products=6,group_count=108,representative_candidate_groups=36,
                  stress_pool_candidate_groups=72,view_count=216,source_sha_unknown_views=216,formal_partition=False,stress_selected=False)
        data.update(opaque_image_id='IMG_SECRET',source_image_path='fictional/private.jpg',defect_type='hidden')
        out=json.dumps(s.public_summary(data))
        for token in ('IMG_SECRET','fictional/private.jpg','defect_type','hidden'):self.assertNotIn(token,out)

    def test_missing_input_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(s.IntegrityError):s.verify_inputs(Path(tmp),{'absent.json':'0'*64})

    def test_hash_mismatch_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'fake.json';p.write_text('synthetic',encoding='utf8')
            with self.assertRaises(s.IntegrityError):s.verify_inputs(Path(tmp),{'fake.json':'0'*64})

    def test_no_image_mask_or_protected_file_access(self):
        f=frame()
        with patch('builtins.open',side_effect=AssertionError('file access forbidden')),patch.object(Path,'read_bytes',side_effect=AssertionError('file access forbidden')):
            s.choose(f)

    def test_labels_defect_or_scores_cannot_enter_frame(self):
        for key in ('attribute_state','defect_type','model_score'):
            rows,order,mapping,products=synthetic();rows[0][key]='synthetic'
            with self.assertRaises(s.IntegrityError):s.validate_frame(rows,order,mapping,products,216,1080)

    def test_existing_output_rejected_before_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileExistsError):s.create_output(Path(tmp),[],[],[],{},None,{})

    def test_duplicate_view_blocked(self):
        rows,order,mapping,products=synthetic();rows[1]['view']=rows[0]['view'];mapping[1]['view']=mapping[0]['view']
        with self.assertRaises(s.IntegrityError):s.validate_frame(rows,order,mapping,products,216,1080)

    def test_order_tampering_blocked(self):
        rows,order,mapping,products=synthetic();order[0],order[1]=order[1],order[0]
        with self.assertRaises(s.IntegrityError):s.validate_frame(rows,order,mapping,products,216,1080)

    def test_path_traversal_blocked(self):
        rows,order,mapping,products=synthetic();mapping[0]['source_image_identity']='../restricted.jpg'
        with self.assertRaises(s.IntegrityError):s.validate_frame(rows,order,mapping,products,216,1080)

    def test_unknown_promotion_blocked(self):
        rows,order,mapping,products=synthetic();rows[0]['exclusion_status']='CLEAR'
        with self.assertRaises(s.IntegrityError):s.validate_frame(rows,order,mapping,products,216,1080)

if __name__=='__main__':unittest.main()
