"""R2 tests use only fictional identity records and in-memory fixtures."""
import json
import unittest
from unittest.mock import patch
from pathlib import Path

from scripts.research_d6.p3_g0_r2 import partition_eligibility as p
from scripts.research_d6.p3_g0 import validate_identity_manifest as r1
from tests.research_d6.test_p3_g0_r1_manifest import fixture


class R2Tests(unittest.TestCase):
    def assess(self, rows):
        return p.assess(rows, authorized=True, manifest_verified=True, relation_status='CLEAR_FRAME')

    def test_valid_six_product_partition(self):
        result, plan = self.assess(fixture(groups=18))
        self.assertEqual(result['status'], 'PASS')
        self.assertEqual((result['cal_products'], result['eval_products']), (2, 4))
        self.assertEqual(result['representative_groups'], 36)
        self.assertEqual(result['stress_selected'], 0)
        self.assertTrue(all(len(set(g['views'])) == 2 for g in plan['groups']))

    def test_product_cannot_have_both_roles(self):
        result=p.role_sets(['one','two'],['one','three','four','five'])
        self.assertEqual(result,{'status':'BLOCKED','reason':'PRODUCT_CROSSES_ROLES'})

    def test_same_group_crosses_roles(self):
        rows=fixture(); rows[20]['opaque_group_id']=rows[0]['opaque_group_id']
        assigned={f'synthetic_{i}': 'D_CAL' if i<2 else 'D_EVAL' for i in range(6)}
        rows[40]['opaque_group_id']=rows[0]['opaque_group_id']
        self.assertEqual(p.validate_partition(rows, assigned, 'CLEAR_FRAME')['status'], 'BLOCKED')

    def test_fewer_than_ten_groups(self):
        result,plan=self.assess(fixture(groups=9));self.assertNotEqual(result['status'],'PASS');self.assertIsNone(plan)

    def test_single_view_cannot_pass(self):
        result,plan=self.assess(fixture(views=1));self.assertNotEqual(result['status'],'PASS');self.assertIsNone(plan)

    def test_missing_sha_is_unknown(self):
        rows=fixture();rows[0]['source_sha256']=''
        result,plan=self.assess(rows);self.assertNotEqual(result['status'],'PASS');self.assertIsNone(plan)
        self.assertGreater(result['unknown_groups_cal'],0)

    def test_partial_protected_scope_is_unknown(self):
        rows=fixture();rows[0]['exclusion_check_scope']=rows[1]['exclusion_check_scope']='PARTIAL'
        result,plan=self.assess(rows);self.assertNotEqual(result['status'],'PASS');self.assertIsNone(plan)

    def test_known_near_duplicate_not_eligible(self):
        rows=fixture();rows[0]['near_duplicate_status']='CONFLICT'
        result,plan=self.assess(rows);self.assertNotEqual(result['status'],'PASS');self.assertIsNone(plan)

    def test_mixed_train_excluded(self):
        rows=fixture();rows[0]['source_split']=rows[1]['source_split']='train'
        result,plan=self.assess(rows);self.assertNotEqual(result['status'],'PASS');self.assertEqual(result['non_test_groups'],1)

    def test_only_five_products(self):
        result,plan=self.assess(fixture(products=5));self.assertNotEqual(result['status'],'PASS');self.assertIsNone(plan)

    def test_oversize_file_blocks_before_read(self):
        fake=Path('synthetic.identity-only.csv')
        with patch.object(Path,'is_file',return_value=True), patch.object(Path,'stat') as st, patch.object(Path,'read_bytes',side_effect=AssertionError('read forbidden')):
            st.return_value.st_size=r1.MAX_BYTES+1
            self.assertEqual(r1.validate_file(fake,'D_EVAL',True,'synthetic','0'*64)['status'],'BLOCKED')

    def test_determinism_under_permutation(self):
        rows=fixture(groups=18)
        self.assertEqual(self.assess(rows),self.assess(list(reversed(rows))))

    def test_secret_pseudonyms_are_stable_domain_separated(self):
        key=b'k'*32;self.assertEqual(p.opaque_id(key,'GRP','fictional'),p.opaque_id(key,'GRP','fictional'))
        self.assertNotEqual(p.opaque_id(key,'GRP','fictional'),p.opaque_id(key,'IMG','fictional'))
        with self.assertRaises(ValueError):p.opaque_id(b'weak','GRP','fictional')

    def test_public_summary_has_no_sample_ids_or_labels(self):
        result,_=self.assess(fixture(groups=18));encoded=json.dumps(result)
        for forbidden in ('GRP_','IMG_','sample_key','defect_type','attribute_state'):
            self.assertNotIn(forbidden,encoded)

    def test_missing_authorization_or_input_stops(self):
        self.assertEqual(p.assess(None,authorized=False,manifest_verified=False)[0]['status'],'BLOCKED')
        self.assertEqual(p.assess(fixture(),authorized=True,manifest_verified=False)[0]['status'],'BLOCKED')

    def test_protected_relation_unknown_prevents_partition(self):
        result,plan=p.assess(fixture(),authorized=True,manifest_verified=True,relation_status='UNKNOWN')
        self.assertNotEqual(result['status'],'PASS');self.assertIsNone(plan)

    def test_no_data_file_open_in_assess(self):
        rows=fixture(groups=18)
        with patch('builtins.open',side_effect=AssertionError('I/O prohibited')), patch.object(Path,'read_bytes',side_effect=AssertionError('I/O prohibited')):
            self.assertEqual(self.assess(rows)[0]['status'],'PASS')


if __name__=='__main__':unittest.main()
