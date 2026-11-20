"""Synthetic-only R2b extraction and privacy controls."""
import copy
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.research_d6.p3_g0_r2b import extract_identity as e
from scripts.research_d6.p3_g0 import validate_identity_manifest as v
from scripts.research_d6.p3_g0_r2 import partition_eligibility as partition

KEY=b'z'*32
SHA='1'*64

def document(product='regulator'):
    def rec(n,view):return {'category':product,'image_path':f'{product}/test/hidden/{product}_{n:04d}_NG_CH_C{view}_fiction.png','anomaly_class':'fictional','mask_path':'fictional'}
    return {'meta':{},'train':[rec(3,1)],'test':[rec(1,1),rec(1,2),rec(2,1),rec(2,2)]}

def extract(doc=None,product='regulator',key=KEY):
    return e.extract_document(document(product) if doc is None else doc,product,key,SHA)

class R2BTests(unittest.TestCase):
    def test_workspace_root_is_project_root(self):
        self.assertTrue((e.ROOT/'scripts/research_d6/p3_g0/validate_identity_manifest.py').is_file())

    def test_twelve_whitelisted_products_parse(self):
        for p in e.PRODUCTS:
            with self.subTest(product=p):self.assertEqual(extract(product=p)[2]['test_views'],4)

    def test_only_test_identity_exported(self):
        rows,_,stats=extract();self.assertEqual((len(rows),stats['test_groups']),(4,2));self.assertTrue(all(r['source_split']=='test' for r in rows))

    def test_train_mutation_has_no_effect(self):
        d=document();before=extract(d)[0];d['train']=[{'category':'arbitrary'}];self.assertEqual(before,extract(d)[0])

    def test_nonidentity_fields_do_not_affect_manifest_or_order(self):
        d=document();before=extract(d)[0];modified=copy.deepcopy(d)
        for r in modified['test']:r['anomaly_class']='different';r['mask_path']='different';r['other_hidden']='changed'
        after=extract(modified)[0];self.assertEqual(before,after)
        self.assertEqual(partition.assess(before,authorized=True,manifest_verified=True),partition.assess(after,authorized=True,manifest_verified=True))

    def test_manifest_does_not_reveal_source_fields(self):
        rows,_,_=extract();text=json.dumps(rows)
        for x in ('anomaly_class','mask_path','image_path','NG_CH','fiction.png','defect_type'):
            self.assertNotIn(x,text)
        self.assertEqual(set(rows[0]),set(v.FIELDS))

    def test_hmac_stable(self):self.assertEqual(extract()[0],extract()[0])
    def test_hmac_domains_distinct(self):self.assertNotEqual(e.pseudonym(KEY,'IMG','x')[4:],e.pseudonym(KEY,'GRP','x')[4:])
    def test_wrong_key_detected(self):
        with self.assertRaises(e.SourceIntegrityError):e.verify_key(b'y'*32,e.sha(KEY))
    def test_matching_prior_key_is_accepted(self):e.verify_key(KEY,e.sha(KEY))
    def test_missing_key_blocked(self):
        with self.assertRaises(e.SourceIntegrityError):e.pseudonym(b'','IMG','x')
    def test_reordering_is_stable(self):
        d=document();before=extract(d)[0];d['test'].reverse();self.assertEqual(before,extract(d)[0])
    def test_cross_product_ids_domain_separated(self):
        self.assertNotEqual(e.pseudonym(KEY,'GRP','regulator\x1fshared'),e.pseudonym(KEY,'GRP','switch\x1fshared'))
    def test_duplicate_group_view_blocked(self):
        d=document();r=copy.deepcopy(d['test'][0]);r['image_path']=r['image_path'].replace('fiction.png','other.png');d['test'].append(r)
        with self.assertRaises(e.SourceIntegrityError):extract(d)
    def test_duplicate_image_blocked(self):
        d=document();d['test'].append(copy.deepcopy(d['test'][0]))
        with self.assertRaises(e.SourceIntegrityError):extract(d)
    def test_missing_image_sha_unknown(self):
        rows,_,_=extract();self.assertTrue(all(not r['source_sha256'] for r in rows));self.assertEqual(v.validate_rows(rows,'D_EVAL')['group_status_counts'],{'UNKNOWN':2})
    def test_protected_status_remains_unknown(self):
        rows,_,_=extract();self.assertTrue(all(r['exclusion_status']=='UNKNOWN' and r['permission_status']=='UNKNOWN' for r in rows))
    def test_malformed_schema_blocked(self):
        with self.assertRaises(e.SourceIntegrityError):extract({'meta':{},'test':[]})
    def test_bad_group_view_blocked(self):
        d=document();d['test'][0]['image_path']='not_a_real_identity'
        with self.assertRaises(e.SourceIntegrityError):extract(d)
    def test_extraction_does_not_open_files(self):
        with patch('builtins.open',side_effect=AssertionError('no source image/mask reads')),patch.object(Path,'read_bytes',side_effect=AssertionError('no file reads')):
            self.assertEqual(extract()[2]['test_views'],4)
    def test_thirteenth_product_rejected(self):
        with self.assertRaises(e.SourceIntegrityError):e.extract_document({'meta':{},'train':[],'test':[]},'unlisted',KEY,SHA)
    def test_protected_path_not_dereferenced(self):
        d=document();d['test'][0]['mask_path']='sealed/HOLD/fixture'
        with patch('builtins.open',side_effect=AssertionError('protected read')):
            self.assertEqual(extract(d)[2]['test_views'],4)
    def test_oversized_r1_shard_not_read_or_truncated(self):
        with patch.object(Path,'is_file',return_value=True),patch.object(Path,'stat') as st,patch.object(Path,'read_bytes',side_effect=AssertionError('file must not be read')):
            st.return_value.st_size=v.MAX_BYTES+1
            self.assertEqual(v.validate_file(Path('test.identity-only.csv'),'D_EVAL',True,'fictional','0'*64)['status'],'BLOCKED')
    def test_unknown_pool_cannot_form_formal_partition(self):
        rows=[]
        for p in e.PRODUCTS[:6]:rows.extend(extract(product=p)[0])
        result,plan=partition.assess(rows,authorized=True,manifest_verified=True,relation_status='UNKNOWN')
        self.assertNotEqual(result['status'],'PASS');self.assertIsNone(plan)

if __name__=='__main__':unittest.main()
