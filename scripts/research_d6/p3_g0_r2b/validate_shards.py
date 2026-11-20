"""Validate R2b private identity-only shards; never opens source JSON or images."""
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from scripts.research_d6.p3_g0 import validate_identity_manifest as v
from scripts.research_d6.p3_g0_r2 import partition_eligibility as p

OUT=ROOT/'outputs/research_d6/p3_g0_r2b/run_20261121_01'
def file_sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    destination=OUT/'independent_structural_check.json'
    if destination.exists(): raise FileExistsError('Output already exists')
    receipt=json.loads((OUT/'extraction_receipt.json').read_text(encoding='utf8'))
    rows=[]; shard_results=[]
    for entry in receipt['source_outcomes']:
        if entry['status']!='EXTRACTED':continue
        path=OUT/(entry['product']+'.identity-only.csv')
        if file_sha(path)!=entry['manifest_sha256']: raise ValueError('shard hash mismatch')
        checks={}
        for role in ('D_CAL','D_EVAL'):
            result=v.validate_file(path,role,True,'RESEARCH_LEAD_CHAT_R2B',entry['manifest_sha256'])
            if result['status']=='BLOCKED' or result.get('row_count')!=entry['test_views']:
                raise ValueError('shard validation mismatch')
            checks[role]={'r1_overall':result['status'],'group_status_counts':result.get('group_status_counts',{})}
        with path.open(encoding='utf8',newline='') as f: part=list(csv.DictReader(f))
        if any(x['product']!=entry['product'] for x in part):raise ValueError('shard product mismatch')
        rows.extend(part)
        shard_results.append({'product':entry['product'],'sha256':entry['manifest_sha256'],'bytes':path.stat().st_size,'rows':len(part),'checks':checks})
    if len(rows)!=receipt['test_views']:raise ValueError('view total mismatch')
    imgs=Counter(x['opaque_image_id'] for x in rows)
    groups=defaultdict(set);group_products=defaultdict(set)
    for row in rows:
        groups[row['opaque_group_id']].add(row['view']);group_products[row['opaque_group_id']].add(row['product'])
    if any(n!=1 for n in imgs.values()) or any(len(x)!=1 for x in group_products.values()):
        raise ValueError('cross-shard identity collision')
    if len(groups)!=receipt['test_groups'] or any(len(x)!=5 for x in groups.values()):
        raise ValueError('group views mismatch')
    result,plan=p.assess(rows,authorized=True,manifest_verified=True,relation_status='UNKNOWN')
    if plan is not None or result['status']=='PASS':raise ValueError('unknown permission promoted')
    payload={'schema':'lp.d6.p3.r2b.structural-check.v1','shards':shard_results,
             'total_views':len(rows),'total_groups':len(groups),'cross_shard_duplicate_image_ids':0,
             'cross_shard_group_product_conflicts':0,'unique_nonempty_source_sha':0,
             'missing_source_sha':sum(not x['source_sha256'] for x in rows),
             'r2_aggregate':result,'formal_partition_created':False,
             'manifest_sha_provenance':'INTERNAL_CONSISTENCY_NOT_INDEPENDENT_CUSTODIAN_ATTESTATION',
             'validator_code_sha256':file_sha(Path(v.__file__)),'r2_code_sha256':file_sha(Path(p.__file__)),
             'validation_code_sha256':file_sha(Path(__file__))}
    with destination.open('x',encoding='utf8') as f:json.dump(payload,f,indent=2)
    print(json.dumps({k:payload[k] for k in ('total_views','total_groups','missing_source_sha','r2_aggregate','formal_partition_created')},indent=2))

if __name__=='__main__':
    try:main()
    except (ValueError,OSError,KeyError):
        print(json.dumps({'status':'R2B_BLOCKED_INTEGRITY','reason':'SHARD_VALIDATION_FAILED'}));raise SystemExit(2)
