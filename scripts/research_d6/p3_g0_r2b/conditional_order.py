"""Create private identity-only ordering; makes no product-role or stress selection."""
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from scripts.research_d6.p3_g0_r2 import partition_eligibility as p
OUT=ROOT/'outputs/research_d6/p3_g0_r2b/run_20261121_01'

def main():
    dest=OUT/'conditional_identity_order.csv'
    if dest.exists():raise FileExistsError('Output exists')
    receipt=json.loads((OUT/'independent_structural_check.json').read_text(encoding='utf8'))
    entries=defaultdict(set)
    for shard in receipt['shards']:
        path=OUT/(shard['product']+'.identity-only.csv')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=shard['sha256']:raise ValueError('Shard identity mismatch')
        with path.open(encoding='utf8',newline='') as f:
            for row in csv.DictReader(f):entries[row['product']].add(row['opaque_group_id'])
    if sum(map(len,entries.values()))!=receipt['total_groups']:raise ValueError('Group count mismatch')
    products=sorted(entries,key=lambda x:p.order_key('PRODUCT',x))
    with dest.open('x',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['product','product_rank','opaque_group_id','group_rank','state'])
        w.writeheader()
        for i,product in enumerate(products,1):
            groups=sorted(entries[product],key=lambda x:p.order_key('REP_GROUP',product+p.SEP+x))
            for j,group in enumerate(groups,1):
                w.writerow({'product':product,'product_rank':i,'opaque_group_id':group,'group_rank':j,'state':'CONDITIONAL_METADATA_POOL'})
    print(json.dumps({'products':len(products),'groups':sum(map(len,entries.values())),'state':'CONDITIONAL_METADATA_POOL','formal_role_assignments':0}))

if __name__=='__main__':main()
