"""Freeze an UNKNOWN-only shortlist from the registered R2b identity frame.

Only explicitly named R2b metadata and the historical P2 selection manifest
are read. Source image/mask/feature bytes and HMAC keys are never opened.
"""
from __future__ import annotations
import csv
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path, PureWindowsPath

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from scripts.research_d6.p3_g0_r2 import partition_eligibility as ranking
from scripts.research_d6.p3_g0 import validate_identity_manifest as r1

INPUT=ROOT/'outputs/research_d6/p3_g0_r2b/run_20261121_01'
OUTPUT=ROOT/'outputs/research_d6/p3_g0_r2c/run_20261121_01'
P2_MANIFEST=ROOT/'research_pilot/LP_W1_D5R_IMAGE_PILOT_V2_2V/selection_manifest.csv'
HEAD='573e1fa7988205a12c263bc0b3329efdec67b2e2'
PINNED={
 'extraction_receipt.json':'3aa6d1e7a8b10c426e462adb0f5085281acf62d4cc480dd654328f07fc5f49d6',
 'independent_structural_check.json':'08348e5378e2ba3fde51b8b8cf5597e4c36fcaabbad1b4f110ca762a34430ecd',
 'conditional_identity_order.csv':'38abc314eac0b4d79405e947a87f9cbe5b1fbe619efe7c5961cbdf8d345fe50d',
 'custodian_source_mapping.csv':'6b0754dd466a6a29dff43004a4853489eb80871fa84e3ec1ae06324b9d83df78',
}
P2_SHA='c0def1f598d38ea68c78da028958f7ee030a7bdf21528d5a249fe8f8a6f81b7f'
PRODUCTS={'regulator','rolled_strip_base','switch','tape','terminalblock','toy','toy_brick','usb','vcpill','wooden_beads','woodstick','zipper'}
VERSION='LP-D6-P3-G0-R2C-v1'


class IntegrityError(Exception):pass

def sha_file(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def read_csv(path:Path)->list[dict]:
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def verify_inputs(base:Path, pinned:dict=PINNED)->tuple[dict,dict,list[dict],list[dict],dict]:
    for name,expected in pinned.items():
        path=base/name
        if not path.is_file():raise IntegrityError('MISSING_INPUT')
        if sha_file(path)!=expected:raise IntegrityError('INPUT_SHA_MISMATCH')
    receipt=json.loads((base/'extraction_receipt.json').read_text(encoding='utf8'))
    structural=json.loads((base/'independent_structural_check.json').read_text(encoding='utf8'))
    if (receipt.get('status')!='R2B_IDENTITY_EXPORTED_ELIGIBILITY_PENDING' or
        receipt.get('head')!='d0f75a45f10225a53874d659937c139e4c5a9951' or
        receipt.get('mapping_sha256')!=pinned['custodian_source_mapping.csv'] or
        receipt.get('test_views')!=45705 or receipt.get('test_groups')!=9141 or
        receipt.get('unknown_groups')!=9141 or receipt.get('confirmed_groups')!=0 or
        structural.get('formal_partition_created') is not False or
        structural.get('total_groups')!=9141 or structural.get('total_views')!=45705):
        raise IntegrityError('RECEIPT_IDENTITY_MISMATCH')
    source={x['product']:x for x in receipt['source_outcomes']}
    check={x['product']:x for x in structural['shards']}
    if set(source)!=PRODUCTS or set(check)!=PRODUCTS or len(receipt['source_outcomes'])!=12 or len(structural['shards'])!=12:
        raise IntegrityError('PRODUCT_SET_MISMATCH')
    rows=[]; shard_hashes={}
    for product in sorted(PRODUCTS):
        a,b=source[product],check[product];path=base/(product+'.identity-only.csv')
        if not path.is_file():raise IntegrityError('MISSING_SHARD')
        actual=sha_file(path)
        if a.get('status')!='EXTRACTED' or actual!=a.get('manifest_sha256') or actual!=b.get('sha256'):
            raise IntegrityError('SHARD_SHA_MISMATCH')
        part=read_csv(path)
        if len(part)!=a['test_views'] or len(part)!=b['rows'] or any(set(x)!=set(r1.FIELDS) or x['product']!=product for x in part):
            raise IntegrityError('SHARD_SCHEMA_OR_COUNT_MISMATCH')
        rows.extend(part);shard_hashes[product]=actual
    order=read_csv(base/'conditional_identity_order.csv')
    mapping=read_csv(base/'custodian_source_mapping.csv')
    return receipt,structural,rows,order,{'mapping':mapping,'shard_hashes':shard_hashes}

def validate_frame(rows:list[dict],order:list[dict],mapping:list[dict], expected_products:set=PRODUCTS,
                   expected_groups:int=9141,expected_views:int=45705)->dict:
    if len(rows)!=expected_views or len(mapping)!=expected_views or len(order)!=expected_groups:
        raise IntegrityError('FRAME_COUNTS_MISMATCH')
    groups=defaultdict(list); images={};group_products={}
    for row in rows:
        if set(row)!=set(r1.FIELDS):raise IntegrityError('INVALID_ROW_FIELDS')
        prod,gid,iid=row['product'],row['opaque_group_id'],row['opaque_image_id']
        if prod not in expected_products or not r1.ID_GROUP.fullmatch(gid) or not r1.ID_IMAGE.fullmatch(iid):raise IntegrityError('INVALID_OPAQUE_IDENTITY')
        if iid in images:raise IntegrityError('DUPLICATE_IMAGE')
        if gid in group_products and group_products[gid]!=prod:raise IntegrityError('CROSS_PRODUCT_GROUP')
        group_products[gid]=prod;images[iid]=row;groups[(prod,gid)].append(row)
        if (row['source_split']!='test' or row['source_sha256']!='' or
            any(row[k]!='UNKNOWN' for k in ('exposure_status','allowed_research_role','permission_status','exclusion_status','exclusion_check_scope','near_duplicate_status','group_conflict_status')) or
            row['object_identity_confidence']!='INFERRED'):
            raise IntegrityError('UNKNOWN_STATUS_PROMOTED')
    if len(groups)!=expected_groups or set(group_products.values())!=expected_products:
        raise IntegrityError('GROUP_PRODUCT_COUNTS_MISMATCH')
    if any(Counter(r['view'] for r in members)!=Counter({'C1':1,'C2':1,'C3':1,'C4':1,'C5':1}) for members in groups.values()):
        raise IntegrityError('GROUP_VIEWS_MISMATCH')
    mapped={};group_source={};source_group={};source_images=set()
    for row in mapping:
        if set(row)!={'product','opaque_group_id','opaque_image_id','source_group_key','source_image_identity','view'}:
            raise IntegrityError('MAPPING_SCHEMA_MISMATCH')
        iid=row['opaque_image_id']; ref=images.get(iid)
        if ref is None or iid in mapped or any(row[k]!=ref[k] for k in ('product','opaque_group_id','view')):
            raise IntegrityError('MAPPING_IDENTITY_MISMATCH')
        source=PureWindowsPath(row['source_image_identity'])
        if source.is_absolute() or source.drive or not source.parts or any(x in ('.','..') or ':' in x for x in source.parts):
            raise IntegrityError('SOURCE_RELATIVE_PATH_INVALID')
        gid=row['opaque_group_id']; group_key=(row['product'],row['source_group_key']); image_key=(row['product'],row['source_image_identity'])
        if not row['source_group_key'] or (gid in group_source and group_source[gid]!=group_key) or (group_key in source_group and source_group[group_key]!=gid) or image_key in source_images:
            raise IntegrityError('MAPPING_SOURCE_COLLISION')
        group_source[gid]=group_key;source_group[group_key]=gid;source_images.add(image_key)
        mapped[iid]=row['source_image_identity']
    if len(mapped)!=expected_views:raise IntegrityError('MAPPING_COUNT_MISMATCH')
    products=sorted(expected_products,key=lambda x:ranking.order_key('PRODUCT',x))
    expected_order=[]
    for n,prod in enumerate(products,1):
        gids=sorted((gid for p,gid in groups if p==prod),key=lambda g:ranking.order_key('REP_GROUP',prod+ranking.SEP+g))
        for m,gid in enumerate(gids,1):
            expected_order.append({'product':prod,'product_rank':str(n),'opaque_group_id':gid,'group_rank':str(m),'state':'CONDITIONAL_METADATA_POOL'})
    if order!=expected_order:raise IntegrityError('CONDITIONAL_ORDER_MISMATCH')
    return {'groups':groups,'images':images,'mapping':mapped,'products':products,'expected_order':expected_order}

def choose(frame:dict)->tuple[list[dict],list[dict],list[dict]]:
    products=frame['products'];groups=frame['groups']
    if len(products)!=12:raise IntegrityError('PRODUCT_COUNT_MISMATCH')
    prod_rows=[];group_rows=[];view_rows=[]
    for i,prod in enumerate(products):
        role='TENTATIVE_D_CAL' if i<2 else 'TENTATIVE_D_EVAL' if i<6 else 'BACKUP_PRODUCT'
        prod_rows.append({'product':prod,'product_rank':i+1,'conditional_role':role,'eligibility':'UNKNOWN'})
        if i>=6:continue
        gids=sorted((g for p,g in groups if p==prod),key=lambda g:ranking.order_key('REP_GROUP',prod+ranking.SEP+g))
        if len(gids)<18:raise IntegrityError('INSUFFICIENT_GROUPS')
        for index,gid in enumerate(gids[:18]):
            tag='REPRESENTATIVE_CANDIDATE' if index<6 else 'STRESS_POOL_CANDIDATE'
            candidates=sorted(groups[(prod,gid)],key=lambda r:ranking.order_key('VIEW',prod+ranking.SEP+gid+ranking.SEP+r['view']))
            if len(candidates)!=5 or len({r['view'] for r in candidates})!=5:raise IntegrityError('VIEW_COUNT_MISMATCH')
            group_rows.append({'product':prod,'product_rank':i+1,'opaque_group_id':gid,'group_rank':index+1,
                               'conditional_role':role,'candidate_kind':tag,'eligibility':'UNKNOWN',
                               'object_identity_confidence':'INFERRED','unselected_views_inherit_group_role':3})
            for row in candidates[:2]:
                view_rows.append({'opaque_image_id':row['opaque_image_id'],'product':prod,'opaque_group_id':gid,'view':row['view'],
                                  'conditional_role':role,'candidate_kind':tag,'eligibility':'UNKNOWN'})
    if (len(group_rows),len(view_rows))!=(108,216):raise IntegrityError('OUTPUT_COUNTS_MISMATCH')
    return prod_rows,group_rows,view_rows

def infer_image_base(path:Path=P2_MANIFEST)->PureWindowsPath:
    if sha_file(path)!=P2_SHA:raise IntegrityError('HISTORICAL_MANIFEST_SHA_MISMATCH')
    rows=read_csv(path);parents=set()
    if len(rows)!=48:raise IntegrityError('HISTORICAL_MANIFEST_COUNT_MISMATCH')
    permitted=PureWindowsPath(r'G:\dataset\realiad')
    for row in rows:
        absolute=PureWindowsPath(row['image_path']);relative=PureWindowsPath(row['image_relpath'])
        if not absolute.is_absolute() or relative.is_absolute() or relative.drive or '..' in relative.parts:
            raise IntegrityError('HISTORICAL_PATH_SCHEMA_MISMATCH')
        if len(relative.parts)>=len(absolute.parts) or absolute.parts[-len(relative.parts):]!=relative.parts:
            raise IntegrityError('HISTORICAL_PATH_RELATION_MISMATCH')
        root=PureWindowsPath(*absolute.parts[:-len(relative.parts)])
        if root.name!=row['product'] or permitted not in root.parents:
            raise IntegrityError('HISTORICAL_ROOT_MISMATCH')
        parents.add(root.parent)
    if len(parents)!=1:raise IntegrityError('HISTORICAL_PRODUCT_PARENT_AMBIGUOUS')
    return parents.pop()

def restrict_directory(path:Path)->None:
    if os.name!='nt':raise IntegrityError('ACL_PLATFORM_UNSUPPORTED')
    user=subprocess.check_output(['whoami'],text=True).strip()
    cmd=['icacls',str(path),'/inheritance:r','/grant:r',f'{user}:(OI)(CI)F','*S-1-5-18:(OI)(CI)F','*S-1-5-32-544:(OI)(CI)F']
    if subprocess.run(cmd,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode:raise IntegrityError('ACL_SETUP_FAILED')

def save_csv(path:Path,rows:list[dict])->None:
    if not rows:raise IntegrityError('EMPTY_OUTPUT')
    with path.open('x',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def create_output(path:Path,product_rows:list[dict],group_rows:list[dict],view_rows:list[dict],
                  frame:dict,image_base:PureWindowsPath,shard_hashes:dict)->dict:
    if path.exists():raise FileExistsError('OUTPUT_ALREADY_EXISTS')
    if subprocess.run(['git','check-ignore','-q',str(path)],cwd=ROOT).returncode:raise IntegrityError('OUTPUT_NOT_GIT_IGNORED')
    path.mkdir(parents=True,exist_ok=False);restrict_directory(path)
    whitelist=[]
    for v in view_rows:
        raw=frame['mapping'][v['opaque_image_id']];rel=PureWindowsPath(raw);candidate=image_base/v['product']/rel
        if image_base not in candidate.parents or rel.is_absolute() or rel.drive or '..' in rel.parts:
            raise IntegrityError('TARGET_PATH_INVALID')
        whitelist.append({**v,'source_image_path':str(candidate),'source_sha256_status':'UNKNOWN',
                          'future_file_access_permission':'NOT_AUTHORIZED',
                          'historical_exposure_status':'UNKNOWN','protected_exclusion_status':'UNKNOWN',
                          'source_receipt_sha256':PINNED['extraction_receipt.json'],'whitelist_rule_version':VERSION})
    save_csv(path/'conditional_products.csv',product_rows)
    save_csv(path/'conditional_groups.csv',group_rows)
    save_csv(path/'targeted_sha_whitelist.csv',whitelist)
    reference={'source':'R2b/conditional_identity_order.csv','sha256':PINNED['conditional_identity_order.csv'],
               'group_count':len(frame['expected_order']),'rule':'retain entire fixed product/group order; reverify hash before any replacement',
               'replacement_rule':'eligible exclusions require recorded reason; use next rank in same product/kind; if insufficient stop for researcher decision; no labels or model outcomes',
               'eligibility':'UNKNOWN'}
    (path/'full_order_reference.json').write_text(json.dumps(reference,indent=2),encoding='utf8')
    receipt={'schema':'lp.d6.p3.r2c.shortlist.v1','head':HEAD,'version':VERSION,'input_sha256':PINNED,
             'p2_selection_manifest_sha256':P2_SHA,'shard_sha256':shard_hashes,
             'selection_code_sha256':sha_file(Path(__file__)),'ranking_code_sha256':sha_file(Path(ranking.__file__)),
             'product_count':6,'backup_products':6,'group_count':108,'representative_candidate_groups':36,
             'stress_pool_candidate_groups':72,'view_count':216,'source_sha_unknown_views':216,
             'protected_intersection':'UNKNOWN','cross_source_near_duplicate':'UNKNOWN','fsad_overlap':'UNKNOWN',
             'physical_identity':'INFERRED','formal_partition':False,'stress_selected':False,
             'target_paths_status':'LEXICAL_CANDIDATE_NOT_OPENED_OR_RESOLVED',
             'order_reference_sha256':sha_file(path/'full_order_reference.json'),
             'outputs_sha256':{n:sha_file(path/n) for n in ('conditional_products.csv','conditional_groups.csv','targeted_sha_whitelist.csv')},
             'status':'R2C_CONDITIONAL_SHORTLIST_READY'}
    (path/'selection_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf8')
    return receipt

def public_summary(result:dict)->dict:
    return {k:result[k] for k in ('status','product_count','backup_products','group_count','representative_candidate_groups','stress_pool_candidate_groups','view_count','source_sha_unknown_views','formal_partition','stress_selected')}

def main()->int:
    if OUTPUT.exists():print(json.dumps({'status':'R2C_BLOCKED_INTEGRITY','reason':'OUTPUT_ALREADY_EXISTS'}));return 2
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()
    if head!=HEAD or branch!='stage3-eg-famar-execution':raise IntegrityError('GIT_IDENTITY_MISMATCH')
    receipt,structural,rows,order,aux=verify_inputs(INPUT)
    frame=validate_frame(rows,order,aux['mapping'])
    image_base=infer_image_base()
    prod,groups,views=choose(frame)
    result=create_output(OUTPUT,prod,groups,views,frame,image_base,aux['shard_hashes'])
    print(json.dumps(public_summary(result),indent=2))
    return 0

if __name__=='__main__':
    try:raise SystemExit(main())
    except (IntegrityError,FileNotFoundError,ValueError,KeyError,OSError):
        print(json.dumps({'status':'R2C_BLOCKED_INTEGRITY','reason':'INPUT_OR_SELECTION_VERIFICATION_FAILED'}));raise SystemExit(2)
