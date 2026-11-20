"""Controlled P3-G0-R2b extraction of identities from 12 registered JSON files.

Research source JSON contains hidden fields. Only this module reads it; it never
prints raw records or follows image/mask paths. Outputs and mapping are private.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import hmac
import json
import os
import re
import secrets
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = Path(r'G:\dataset\realiad\jsons\realiad_jsons')
LEDGER = ROOT / 'research_pilot/LP_W1_D5R_SOURCE_METADATA/P1_Source_Metadata_Exposure_Ledger.csv'
OUT = ROOT / 'outputs/research_d6/p3_g0_r2b/run_20261121_01'
PRODUCTS = ('regulator','rolled_strip_base','switch','tape','terminalblock','toy',
            'toy_brick','usb','vcpill','wooden_beads','woodstick','zipper')
LEDGER_SHA = 'd16dfb81f414b5197e80e955b50e636cc2eb69f5b39a3fdb5bad1fa3486f8c90'
EXPECTED_HEAD = 'd0f75a45f10225a53874d659937c139e4c5a9951'
PATTERN = re.compile(r'^([a-zA-Z][a-zA-Z0-9_]*_[0-9]{4}_[A-Z][A-Z_]*[A-Z])_(C[1-9][0-9]?)_')
sys.path.insert(0, str(ROOT))
from scripts.research_d6.p3_g0 import validate_identity_manifest as validator


class SourceIntegrityError(Exception):
    pass


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def pseudonym(secret: bytes, domain: str, value: str) -> str:
    if len(secret) != 32 or domain not in ('IMG','GRP') or not value:
        raise SourceIntegrityError('INVALID_KEY_OR_IDENTITY')
    return domain + '_' + hmac.new(secret, (domain+'\x1f'+value).encode('utf8'), hashlib.sha256).hexdigest()[:32]


def verify_key(secret: bytes, fingerprint: str) -> None:
    if len(secret) != 32 or not hmac.compare_digest(sha(secret), fingerprint):
        raise SourceIntegrityError('KEY_IDENTITY_MISMATCH')


def identity_fields(record: dict, product: str) -> tuple[str, str, str]:
    if not isinstance(record, dict) or record.get('category') != product:
        raise SourceIntegrityError('CATEGORY_OR_RECORD_INVALID')
    raw = record.get('image_path')
    if not isinstance(raw, str) or not raw or '\x00' in raw:
        raise SourceIntegrityError('IMAGE_IDENTITY_INVALID')
    # The pathname is used only as an opaque source identity; it is never opened.
    canonical = raw.replace('\\','/')
    basename = canonical.rsplit('/', 1)[-1]
    match = PATTERN.match(basename)
    if not match or not match.group(1).startswith(product + '_'):
        raise SourceIntegrityError('GROUP_OR_VIEW_UNPARSEABLE')
    return match.group(1), match.group(2), canonical


def extract_document(doc: dict, product: str, secret: bytes, source_sha: str) -> tuple[list[dict],list[dict],dict]:
    if product not in PRODUCTS:
        raise SourceIntegrityError('PRODUCT_NOT_WHITELISTED')
    if not isinstance(doc, dict) or not isinstance(doc.get('meta'),dict) or not isinstance(doc.get('train'),list) or not isinstance(doc.get('test'),list):
        raise SourceIntegrityError('UNEXPECTED_SOURCE_SCHEMA')
    groups = defaultdict(set)
    image_source_set, img_id_to_source, group_id_to_source = set(), {}, {}
    rows, mapping = [], []
    for record in doc['test']:
        group, view, image_path = identity_fields(record, product)
        if image_path in image_source_set:
            raise SourceIntegrityError('DUPLICATE_IMAGE_IDENTITY')
        image_source_set.add(image_path)
        gid = pseudonym(secret,'GRP',product+'\x1f'+group)
        iid = pseudonym(secret,'IMG',product+'\x1f'+image_path)
        if gid in group_id_to_source and group_id_to_source[gid] != group:
            raise SourceIntegrityError('GROUP_HMAC_COLLISION')
        if iid in img_id_to_source and img_id_to_source[iid] != image_path:
            raise SourceIntegrityError('IMAGE_HMAC_COLLISION')
        group_id_to_source[gid], img_id_to_source[iid] = group, image_path
        if view in groups[gid]:
            raise SourceIntegrityError('DUPLICATE_GROUP_VIEW')
        groups[gid].add(view)
        rows.append({
            'opaque_image_id':iid,'product':product,'opaque_group_id':gid,'view':view,
            'source_split':'test','source_sha256':'','exposure_status':'UNKNOWN',
            'allowed_research_role':'UNKNOWN','permission_authority':'RESEARCH_LEAD',
            'permission_record_version':'RESEARCH_LEAD_CHAT_APPROVAL_20261121_R2B',
            'permission_status':'UNKNOWN','exclusion_status':'UNKNOWN','exclusion_check_scope':'UNKNOWN',
            'exclusion_authority':'','exclusion_record_version':'','near_duplicate_status':'UNKNOWN',
            'group_conflict_status':'UNKNOWN','object_identity_confidence':'INFERRED',
            'source_manifest_version':'P1_JSON_'+source_sha[:32],
        })
        mapping.append({'product':product,'opaque_group_id':gid,'opaque_image_id':iid,
                        'source_group_key':group,'source_image_identity':image_path,'view':view})
    rows.sort(key=lambda x:(x['opaque_group_id'],x['view'],x['opaque_image_id']))
    mapping.sort(key=lambda x:(x['opaque_group_id'],x['view'],x['opaque_image_id']))
    return rows,mapping,{'test_views':len(rows),'test_groups':len(groups),
                         'group_view_distribution':dict(sorted(Counter(len(v) for v in groups.values()).items())),
                         'all_C1_C5':sum(v=={'C1','C2','C3','C4','C5'} for v in groups.values())}


def save_csv(path: Path, rows: list[dict], fields: list[str] | tuple[str,...]) -> None:
    with path.open('x',encoding='utf8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='raise');writer.writeheader();writer.writerows(rows)


def read_ledger() -> dict[str,dict]:
    if sha(LEDGER.read_bytes()) != LEDGER_SHA:
        raise SourceIntegrityError('P1_LEDGER_HASH_MISMATCH')
    with LEDGER.open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    result={}
    for row in rows:
        product=row.get('product')
        if product not in PRODUCTS: continue
        if product in result or row.get('relative_json_path') != f'jsons/realiad_jsons/{product}.json':
            raise SourceIntegrityError('P1_LEDGER_PATH_IDENTITY_MISMATCH')
        if not re.fullmatch(r'[0-9a-f]{64}',row.get('sha256','')):
            raise SourceIntegrityError('P1_LEDGER_SOURCE_HASH_INVALID')
        result[product]=row
    if set(result)!=set(PRODUCTS):
        raise SourceIntegrityError('P1_LEDGER_PRODUCT_SET_MISMATCH')
    return result


def restrict_directory(path: Path) -> None:
    if os.name != 'nt':
        raise SourceIntegrityError('PRIVATE_ACL_PLATFORM_UNSUPPORTED')
    user=subprocess.check_output(['whoami'],text=True).strip()
    command=['icacls',str(path),'/inheritance:r','/grant:r',f'{user}:(OI)(CI)F',
             '*S-1-5-18:(OI)(CI)F','*S-1-5-32-544:(OI)(CI)F']
    res=subprocess.run(command,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    if res.returncode != 0:
        raise SourceIntegrityError('PRIVATE_ACL_CONFIGURATION_FAILED')


def main() -> int:
    global OUT
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default=str(OUT),help='new private R2b output directory')
    parser.add_argument('--reuse-from',help='existing R2b private directory containing key and receipt')
    args=parser.parse_args()
    OUT=Path(args.output).resolve()
    allowed=(ROOT/'outputs/research_d6/p3_g0_r2b').resolve()
    if OUT==allowed or allowed not in OUT.parents:
        raise SourceIntegrityError('PRIVATE_OUTPUT_OUTSIDE_ALLOWED_ROOT')
    if OUT.exists():
        print(json.dumps({'status':'BLOCKED','reason':'OUTPUT_ALREADY_EXISTS'}));return 2
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()
    if head != EXPECTED_HEAD or branch != 'stage3-eg-famar-execution':
        raise SourceIntegrityError('GIT_IDENTITY_MISMATCH')
    if subprocess.run(['git','check-ignore','-q',str(OUT)],cwd=ROOT).returncode:
        raise SourceIntegrityError('PRIVATE_OUTPUT_NOT_IGNORED')
    ledger=read_ledger()
    if args.reuse_from:
        previous=Path(args.reuse_from).resolve()
        if previous==allowed or allowed not in previous.parents or previous==OUT:
            raise SourceIntegrityError('PRIVATE_REUSE_OUTSIDE_ALLOWED_ROOT')
        prior_receipt=json.loads((previous/'extraction_receipt.json').read_text(encoding='utf8'))
        secret=(previous/'custodian_hmac_key.bin').read_bytes()
        verify_key(secret,prior_receipt['key_fingerprint_sha256'])
    else:
        secret=secrets.token_bytes(32)
    OUT.mkdir(parents=True,exist_ok=False)
    restrict_directory(OUT)
    key_path=OUT/'custodian_hmac_key.bin'
    with key_path.open('xb') as f:f.write(secret)
    verify_key(key_path.read_bytes(),sha(secret))
    outcomes=[]; all_rows=[]; all_mapping=[]
    ids, group_sources = {}, {}
    for product in PRODUCTS:
        try:
            path=SOURCE_ROOT/f'{product}.json'
            if not path.is_file(): raise SourceIntegrityError('REGISTERED_JSON_MISSING')
            resolved=path.resolve(strict=True)
            if resolved.parent != SOURCE_ROOT.resolve(strict=True) or resolved.name != f'{product}.json':
                raise SourceIntegrityError('REGISTERED_JSON_RESOLVED_PATH_MISMATCH')
            source=resolved.read_bytes()
            digest=sha(source)
            if digest != ledger[product]['sha256']:
                raise SourceIntegrityError('REGISTERED_JSON_SHA_MISMATCH')
            try: doc=json.loads(source)
            except (UnicodeError,json.JSONDecodeError): raise SourceIntegrityError('INVALID_JSON') from None
            rows,mapping,counts=extract_document(doc,product,secret,digest)
            for row in rows:
                key=row['opaque_image_id']
                if key in ids: raise SourceIntegrityError('CROSS_PRODUCT_IMAGE_ID_COLLISION')
                ids[key]=product
                group=row['opaque_group_id']
                if group in group_sources and group_sources[group]!=product:
                    raise SourceIntegrityError('CROSS_PRODUCT_GROUP_ID_COLLISION')
                group_sources[group]=product
            shard=OUT/f'{product}.identity-only.csv'
            save_csv(shard,rows,validator.FIELDS)
            size=shard.stat().st_size
            if size > validator.MAX_BYTES:
                raise SourceIntegrityError('SHARD_EXCEEDS_R1_LIMIT')
            check=validator.validate_file(shard,'D_EVAL',True,'RESEARCH_LEAD_CHAT_R2B',sha(shard.read_bytes()))
            if check['status'] == 'BLOCKED': raise SourceIntegrityError('R1_SHARD_VALIDATION_BLOCKED')
            if check.get('row_count')!=len(rows): raise SourceIntegrityError('R1_SHARD_COUNT_MISMATCH')
            outcomes.append({'product':product,'status':'EXTRACTED','source_sha256':digest,
                'historical_test_groups':int(ledger[product]['test_sample_group_count']),
                'historical_test_views':int(ledger[product]['test_metadata_record_count']),
                'manifest_sha256':sha(shard.read_bytes()),'manifest_bytes':size,
                'r1_status':check['status'],'r1_group_status_counts':check.get('group_status_counts',{}),**counts})
            all_rows.extend(rows); all_mapping.extend(mapping)
        except (OSError,SourceIntegrityError,ValueError,KeyError) as exc:
            reason=str(exc) if isinstance(exc,SourceIntegrityError) else 'SOURCE_OR_OUTPUT_ERROR'
            outcomes.append({'product':product,'status':'BLOCKED','reason':reason})
            # Any error after an output was created is an integrity failure; do not proceed or overwrite.
            if (OUT/f'{product}.identity-only.csv').exists():
                raise SourceIntegrityError('PARTIAL_SHARD_OUTPUT_INTEGRITY_BLOCK') from None
    mapping_path=OUT/'custodian_source_mapping.csv'
    save_csv(mapping_path,all_mapping,['product','opaque_group_id','opaque_image_id','source_group_key','source_image_identity','view'])
    if len({r['opaque_image_id'] for r in all_rows}) != len(all_rows):
        raise SourceIntegrityError('GLOBAL_IMAGE_ID_DUPLICATE')
    group_product={}; group_views=defaultdict(set)
    for row in all_rows:
        gid=row['opaque_group_id'];prod=row['product']
        if gid in group_product and group_product[gid]!=prod: raise SourceIntegrityError('GLOBAL_GROUP_ID_COLLISION')
        group_product[gid]=prod
        if row['view'] in group_views[gid]: raise SourceIntegrityError('GLOBAL_GROUP_VIEW_DUPLICATE')
        group_views[gid].add(row['view'])
    confirmed=0
    receipt={
        'schema':'lp.d6.p3.r2b.extraction.v1','head':head,'branch':branch,
        'authorization':'RESEARCH_LEAD_CHAT_APPROVAL_20261121_R2B',
        'authorization_limit':'identity metadata extraction only; not D-cal/D-eval permission or independent protected-exclusion attestation',
        'ledger_sha256':LEDGER_SHA,'code_sha256':sha(Path(__file__).read_bytes()),
        'key_fingerprint_sha256':sha(secret),'key_access':'private inherited restricted Windows ACL',
        'mapping_sha256':sha(mapping_path.read_bytes()),'source_outcomes':outcomes,
        'extracted_products':sum(x['status']=='EXTRACTED' for x in outcomes),
        'test_views':len(all_rows),'test_groups':len(group_views),
        'historical_test_views':45705,'historical_test_groups':9141,
        'confirmed_groups':confirmed,'unknown_groups':len(group_views),'ineligible_groups':0,
        'source_sha_present_views':sum(bool(r['source_sha256']) for r in all_rows),
        'source_sha_missing_views':sum(not r['source_sha256'] for r in all_rows),
        'group_view_distribution':dict(sorted(Counter(len(v) for v in group_views.values()).items())),
        'physical_object_independence':'INFERRED','protected_overlap':'UNKNOWN',
        'cross_source_near_duplicates':'UNKNOWN','fsad_overlap':'UNKNOWN',
        'identity_hmac_collision_check':'PASS','cross_shard_duplicate_image_group_view_check':'PASS',
        'image_source_sha_duplicate_check':'UNKNOWN_ALL_IMAGE_SOURCE_SHA_MISSING',
        'independent_custodian_manifest_sha':'NOT_PROVIDED',
        'mixed_json_read':True,'images_read':False,'mask_files_read':False,'protected_assets_read':False,
        'model_or_annotation_run':False,
        'status':'R2B_IDENTITY_EXPORTED_ELIGIBILITY_PENDING' if len(outcomes)==12 and all(x['status']=='EXTRACTED' for x in outcomes) else 'R2B_PARTIAL_SOURCE_EXTRACTION',
    }
    with (OUT/'extraction_receipt.json').open('x',encoding='utf8') as f:json.dump(receipt,f,ensure_ascii=False,indent=2)
    public={k:receipt[k] for k in ('status','extracted_products','test_views','test_groups','confirmed_groups','unknown_groups','ineligible_groups','source_sha_missing_views','group_view_distribution','protected_overlap','cross_source_near_duplicates','fsad_overlap')}
    print(json.dumps(public,ensure_ascii=False,indent=2))
    return 0 if receipt['status']=='R2B_IDENTITY_EXPORTED_ELIGIBILITY_PENDING' else 2


if __name__=='__main__':
    try: raise SystemExit(main())
    except (SourceIntegrityError,OSError,ValueError,KeyError):
        # Never print source records, keys, mapping, or exception strings with source paths.
        print(json.dumps({'status':'R2B_BLOCKED_INTEGRITY','reason':'CONTROLLED_EXTRACTION_FAILURE'}))
        raise SystemExit(2)
