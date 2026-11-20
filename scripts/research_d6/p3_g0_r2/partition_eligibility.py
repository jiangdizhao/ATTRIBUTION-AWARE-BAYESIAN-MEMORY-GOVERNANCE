"""Pure, identity-only P3 2+4 eligibility checks. No source file discovery.

Inputs must already have been lawfully supplied and authenticated. This module
does not read images, masks, JSON datasets, labels, or feature tensors and does
not itself write candidate identities or make a custodian assertion.
"""
from __future__ import annotations

import hashlib
import hmac
import unicodedata
from collections import Counter, defaultdict

from scripts.research_d6.p3_g0 import validate_identity_manifest as r1

SALT = 'LP-D6-P3-G0-v0.2-20261120'
SEP = '\x1f'


def order_key(domain: str, key: str) -> tuple[bytes, bytes]:
    domain, key = unicodedata.normalize('NFC', domain), unicodedata.normalize('NFC', key)
    if not domain or SEP in domain:
        raise ValueError('invalid domain')
    encoded = key.encode('utf-8')
    return hashlib.sha256((SALT + SEP + domain + SEP + key).encode('utf-8')).digest(), encoded


def opaque_id(secret: bytes, domain: str, source_key: str) -> str:
    """Custodian-only HMAC illustration; never publish the secret/source map."""
    if len(secret) < 32 or domain not in {'IMG', 'GRP'}:
        raise ValueError('custodian secret/domain invalid')
    key = unicodedata.normalize('NFC', source_key)
    if not key or SEP in key:
        raise ValueError('invalid source identity')
    return domain + '_' + hmac.new(secret, (domain + SEP + key).encode('utf-8'), hashlib.sha256).hexdigest()[:32]


def role_sets(cal_products: list[str], eval_products: list[str]) -> dict:
    cal, evaluation = set(cal_products), set(eval_products)
    if len(cal_products) != len(cal) or len(eval_products) != len(evaluation) or cal & evaluation:
        return {'status': 'BLOCKED', 'reason': 'PRODUCT_CROSSES_ROLES'}
    if len(cal) != 2 or len(evaluation) != 4:
        return {'status': 'BLOCKED', 'reason': 'PRODUCT_ROLE_COUNTS'}
    return {'status': 'PASS'}


def validate_partition(rows: list[dict], assigned: dict[str, str], relation_status: str) -> dict:
    """Check already proposed product roles against grouped authorized rows.

    A custodian must separately attest that same-object and identifiable near
    duplicate relationships are resolved across the entire candidate frame.
    The assertion is not independently authenticated by this function.
    """
    if relation_status != 'CLEAR_FRAME':
        return {'status': 'UNKNOWN', 'reason': 'CROSS_GROUP_RELATION_NOT_CLEARED'}
    if (set(assigned.values()) != {'D_CAL', 'D_EVAL'} or
            role_sets([k for k,v in assigned.items() if v == 'D_CAL'],
                      [k for k,v in assigned.items() if v == 'D_EVAL'])['status'] != 'PASS'):
        return {'status': 'BLOCKED', 'reason': 'PRODUCT_ROLE_COUNTS'}
    roles_by_group = defaultdict(set)
    for row in rows:
        role = assigned.get(row['product'])
        if role:
            roles_by_group[row['opaque_group_id']].add(role)
    if any(len(s) > 1 for s in roles_by_group.values()):
        return {'status': 'BLOCKED', 'reason': 'GROUP_CROSSES_ROLES'}
    return {'status': 'PASS', 'groups': len(roles_by_group)}


def assess(rows: list[dict] | None, *, authorized: bool, manifest_verified: bool,
           relation_status: str = 'UNKNOWN') -> tuple[dict, dict | None]:
    """Return a public aggregate and, only on success, a private role plan.

    Explicit caller flags are not a substitute for independent authorization,
    source SHA verification or the R1 validator's own file-level checks.
    """
    if not authorized or not manifest_verified or rows is None:
        return {'status': 'BLOCKED', 'reason': 'MISSING_AUTHORIZED_VERIFIED_MANIFEST'}, None
    if not rows:
        return {'status': 'BLOCKED', 'reason': 'EMPTY_MANIFEST'}, None
    checks = {role: r1.validate_rows(rows, role) for role in ('D_CAL', 'D_EVAL')}
    if any(check['status'] == 'BLOCKED' for check in checks.values()):
        return {'status': 'BLOCKED', 'reason': 'R1_INTEGRITY_BLOCK'}, None
    counts_sha = Counter(row['source_sha256'] for row in rows if row['source_sha256'])
    dup_sha = {sha for sha, count in counts_sha.items() if count > 1}
    groups = defaultdict(list)
    for row in rows:
        groups[(row['product'], row['opaque_group_id'])].append(row)
    usable = {role: defaultdict(list) for role in ('D_CAL', 'D_EVAL')}
    unknown = {role: 0 for role in ('D_CAL', 'D_EVAL')}
    non_test = 0
    for (product, gid), members in groups.items():
        if any(row['source_split'] != 'test' for row in members):
            non_test += 1
            continue
        if any(row['source_sha256'] in dup_sha for row in members if row['source_sha256']):
            continue
        for role in ('D_CAL', 'D_EVAL'):
            result = r1.validate_rows(members, role)
            group_counts = result.get('group_status_counts', {})
            if group_counts.get('BLOCKED', 0):
                return {'status': 'BLOCKED', 'reason': 'R1_GROUP_INTEGRITY_BLOCK'}, None
            if group_counts.get('PASS', 0) == 1:
                usable[role][product].append((gid, members))
            elif group_counts.get('UNKNOWN', 0) == 1:
                unknown[role] += 1
    products = sorted({row['product'] for row in rows}, key=lambda p: order_key('PRODUCT', p))
    aggregate = {
        'status': 'UNKNOWN', 'nominal_products': len(products), 'nominal_groups': len(groups),
        'confirmed_products_10_cal': sum(len(v) >= 10 for v in usable['D_CAL'].values()),
        'confirmed_products_10_eval': sum(len(v) >= 10 for v in usable['D_EVAL'].values()),
        'confirmed_products_18_cal': sum(len(v) >= 18 for v in usable['D_CAL'].values()),
        'confirmed_products_18_eval': sum(len(v) >= 18 for v in usable['D_EVAL'].values()),
        'unknown_groups_cal': unknown['D_CAL'], 'unknown_groups_eval': unknown['D_EVAL'],
        'non_test_groups': non_test, 'duplicate_source_sha_count': len(dup_sha),
        'physical_independence': 'NOT_PROVEN_BY_THIS_CHECK',
    }
    # Only products with at least 10 individually confirmed groups for at least
    # one role enter the frozen ranking. Never use evaluator or defect labels.
    eligible = [p for p in products if len(usable['D_CAL'][p]) >= 10 or len(usable['D_EVAL'][p]) >= 10]
    if len(eligible) < 6:
        aggregate['reason'] = 'FEWER_THAN_SIX_CONFIRMED_PRODUCTS'
        return aggregate, None
    chosen = eligible[:6]
    assigned = {p: 'D_CAL' if i < 2 else 'D_EVAL' for i, p in enumerate(chosen)}
    if any(len(usable[role][p]) < 10 for p, role in assigned.items()):
        aggregate['reason'] = 'FIXED_PRODUCT_ROLE_CAPACITY_INSUFFICIENT'
        return aggregate, None
    part = validate_partition(rows, assigned, relation_status)
    if part['status'] != 'PASS':
        aggregate['reason'] = part['reason']
        return aggregate, None
    plan = []
    for product in chosen:
        role = assigned[product]
        entries = sorted(usable[role][product], key=lambda e: order_key('REP_GROUP', product + SEP + e[0]))
        for position, (gid, members) in enumerate(entries):
            views = sorted(members, key=lambda r: order_key('VIEW', product + SEP + gid + SEP + r['view']))
            plan.append({'product': product, 'role': role, 'opaque_group_id': gid,
                         'treatment': 'REPRESENTATIVE_SELECTED' if position < 6 else 'STRESS_POOL_ONLY',
                         'view_ids': [r['opaque_image_id'] for r in views[:2]],
                         'views': [r['view'] for r in views[:2]]})
    aggregate.update(status='PASS', selected_products=6, cal_products=2, eval_products=4,
                     representative_groups=36, stress_selected=0,
                     stress_pool_groups=sum(x['treatment'] == 'STRESS_POOL_ONLY' for x in plan),
                     stress_selection='STRESS_SELECTION_PENDING_FEATURE_AUTHORIZATION')
    return aggregate, {'assignments': assigned, 'groups': plan}
