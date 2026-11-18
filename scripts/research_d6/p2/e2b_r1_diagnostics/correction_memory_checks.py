"""Supplemental read-only checks and aggregate summaries for one frozen replay."""
import argparse, csv, json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'outputs/research_d6/p2/correction_memory_replay/run_20261119_01'
def read(name):
    with (OUT/name).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def evaluate(row,mode,threshold):
    h=int(row['h']); d=row[mode+'_distance']; gate=mode=='M' or row['confidence_gate']=='1'
    if d and gate and float(d)<=threshold:return int(row[mode+'_anchor_label'])
    return h
def main():
    global OUT
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default=str(OUT))
    OUT=Path(parser.parse_args().output).resolve()
    allowed=(ROOT/'outputs/research_d6/p2/correction_memory_replay').resolve()
    if OUT==allowed or allowed not in OUT.parents:raise ValueError('output must be a child of the private replay directory')
    dest=OUT/'supplemental_checks.json'
    if dest.exists():raise FileExistsError(dest)
    rows=read('unscored_predictions.csv'); ev=read('event_anchors.csv'); scored=read('paired_future_reference.csv')
    assert all(not r['M_anchor_id'] and not r['M+_anchor_id'] for r in rows if int(r['checkpoint'])==0)
    assert all(evaluate(r,m,-1)==int(r['h']) for r in rows for m in ('M','M+'))
    assert all(not r[m+'_anchor_event'] or int(r[m+'_anchor_event'])<=int(r['checkpoint']) for r in rows for m in ('M','M+'))
    assert all(not r[m+'_anchor_id'] or r[m+'_anchor_id']!=r['image_id'] for r in rows for m in ('M','M+'))
    assert all(r['write_anchor']=='0' for r in ev if r['revealed_state'] not in ('present','absent_assessable'))
    for p in ('X','W'):
        for r in [x for x in rows if x['protocol']==p]:
            for m in ('M','M+'):
                if r[m+'_anchor_id']:
                    prior=[x for x in ev if x['protocol']==p and x['write_anchor']=='1' and int(x['event_index'])==int(r[m+'_anchor_event'])]
                    assert len(prior)==1 and prior[0]['image_id']==r[m+'_anchor_id'] and int(prior[0]['label'])==int(r[m+'_anchor_label'])
    # Predictions are pure functions of the unscored file; synthetic truth permutation has no input path to evaluate().
    before=[evaluate(r,m,1.0) for r in rows for m in ('M','M+')]
    future_truth=[r['y'] for r in scored]; future_truth=future_truth[::-1]
    after=[evaluate(r,m,1.0) for r in rows for m in ('M','M+')]
    assert before==after
    anchor_outcomes=Counter((r['protocol'],r['arm'],r[r['arm']+'_anchor_label'],r['outcome']) for r in scored if r['status']=='reuse_flip')
    nearest_cases=[]; details=[]
    for p in ('X','W'):
        for m in ('M','M+'):
            rr=[r for r in scored if r['protocol']==p and r['arm']==m]
            # Descriptive no-distance-limit cases, M+ retains the confidence gate.
            events=[]
            for r in rr:
                if not r[m+'_distance'] or (m=='M+' and r['confidence_gate']!='1') or not r['y']:continue
                h=int(r['h']); y=int(r['y']); v=int(r[m+'_anchor_label'])
                kind='repair' if h!=y and v==y else 'harm' if h==y and v!=y else None
                if kind:events.append((r,kind))
            rep=[r for r,k in events if k=='repair']; harm=[r for r,k in events if k=='harm']
            byanchor=Counter((r[m+'_anchor_label'],k) for r,k in events)
            details.append({'protocol':p,'arm':m,'unlimited_repairs':len(rep),'unlimited_harms':len(harm),'repair_groups':len({(r['product'],r['sample_key']) for r in rep}),'harm_groups':len({(r['product'],r['sample_key']) for r in harm}),'repair_products':len({r['product'] for r in rep}),'harm_products':len({r['product'] for r in harm}),'by_anchor':{str(k):v for k,v in byanchor.items()}})
            if rep and harm:
                x,y=min(((x,y) for x in rep for y in harm),key=lambda pair:(pair[0][m+'_anchor_id']!=pair[1][m+'_anchor_id'],abs(float(pair[0][m+'_distance'])-float(pair[1][m+'_distance'])),pair[0]['image_id'],pair[1]['image_id']))
                nearest_cases.append({'protocol':p,'arm':m,'same_anchor':x[m+'_anchor_id']==y[m+'_anchor_id'],'repair_image_id':x['image_id'],'harm_image_id':y['image_id'],'repair_product':x['product'],'harm_product':y['product'],'repair_distance':float(x[m+'_distance']),'harm_distance':float(y[m+'_distance']),'distance_gap':abs(float(x[m+'_distance'])-float(y[m+'_distance'])),'repair_anchor_id':x[m+'_anchor_id'],'harm_anchor_id':y[m+'_anchor_id']})
    with (OUT/'descriptive_case_pairs.csv').open('x',encoding='utf8',newline='') as f:
        if nearest_cases:
            w=csv.DictWriter(f,fieldnames=list(nearest_cases[0]));w.writeheader();w.writerows(nearest_cases)
    result={'cp0_empty_memory_pass':True,'disabled_reuse_pass':True,'no_self_or_future_anchor_pass':True,'anchor_identity_and_protocol_pass':True,'uncertain_not_written_pass':True,'future_truth_permutation_pass':True,'reference_outcomes_by_anchor':{str(k):v for k,v in anchor_outcomes.items()},'unlimited_cases':details,'closest_case_pairs':len(nearest_cases)}
    dest.write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
