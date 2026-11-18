"""Development-only replay of correction memory against frozen R1 G-Mean heads."""
from __future__ import annotations
import argparse, csv, hashlib, json, math, os, subprocess, sys
from pathlib import Path
from collections import Counter, defaultdict
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[k]='1'
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import attribution_spatial as a
ROOT=a.ROOT
sys.path.insert(0,str(ROOT/'scripts/research_d6/p2/e2b'))
import config
import evaluator
OUT=ROOT/'outputs/research_d6/p2/correction_memory_replay/run_20261119_01'
Y=a.STATE_TO_Y

def write(path,rows):
    if not rows: path.write_text('',encoding='utf8'); return
    with path.open('x',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def self_sha(r):
    body={k:v for k,v in r.items() if k!='record_sha256'}
    return hashlib.sha256(a.canonical_json_bytes(body)).hexdigest()

def local(feature):
    z=np.asarray(feature['patches'],dtype=np.float64)
    mask=feature['mask']; n=len(mask); side=math.isqrt(n)
    if side*side!=n: raise ValueError('patch grid cannot be reconstructed')
    ids=np.flatnonzero(mask); cells=[]
    for i in range(4):
        for j in range(4):
            keep=(ids//side*4//side==i)&(ids%side*4//side==j)
            if keep.any():
                v=z[keep].mean(axis=0); norm=np.linalg.norm(v)
                if norm>1e-12: cells.append(v/norm)
    if not cells: raise ValueError('no nonzero local cells')
    return np.stack(cells)

def dist(x,y): return float(np.clip((1-np.dot(x,y))/2,0,1))
def locdist(x,y):
    d=np.clip((1-x@y.T)/2,0,1)
    return float((d.min(axis=1).mean()+d.min(axis=0).mean())/2)

def nearest(img, anchors, feats, mode):
    if not anchors: return None
    vals=[]
    for q in anchors:
        dg=dist(feats[img]['global'],feats[q['image_id']]['global'])
        dl=locdist(feats[img]['local'],feats[q['image_id']]['local']) if mode=='M+' else None
        vals.append((dg if mode=='M' else (dg+dl)/2,q['event_index'],q['image_id'],q,dg,dl))
    return min(vals,key=lambda x:(x[0],x[1],x[2]))

def summary(items,pred):
    rows=[r for r in items if r['y'] is not None]; p=np.asarray([pred[r['image_id']] for r in rows]); y=np.asarray([r['y'] for r in rows]); h=np.asarray([r['h'] for r in rows]);
    errors=h!=y; correct=~errors; repair=(errors & (p==y)); harm=(correct & (p!=y))
    assert int(repair.sum()-harm.sum())==int((p==y).sum()-(h==y).sum())
    groups=lambda ix: len({(r['product'],r['sample_key']) for r,b in zip(rows,ix) if b})
    products=lambda ix: len({r['product'] for r,b in zip(rows,ix) if b})
    byprod=[]
    for name in sorted({r['product'] for r in rows}):
        idx=np.array([r['product']==name for r in rows]); yy=y[idx]; pp=p[idx]
        if (yy==1).any() and (yy==0).any(): byprod.append((np.mean(pp[yy==1]==1)+np.mean(pp[yy==0]==0))/2)
    tp=int(((y==1)&(p==1)).sum()); fn=int(((y==1)&(p==0)).sum()); tn=int(((y==0)&(p==0)).sum()); fp=int(((y==0)&(p==1)).sum())
    return dict(n=len(rows),h_errors=int(errors.sum()),h_correct=int(correct.sum()),error_groups=groups(errors),correct_groups=groups(correct),repair=int(repair.sum()),harm=int(harm.sum()),repair_groups=groups(repair),harm_groups=groups(harm),repair_products=products(repair),harm_products=products(harm),repair_rate=float(repair.sum()/errors.sum()) if errors.any() else None,harm_rate=float(harm.sum()/correct.sum()) if correct.any() else None,tp=tp,fn=fn,tn=tn,fp=fp,fnr=fn/(tp+fn) if tp+fn else None,fpr=fp/(fp+tn) if fp+tn else None,ba=float(np.mean(byprod)) if byprod else None,ba_products=len(byprod))

def main():
    global OUT
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default=str(OUT),help='new directory beneath outputs/research_d6/p2/correction_memory_replay')
    args=parser.parse_args()
    OUT=Path(args.output).resolve()
    allowed=(ROOT/'outputs/research_d6/p2/correction_memory_replay').resolve()
    if OUT==allowed or allowed not in OUT.parents: raise ValueError('output must be a new child of the private replay directory')
    if OUT.exists(): raise FileExistsError(OUT)
    if subprocess.run(['git','check-ignore','-q',str(OUT)],cwd=ROOT).returncode!=0: raise RuntimeError('private path not ignored')
    identity=a.read_json(a.RUN/'run_identity.json')
    hashes=a.current_input_hashes(); assert hashes==identity['input_hashes']
    assert a.sha256_file(a.E2A1/'dinov2/feature_manifest.csv')==identity['feature_manifest_sha256']
    assert a.sha256_file(a.E2A1/'canonical_manifest.csv')==identity['canonical_manifest_sha256']
    manifest=a.read_csv(ROOT/'outputs/research_d6/p1a_features/input_manifest.csv')
    dfit,ddev,source,future_ids,dfit_ids=a.parse_frozen_sources(manifest)
    assert Counter(Y[x] for x in dfit.values())==Counter({1:27,0:25,None:2})
    tape=a.read_csv(config.TAPE_PATH)
    # Feature identity is checked against the frozen manifest and per-item array hashes.
    cache=a.load_feature_cache('dinov2',manifest)
    feats={i:{'global':a.global_feature('G-Mean',v),'local':local(v)} for i,v in cache.items() if source[i]['data_role']=='D-dev'}
    all_unscored=[]; events=[]; refs={}; meta={}; maxdiff=0.0
    for protocol in ('X','W'):
        query=sorted([r for r in tape if r['protocol']==protocol and r['feedback_or_future_role']=='feedback_query'],key=lambda r:int(r['stream_order']))
        futures=[r for r in tape if r['protocol']==protocol and r['feedback_or_future_role']=='future_eval_candidate']
        assert len(query)==12 and len(futures)==24 and len({r['image_id'] for r in query})==12
        qgroup={(r['product'],r['sample_key']) for r in query}; fgroup={(r['product'],r['sample_key']) for r in futures}
        assert len(fgroup)==12 and not qgroup&fgroup and len(qgroup)==12
        assert Counter((r['product'],r['sample_key']) for r in futures)==Counter({g:2 for g in fgroup})
        if protocol=='X': assert not {r['product'] for r in query}&{r['product'] for r in futures}
        assert Counter(ddev[r['image_id']] for r in query)==(Counter({'present':7,'absent_assessable':5}) if protocol=='X' else Counter({'present':5,'absent_assessable':6,'unobservable_or_uncertain':1}))
        assert Counter(ddev[r['image_id']] for r in futures)==(Counter({'present':12,'absent_assessable':9,'unobservable_or_uncertain':3}) if protocol=='X' else Counter({'present':14,'absent_assessable':7,'unobservable_or_uncertain':3}))
        pair=f'dinov2__G-Mean__{protocol}'; base=a.RUN/'trajectories'/pair
        L=[a.verify_r1_bundle(base/'learner'/f'checkpoint_{j:02d}') for j in range(13)]
        C=[a.verify_r1_bundle(base/'control'/f'checkpoint_{j:02d}') for j in range(13)]
        assert np.array_equal(L[0][1]['theta'],C[0][1]['theta'])
        for rec,arr in C:
            assert np.array_equal(arr['theta'],C[0][1]['theta']) and np.array_equal(arr['future_scores'],C[0][1]['future_scores'])
        anchors=[]; histdist={'M':[],'M+':[]}; absqueries=[]
        for j in range(13):
            rec,arr=L[j]; ids=[x['image_id'] for x in rec['item_predictions']]
            assert set(ids)==set(r['image_id'] for r in futures)
            scores=np.asarray([feats[i]['global']@arr['theta'][:-1]+arr['theta'][-1] for i in ids])
            delta=a.require_score_match(scores,arr['future_scores']); maxdiff=max(maxdiff,delta)
            for i,score in zip(ids,scores):
                row={'protocol':protocol,'checkpoint':j,'image_id':i,'product':source[i]['product'],'sample_key':source[i]['sample_key'],'h_score':float(score),'h':int(score>=0),'confidence_gate':None if not absqueries else int(abs(score)<=float(np.median(absqueries))),'confidence_median':None if not absqueries else float(np.median(absqueries))}
                for mode in ('M','M+'):
                    n=nearest(i,anchors,feats,mode)
                    assert n is None or n[1]<=j
                    row[mode+'_distance']=None if n is None else n[0]; row[mode+'_anchor_event']=None if n is None else n[1]; row[mode+'_anchor_id']=None if n is None else n[2]; row[mode+'_anchor_label']=None if n is None else n[3]['label']; row[mode+'_global_distance']=None if n is None else n[4]; row[mode+'_local_distance']=None if n is None else n[5]
                all_unscored.append(row)
            if j==12: break
            q=query[j]; event=a.read_json(base/'learner'/'events'/f'event_{j+1:02d}.json'); prepared=a.read_json(base/'learner'/'events'/f'prepared_{j+1:02d}.json'); nextrec=L[j+1][0]; cp_event=nextrec['event']
            assert self_sha(event)==event['record_sha256'] and self_sha(prepared)==prepared['record_sha256']
            assert event['query_record_sha256']==prepared['record_sha256'] and event['record_sha256']==cp_event['revealed_event_sha256']==nextrec['event_record_sha256']
            assert event['identity']==prepared['identity']==rec['identity']==nextrec['identity']
            assert event['event_id']==q['event_id']==prepared['event_id'] and event['image_id']==q['image_id']==prepared['image_id'] and event['stream_order']==j+1==int(q['stream_order'])
            state_sha=hashlib.sha256(arr['theta'].tobytes()).hexdigest(); assert state_sha==event['state_sha256_before_reveal']==cp_event['query_state_sha256_before_reveal']
            s=float(feats[q['image_id']]['global']@arr['theta'][:-1]+arr['theta'][-1]); assert abs(s-event['score_before_reveal'])<=1e-8 and abs(s-cp_event['query_score_before_reveal'])<=1e-8
            assert event['revealed_state']==ddev[q['image_id']]==cp_event['feedback_state_after_prediction'] and event['binary_target_after_reveal']==Y[ddev[q['image_id']]]
            for mode in ('M','M+'):
                n=nearest(q['image_id'],anchors,feats,mode)
                if n: assert n[1]<j+1; histdist[mode].append(n[0])
            y=Y[ddev[q['image_id']]]; wrong=y is not None and int(s>=0)!=y
            events.append({'protocol':protocol,'event_index':j+1,'event_id':q['event_id'],'image_id':q['image_id'],'product':q['product'],'sample_key':q['sample_key'],'query_score_before_reveal':s,'state_sha256':state_sha,'revealed_state':ddev[q['image_id']],'label':y,'write_anchor':int(wrong),'anchors_before':len(anchors)})
            if wrong: anchors.append({'event_index':j+1,'image_id':q['image_id'],'label':y})
            absqueries.append(abs(s))
        refs[protocol]={mode:{'threshold':float(np.median(histdist[mode])) if histdist[mode] else None,'n':len(histdist[mode])} for mode in ('M','M+')}
        meta[protocol]={'anchors_present':sum(x['label']==1 for x in anchors),'anchors_absent':sum(x['label']==0 for x in anchors),'future_groups':len(fgroup),'feedback_groups':len(qgroup),'future_products':len({r['product'] for r in futures})}
    OUT.mkdir(parents=True,exist_ok=False)
    write(OUT/'unscored_predictions.csv',all_unscored); write(OUT/'event_anchors.csv',events)
    (OUT/'reference_thresholds.json').write_text(json.dumps(refs,indent=2),encoding='utf8')
    # Future labels are joined only after all predictions and nearest-neighbour records are persisted.
    unscored=a.read_csv(OUT/'unscored_predictions.csv'); scored=[]; curves=[]; cases=[]; products=[]; groups=[]; stats=[]
    for protocol in ('X','W'):
        rows=[r for r in unscored if r['protocol']==protocol and int(r['checkpoint'])==12]
        items=[{'image_id':r['image_id'],'product':r['product'],'sample_key':r['sample_key'],'state':ddev[r['image_id']],'y':Y[ddev[r['image_id']]],'h':int(r['h'])} for r in rows]
        def evaluate(mode,threshold):
            pred={}; statuses={}
            for r in rows:
                h=int(r['h']); distance=r[mode+'_distance']; anchor=r[mode+'_anchor_label']; gate=(mode=='M' or r['confidence_gate']=='1')
                status='no_anchor' if not distance else 'distance_rejected' if threshold is None or float(distance)>threshold else 'confidence_rejected' if not gate else 'reuse_same' if int(anchor)==h else 'reuse_flip'
                pred[r['image_id']]=int(anchor) if status in ('reuse_same','reuse_flip') else h; statuses[r['image_id']]=status
            return pred,statuses
        h={r['image_id']:int(r['h']) for r in rows}; hs=summary(items,h); stats.append({'protocol':protocol,'arm':'H',**hs})
        for mode in ('M','M+'):
            threshold=refs[protocol][mode]['threshold']; pred,statuses=evaluate(mode,threshold)
            st=summary(items,pred); stats.append({'protocol':protocol,'arm':mode,**st})
            for r,item in zip(rows,items):
                y=item['y']; p=pred[r['image_id']]; outcome='unknown' if y is None else 'repair' if item['h']!=y and p==y else 'harm' if item['h']==y and p!=y else 'unchanged_or_other'
                scored.append({**r,'arm':mode,'reference_threshold':threshold,'status':statuses[r['image_id']],'frozen_state':item['state'],'y':y,'prediction':p,'outcome':outcome})
            thresholds=sorted({float(r[mode+'_distance']) for r in rows if r[mode+'_distance'] and (mode=='M' or r['confidence_gate']=='1')})
            for label,t in [('never',None)]+[('breakpoint',x) for x in thresholds]+[('unlimited',float('inf'))]:
                pred2,status2=evaluate(mode,t); st2=summary(items,pred2)
                curves.append({'protocol':protocol,'arm':mode,'point':label,'threshold':t,**st2,'reuse_count':sum(x in ('reuse_same','reuse_flip') for x in status2.values())})
            for product in sorted({r['product'] for r in items}):
                subset=[r for r in items if r['product']==product]; products.append({'protocol':protocol,'arm':mode,'product':product,**summary(subset,pred)})
            for group in sorted({(r['product'],r['sample_key']) for r in items}):
                subset=[r for r in items if (r['product'],r['sample_key'])==group]; changes=sum(pred[x['image_id']]!=x['h'] for x in subset)
                if changes: groups.append({'protocol':protocol,'arm':mode,'product':group[0],'sample_key':group[1],'flipped_views':changes,**summary(subset,pred)})
    write(OUT/'paired_future_reference.csv',scored); write(OUT/'threshold_curve.csv',curves); write(OUT/'reference_metrics.csv',stats); write(OUT/'product_metrics.csv',products); write(OUT/'affected_groups.csv',groups)
    fig,axes=plt.subplots(1,2,figsize=(9,4),sharey=True)
    for ax,protocol in zip(axes,('X','W')):
        for mode in ('M','M+'):
            rr=[r for r in curves if r['protocol']==protocol and r['arm']==mode]; ax.step([r['repair'] for r in rr],[r['harm'] for r in rr],where='post',marker='.',label=mode)
        ax.set_title(protocol);ax.set_xlabel('Repaired views');ax.grid(alpha=.3)
    axes[0].set_ylabel('Harmed views');axes[1].legend(); fig.tight_layout();fig.savefig(OUT/'repair_harm_curves.png',dpi=150);plt.close(fig)
    (OUT/'run_metadata.json').write_text(json.dumps({'head':os.popen('git rev-parse HEAD').read().strip(),'run_identity_sha256':a.sha256_file(a.RUN/'run_identity.json'),'inputs':hashes,'feature_manifest':identity['feature_manifest_sha256'],'references':refs,'protocols':meta,'max_score_rebuild_delta':maxdiff,'script_sha256':a.sha256_file(Path(__file__))},indent=2),encoding='utf8')
    print(json.dumps({'output':str(OUT),'references':refs,'protocols':meta,'metrics':stats,'max_score_rebuild_delta':maxdiff},indent=2))

if __name__=='__main__': main()
