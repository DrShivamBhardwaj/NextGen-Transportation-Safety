"""Participant-disjoint prospective pedestrian-collision analysis."""
from __future__ import annotations
import json,re,hashlib,time,sys,platform
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.model_selection import GroupKFold,GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from fusion import score,sigmoid_fit,metrics
ROOT=Path(__file__).resolve().parents[1]
PED={"2310001":0,"2320001":1}
COLLISION={"4311001","4312001","4313001"}
INCORRECT={"4111201","4111301","4112201","4112301",
           "4121201","4121301","4122201","4122301","2781021","2791021"}
CORRECT={"4111101","4112101","4121101","4122101"}
LANE={"1410001","1420001","2551001"}
FOLLOW={"1510001","1520001","2561001"}
PERTURB={"1310001","1320001","1330001","1340001"}
TOGGLE={"4211201","4211301","4222101","4232101","4242101"}
ANY_COLLISION=COLLISION|{"4321001","4322001","4323001","4331001","4332001","4333001"}
ROAD={"3210001":0,"3221001":1,"3222001":1,"3231001":1,"3232001":1,"3240001":1}
CONTEXT=["dynamic_pedestrian","road_curve","recent_perturb_20s"]
DRIVER=["recent_incorrect_20s","recent_correct_20s","recent_lane_20s",
        "recent_follow_20s","recent_toggle_20s","prior_collisions_60s","time_since_incorrect"]
VEHICLE=["bad_reliability","full_autonomy","speed_autonomy","manual"]
INTERACTIONS=["ix_dynamic_incorrect","ix_dynamic_violations","ix_bad_incorrect","ix_bad_violations"]
FEATURES={"Context only":CONTEXT,"Driver history":DRIVER,"Automation only":VEHICLE,
          "Additive":CONTEXT+DRIVER+VEHICLE,"Interaction":CONTEXT+DRIVER+VEHICLE+INTERACTIONS,
          "HGB":CONTEXT+DRIVER+VEHICLE+INTERACTIONS,"MLP":CONTEXT+DRIVER+VEHICLE+INTERACTIONS}

def instances(events,subject,session,horizon=8.):
    events=events.sort_values("onset",kind="stable")
    end=float(events.onset.max())
    hazards=[]
    for r in events[events.value.isin(PED)].itertuples():
        t=float(r.onset)
        if not hazards or t-hazards[-1][0]>.75: hazards.append([t,PED[r.value]])
        else: hazards[-1][1]=max(hazards[-1][1],PED[r.value])
    collisions=events.loc[events.value.isin(COLLISION),"onset"].to_numpy(float)
    rows=[]; censored=0
    for t,dynamic in hazards:
        if t+horizon>end: censored+=1;continue
        prior=events[events.onset<t]
        recent=prior[prior.onset>=t-20]
        counts=lambda codes:int(recent.value.isin(codes).sum())
        inc=counts(INCORRECT); cor=counts(CORRECT); lane=counts(LANE);follow=counts(FOLLOW)
        roads=prior[prior.value.isin(ROAD)]
        incorrect=prior[prior.value.isin(INCORRECT)]
        future=collisions[(collisions>t)&(collisions<=t+horizon)]
        bad=int(session in ["SCFB","SCSB"])
        r={"subject":subject,"session":session,"onset":t,"y":int(len(future)>0),
           "lead_seconds":float(future[0]-t) if len(future) else np.nan,
           "dynamic_pedestrian":dynamic,"road_curve":ROAD[roads.iloc[-1].value] if len(roads) else 0,
           "bad_reliability":bad,"full_autonomy":int(session in ["SCFB","SCFG"]),
           "speed_autonomy":int(session in ["SCSB","SCSG"]),"manual":int(session=="SCMM"),
           "recent_incorrect_20s":inc,"recent_correct_20s":cor,"recent_lane_20s":lane,
           "recent_follow_20s":follow,"recent_perturb_20s":counts(PERTURB),
           "recent_toggle_20s":counts(TOGGLE),
           "prior_collisions_60s":int(prior[prior.onset>=t-60].value.isin(ANY_COLLISION).sum()),
           "time_since_incorrect":min(60.,t-float(incorrect.iloc[-1].onset)) if len(incorrect) else 60.,
           "ix_dynamic_incorrect":dynamic*inc,"ix_dynamic_violations":dynamic*(lane+follow),
           "ix_bad_incorrect":bad*inc,"ix_bad_violations":bad*(lane+follow)}
        rows.append(r)
    return rows,censored

def load(horizon):
    rows=[]; censored=0;sessions=0;codes={}
    for p in sorted((ROOT/"input"/"events").glob("sub-*/ses-*/eeg/*_task-Drive_events.tsv")):
        m=re.search(r"sub-(\d+)/ses-([A-Z]+)",str(p))
        if m is None:continue
        subject=int(m.group(1));session=m.group(2)
        if session=="SCPB":continue
        d=pd.read_csv(p,sep="\t",dtype={"value":str})
        d["onset"]=pd.to_numeric(d.onset,errors="raise")
        rr,cc=instances(d,subject,session,horizon)
        rows+=rr;censored+=cc;sessions+=1
        for k,v in d.value.value_counts().items():codes[k]=codes.get(k,0)+int(v)
    df=pd.DataFrame(rows)
    if df.empty:raise RuntimeError("No hazard inputs")
    audit={"participants":int(df.subject.nunique()),"sessions":sessions,"instances":len(df),
           "positives":int(df.y.sum()),"prevalence":float(df.y.mean()),
           "horizon_seconds":horizon,"censored_onsets":censored,
           "lead_median":float(df.loc[df.y==1,"lead_seconds"].median()),
           "lead_q25":float(df.loc[df.y==1,"lead_seconds"].quantile(.25)),
           "lead_q75":float(df.loc[df.y==1,"lead_seconds"].quantile(.75)),
           "direct_collision_markers":sum(codes.get(k,0) for k in COLLISION)}
    return df,audit

def split_inner(df,indices,fold):
    sub=df.iloc[indices]; y=sub.y.to_numpy(); groups=sub.subject.to_numpy()
    for attempt in range(100):
        seed=42600+100*fold+attempt
        fitcal,th=next(GroupShuffleSplit(n_splits=1,test_size=3,random_state=seed).split(sub,y,groups))
        inner=sub.iloc[fitcal]
        fit,cal=next(GroupShuffleSplit(n_splits=1,test_size=3,random_state=seed+10000).split(
            inner,inner.y.to_numpy(),inner.subject.to_numpy()))
        parts=[indices[fitcal[fit]],indices[fitcal[cal]],indices[th]]
        if all(df.iloc[p].y.sum()>=3 and (1-df.iloc[p].y).sum()>=50 for p in parts):
            return parts,seed
    raise RuntimeError("No viable participant-disjoint inner split")

def estimator(name):
    if name=="HGB":
        est=HistGradientBoostingClassifier(max_iter=120,max_leaf_nodes=7,
            l2_regularization=2.,learning_rate=.07,early_stopping=False,random_state=41)
    elif name=="MLP":
        est=MLPClassifier(hidden_layer_sizes=(32,16),alpha=2.,batch_size=128,
            max_iter=180,early_stopping=False,random_state=41,n_iter_no_change=20)
    else:est=LogisticRegression(C=1.,class_weight="balanced",max_iter=3000,random_state=41)
    return make_pipeline(StandardScaler(),est)

def grouped_predictions(df,name):
    cols=FEATURES[name]; y=df.y.to_numpy(int);groups=df.subject.to_numpy()
    probabilities=np.empty(len(df)); flags=np.empty(len(df),int); folds=np.empty(len(df),int)
    records=[]
    for f,(tr,te) in enumerate(GroupKFold(n_splits=6).split(df,y,groups),1):
        (fit,cal,th),seed=split_inner(df,tr,f)
        est=estimator(name).fit(df.iloc[fit][cols],y[fit])
        coeff=sigmoid_fit(score(est,df.iloc[cal][cols]),y[cal])
        pp=expit(coeff[0]*score(est,df.iloc[th][cols])+coeff[1])
        # Fixed false-alert budget chosen exclusively from inner threshold participants.
        threshold=float(np.quantile(pp[y[th]==0],.95,method="higher"))
        p=expit(coeff[0]*score(est,df.iloc[te][cols])+coeff[1])
        probabilities[te]=p;flags[te]=p>=threshold;folds[te]=f
        records.append({"fold":f,"model":name,"threshold":threshold,"split_seed":seed,
            "fit_subjects":sorted(map(int,np.unique(groups[fit]))),
            "calibration_subjects":sorted(map(int,np.unique(groups[cal]))),
            "threshold_subjects":sorted(map(int,np.unique(groups[th]))),
            "test_subjects":sorted(map(int,np.unique(groups[te]))),
            "fit_positives":int(y[fit].sum()),"calibration_positives":int(y[cal].sum()),
            "threshold_positives":int(y[th].sum()),"test_positives":int(y[te].sum()),
            "calibration_slope":float(coeff[0]),"calibration_intercept":float(coeff[1])})
    return probabilities,flags,folds,records

def pooled_metrics(y,p,flags):
    r=metrics(y,p,.5)
    from sklearn.metrics import confusion_matrix
    tn,fp,fn,tp=confusion_matrix(y,flags,labels=[0,1]).ravel()
    recall=tp/max(1,tp+fn);precision=tp/max(1,tp+fp)
    r.update(Sensitivity=recall,Precision=precision,FPR=fp/max(1,tn+fp),
             F1=2*recall*precision/max(1e-12,recall+precision),
             TP=int(tp),FP=int(fp),FN=int(fn),TN=int(tn))
    return r

def cluster_bootstrap(df,predictions,B=2000):
    from sklearn.metrics import roc_auc_score,average_precision_score
    subjects=np.unique(df.subject);indices={s:np.flatnonzero(df.subject.to_numpy()==s) for s in subjects}
    y=df.y.to_numpy(int);rng=np.random.default_rng(517)
    rows=[]
    for b in range(B):
        sampled=rng.choice(subjects,len(subjects),replace=True)
        idx=np.concatenate([indices[s] for s in sampled]); yy=y[idx]
        if len(np.unique(yy))<2:continue
        for name,p in predictions.items():
            rows.append({"replicate":b,"model":name,"AUROC":roc_auc_score(yy,p[idx]),
                         "AP":average_precision_score(yy,p[idx])})
    return pd.DataFrame(rows)

def main():
    out=ROOT/"output"/"events";out.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter(); results=[];summary=[];allfold=[]
    for horizon in [8.,4.,12.]:
        df,audit=load(horizon);summary.append(audit)
        df.to_csv(ROOT/"input"/("event_instances_h"+str(int(horizon))+".csv"),index=False)
        names=list(FEATURES) if horizon==8 else ["Additive","Interaction"]
        predictions={};oof=df.copy();flags={}
        for name in names:
            p,phat,fold,records=grouped_predictions(df,name)
            predictions[name]=p;flags[name]=phat
            oof["p_"+name.replace(" ","_")]=p
            oof["flag_"+name.replace(" ","_")]=phat
            oof["fold"]=fold
            results.append({"horizon_seconds":horizon,"model":name,**pooled_metrics(df.y.to_numpy(),p,phat)})
            for r in records:r["horizon_seconds"]=horizon
            allfold+=records
            print("empirical fitted",horizon,name,flush=True)
        oof.to_csv(out/("oof_h"+str(int(horizon))+".csv"),index=False)
        if horizon==8:
            boot=cluster_bootstrap(df,predictions)
            boot.to_csv(out/"participant_bootstrap.csv",index=False)
            ci=[]
            for name,g in boot.groupby("model"):
                r={"model":name}
                for metric in ["AUROC","AP"]:
                    r[metric+"_low"],r[metric+"_high"]=np.quantile(g[metric],[.025,.975])
                ci.append(r)
            pd.DataFrame(ci).to_csv(out/"confidence_intervals.csv",index=False)
            paired=boot.pivot(index="replicate",columns="model",values="AP")
            differences=[]
            for name in ["Additive","Context only","HGB","MLP"]:
                d=paired["Interaction"]-paired[name]
                differences.append({"comparator":name,
                    "AP_delta":next(r["AP"] for r in results if r["horizon_seconds"]==8 and r["model"]=="Interaction")-
                        next(r["AP"] for r in results if r["horizon_seconds"]==8 and r["model"]==name),
                    "low":float(d.quantile(.025)),"high":float(d.quantile(.975))})
            pd.DataFrame(differences).to_csv(out/"paired_AP_intervals.csv",index=False)
            participant=[]
            for s,g in df.groupby("subject"):
                for name,p in predictions.items():
                    yy=g.y.to_numpy();pp=p[g.index]
                    participant.append({"subject":int(s),"model":name,"n":len(g),"positives":int(yy.sum()),
                        "AP":float(__import__("sklearn.metrics",fromlist=["average_precision_score"]).average_precision_score(yy,pp)) if yy.sum() else np.nan})
            pd.DataFrame(participant).to_csv(out/"participant_metrics.csv",index=False)
    pd.DataFrame(results).to_csv(out/"model_comparison.csv",index=False)
    (out/"dataset_summary.json").write_text(json.dumps(summary,indent=2))
    (out/"participant_splits.json").write_text(json.dumps(allfold,indent=2))
    (out/"run_manifest.json").write_text(json.dumps({"python":sys.version,"platform":platform.platform(),
        "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "elapsed_seconds":time.perf_counter()-start,"scope":"secondary analysis of public simulator event logs; no EEG or video"},indent=2))
    print("event study complete",flush=True)
if __name__=="__main__":main()
