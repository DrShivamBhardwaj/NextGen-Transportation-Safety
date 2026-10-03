"""Executed stochastic experiment with current and one-second-ahead targets."""
from __future__ import annotations
import json,hashlib,platform,sys,time,warnings
from pathlib import Path
import numpy as np
import pandas as pd
import scipy,sklearn
from scipy.special import expit
from scipy.stats import wilcoxon,rankdata
from fusion import MODELS,BASELINES,Fitted,feature_cube,metrics
ROOT=Path(__file__).resolve().parents[1]

def generate(seed,config):
    rng=np.random.default_rng(seed)
    N=config["sequences"]; L=config["frames"]+config["forecast_frames"]
    z=np.empty((N,L,3)); z[:,0]=rng.normal(0,1,(N,3))
    rho=np.array(config["rho"]); sd=np.sqrt(1-rho**2)
    for t in range(1,L):
        eps=rng.normal(0,1,(N,3))
        eps[:,2]=.5*eps[:,1]+np.sqrt(.75)*eps[:,2]
        z[:,t]=rho*z[:,t-1]+sd*eps
    latent=expit(z)
    trend=np.concatenate([np.zeros((N,1)),np.maximum(0,np.diff(latent[:,:,1],axis=1))],axis=1)
    E,H,T=latent[:,:,0],latent[:,:,1],latent[:,:,2]
    p=expit(-4.6+3.5*H+2.6*T+2.4*E*H+1.2*H*T+.5*trend)
    y=(rng.random(p.shape)<p).astype(np.uint8)
    u=rng.uniform(.05,.4,(N,L,3))
    outlier=rng.random(u.shape)<.04
    u=np.where(outlier,.8,u)
    noise=rng.normal(0,1,latent.shape)*(.03+.30*u)
    x=np.clip(latent+noise,0,1)
    x=np.where(outlier,rng.random(x.shape),x)
    a=(rng.random(x.shape)>=.02).astype(np.uint8)
    return {"x":x.astype(np.float32),"u":u.astype(np.float32),"a":a,
            "y":y,"latent":latent.astype(np.float32),"event_probability":p.astype(np.float32)}

def perturb(data,seed,scenario):
    d={k:v.copy() for k,v in data.items()}
    rng=np.random.default_rng(seed+200000)
    rate={"iid10":.1,"iid20":.2,"iid30":.3}.get(scenario,0)
    if rate: d["a"]*=rng.random(d["a"].shape)>=rate
    if scenario=="burst30":
        N,L,M=d["a"].shape; width=int(round(.3*L))
        for n in range(N):
            for m in range(M):
                start=int(rng.integers(0,L-width+1)); d["a"][n,start:start+width,m]=0
    if scenario=="noise":
        d["x"]=np.clip(d["x"]+rng.normal(0,.18,d["x"].shape),0,1).astype(np.float32)
        d["u"]=np.minimum(1,d["u"]+.35).astype(np.float32)
    if scenario=="quality_bias":
        d["u"]=np.maximum(0,d["u"]-.35).astype(np.float32)
    if scenario=="driver_shuffle":
        permutation=rng.permutation(len(d["x"]))
        for key in ["x","u","a"]: d[key][:,:,0]=d[key][permutation,:,0]
    return d

def pack(seeds,config,scenario="clean"):
    arrays=[perturb(generate(s,config),s,scenario) for s in seeds]
    return {k:np.stack([d[k] for d in arrays]) for k in arrays[0]} | {"seeds":np.array(seeds)}

def design(data,variant,config):
    n,s,l,m=data["x"].shape; frames=config["frames"]
    x=data["x"][:,:,:frames].reshape(n*s,frames,m)
    u=data["u"][:,:,:frames].reshape(n*s,frames,m)
    a=data["a"][:,:,:frames].reshape(n*s,frames,m)
    f=feature_cube(x,u,a,variant,alpha=config["alpha"])
    return f.reshape(-1,f.shape[-1])

def target(data,horizon,config):
    return data["y"][:,:,horizon:horizon+config["frames"]].reshape(-1)

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def bootstrap(values,B=2000,seed=531):
    values=np.asarray(values); rng=np.random.default_rng(seed)
    means=values[rng.integers(0,len(values),(B,len(values)))].mean(axis=1)
    return np.quantile(means,[.025,.975])

def holm(p):
    p=np.asarray(p); order=np.argsort(p); adjusted=np.zeros(len(p)); v=0.
    for j,i in enumerate(order):
        v=max(v,(len(p)-j)*p[i]); adjusted[i]=min(1,v)
    return adjusted

def main():
    root=ROOT; inp=root/"input"/"synthetic"; out=root/"output"/"synthetic"
    inp.mkdir(parents=True,exist_ok=True); out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((root/"input"/"config.json").read_text())
    source={str(p.relative_to(root)):sha(p) for p in sorted((root/"code").rglob("*.py"))}
    frozen={"configuration":cfg,"source_sha256":source,
            "scope":"versioned experiment; no retrospective preregistration claim"}
    (out/"frozen_protocol.json").write_text(json.dumps(frozen,indent=2))
    train=pack(cfg["train_seeds"],cfg); cal=pack(cfg["calibration_seeds"],cfg)
    threshold=pack(cfg["threshold_seeds"],cfg)
    for name,d in [("train",train),("calibration",cal),("threshold",threshold)]:
        np.savez_compressed(inp/(name+".npz"),**d)
    start=time.perf_counter(); rows=[]; pars=[]; fitted={}; predictions={}
    for horizon in [0,cfg["forecast_frames"]]:
        for name in MODELS:
            begin=time.perf_counter()
            model=Fitted(name).fit(design(train,name,cfg),target(train,horizon,cfg),
                 design(cal,name,cfg),target(cal,horizon,cfg),
                 design(threshold,name,cfg),target(threshold,horizon,cfg))
            fitted[(horizon,name)]=model
            pars.append({"horizon_frames":horizon,"model":name,
                "features":model.scaler.n_features_in_,"threshold":model.threshold,
                "sigmoid_slope":model.cal[0],"sigmoid_intercept":model.cal[1],
                "fit_seconds":time.perf_counter()-begin,
                "iterations":int(np.ravel(getattr(model.estimator,"n_iter_",0))[0])})
            if name=="UATIF": model.export(out/("uatif_h"+str(horizon)+".json"))
            import joblib
            model_dir=out/"models"; model_dir.mkdir(exist_ok=True)
            joblib.dump(model,model_dir/(name.replace(" ","_").replace("+","_")+"_h"+str(horizon)+".joblib"))
            print("fitted",horizon,name,flush=True)
    for scenario in cfg["scenarios"]:
        full_predictions={}
        data=pack(cfg["evaluation_seeds"],cfg,scenario)
        np.savez_compressed(inp/("evaluation_"+scenario+".npz"),**data)
        for horizon in [0,cfg["forecast_frames"]]:
            y=target(data,horizon,cfg)
            for name in MODELS:
                p=fitted[(horizon,name)].predict(design(data,name,cfg))
                if scenario=="clean": full_predictions[str(horizon)+"_"+name]=p.astype(np.float32)
                size=cfg["sequences"]*cfg["frames"]
                for i,seed in enumerate(cfg["evaluation_seeds"]):
                    sl=slice(i*size,(i+1)*size)
                    row={"scenario":scenario,"horizon_frames":horizon,"seed":seed,
                         "model":name,"prevalence":float(y[sl].mean()),
                         **metrics(y[sl],p[sl],fitted[(horizon,name)].threshold)}
                    rows.append(row)
                if name in ["UATIF","HGB full","MLP full"] and scenario=="clean":
                    for i in range(3):
                        sl=slice(i*cfg["frames"],(i+1)*cfg["frames"])
                        predictions[(horizon,name,i)]=(y[sl],p[sl])
        if scenario=="clean":
            full_predictions["y_h0"]=target(data,0,cfg)
            full_predictions["y_h10"]=target(data,cfg["forecast_frames"],cfg)
            np.savez_compressed(out/"clean_predictions.npz",**full_predictions)
        print("evaluated",scenario,flush=True)
    raw=pd.DataFrame(rows); raw.to_csv(out/"raw_seed_metrics.csv",index=False)
    summaries=[]
    fields=["AUROC","AP","Brier","ECE","Sensitivity","Precision","FPR","F1","prevalence"]
    for keys,g in raw.groupby(["horizon_frames","scenario","model"],sort=False):
        r=dict(zip(["horizon_frames","scenario","model"],keys))
        for metric in fields:
            lo,hi=bootstrap(g[metric].to_numpy())
            r[metric]=g[metric].mean();r[metric+"_low"]=lo;r[metric+"_high"]=hi
        summaries.append(r)
    pd.DataFrame(summaries).to_csv(out/"summary.csv",index=False)
    primary=raw[(raw.scenario=="clean")&(raw.horizon_frames==0)]
    ref=primary[primary.model=="UATIF"].set_index("seed").AP
    tests=[]
    for name in BASELINES:
        comparator=primary[primary.model==name].set_index("seed").AP
        d=(ref-comparator).to_numpy(); nonzero=d[d!=0]; ranks=rankdata(abs(nonzero))
        rbc=float(np.sum(np.sign(nonzero)*ranks)/sum(ranks)) if len(ranks) else 0.
        lo,hi=bootstrap(d)
        pvalue=wilcoxon(d,alternative="two-sided").pvalue if np.any(d) else 1.
        tests.append({"comparator":name,"mean_delta_AP":d.mean(),"delta_low":lo,
                     "delta_high":hi,"median_delta_AP":np.median(d),
                     "p_two_sided":pvalue,"rank_biserial":rbc})
    adjusted=holm([v["p_two_sided"] for v in tests])
    for r,p in zip(tests,adjusted): r["p_holm"]=p
    pd.DataFrame(tests).to_csv(out/"paired_primary_tests.csv",index=False)
    contrasts=[]
    for (h,scenario),g in raw.groupby(["horizon_frames","scenario"]):
        ref=g[g.model=="UATIF"].set_index("seed").AP
        for name in ["Additive","Quality temporal","HGB full","MLP full"]:
            d=(ref-g[g.model==name].set_index("seed").AP).to_numpy()
            lo,hi=bootstrap(d)
            contrasts.append({"horizon_frames":h,"scenario":scenario,"comparator":name,
                              "mean_delta_AP":d.mean(),"low":lo,"high":hi})
    pd.DataFrame(contrasts).to_csv(out/"paired_stress_intervals.csv",index=False)
    pd.DataFrame(pars).to_csv(out/"model_parameters.csv",index=False)
    ex=[]
    for (h,name,i),(y,p) in predictions.items():
        for t,(yy,pp) in enumerate(zip(y,p)):
            ex.append({"horizon_frames":h,"model":name,"sequence":i,"frame":t,
                       "label":int(yy),"probability":float(pp)})
    pd.DataFrame(ex).to_csv(out/"example_predictions.csv",index=False)
    from fusion import Streaming
    stream=Streaming(out/"uatif_h0.json"); d=generate(cfg["evaluation_seeds"][0],cfg)
    samples=[]
    for repeat in range(20):
        stream.reset()
        for t in range(cfg["frames"]):
            begin=time.perf_counter_ns(); stream.step(d["x"][0,t],d["u"][0,t],d["a"][0,t])
            samples.append((time.perf_counter_ns()-begin)/1000)
    timing={"median_us":float(np.median(samples)),"p95_us":float(np.quantile(samples,.95)),
            "samples":len(samples),"scope":"stream normalization, causal features and portable logistic inference; no cameras, detector or I/O"}
    (out/"runtime.json").write_text(json.dumps(timing,indent=2))
    manifest={"python":sys.version,"platform":platform.platform(),"machine":platform.machine(),
        "numpy":np.__version__,"pandas":pd.__version__,"scipy":scipy.__version__,
        "scikit_learn":sklearn.__version__,"elapsed_seconds":time.perf_counter()-start,
        "seed_metrics":len(raw),"input_sha256":{str(p.relative_to(root)):sha(p) for p in inp.glob("*.npz")},
        "source_sha256":source,"fit_count":len(fitted)}
    (out/"run_manifest.json").write_text(json.dumps(manifest,indent=2))
    print("simulation complete",len(raw),"seed metrics",flush=True)

if __name__=="__main__": main()
