"""Causal scalar-stream fusion. Scores are synthetic or externally supplied."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.special import expit, logit
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score,
    brier_score_loss, confusion_matrix, log_loss)

BASELINES = ["E-only","H-only","T-only","H+T","Additive","Static interaction",
             "Quality static","Quality temporal","HGB quality","HGB full","MLP full"]
ABLATIONS = ["No quality","No temporal","No interactions","No driver","No telemetry"]
MODELS = BASELINES + ["UATIF"] + ABLATIONS

def normalized(x,u,a):
    x=np.asarray(x,float); u=np.asarray(u,float); a=np.asarray(a,float)
    if x.shape != u.shape or x.shape != a.shape or x.shape[-1] != 3:
        raise ValueError("x, uncertainty and availability must have equal (...,3) shape")
    if not (np.isfinite(x).all() and np.isfinite(u).all() and np.isfinite(a).all()):
        raise ValueError("nonfinite input")
    if np.any((x<0)|(x>1)|(u<0)|(u>1)|((a!=0)&(a!=1))):
        raise ValueError("scores and quality must be in [0,1]; availability is binary")
    return np.where(a>0,x,0.),np.where(a>0,u,1.),a

def feature_cube(x,u,a,variant="UATIF",alpha=.35):
    x,u,a=normalized(x,u,a)
    if x.ndim!=3: raise ValueError("Expected (sequence,time,modality)")
    gated=a*(1-u)*x
    smooth=np.zeros_like(x); state=np.zeros((len(x),3)); previous=np.zeros_like(state)
    delta=np.zeros(x.shape[:2]+(1,))
    source=x if variant=="No quality" else gated
    for t in range(x.shape[1]):
        previous=state.copy()
        state=np.where(a[:,t]>0,alpha*source[:,t]+(1-alpha)*state,state)
        smooth[:,t]=state
        delta[:,t,0]=np.maximum(0,state[:,1]-previous[:,1])
    if variant=="E-only": return x[:,:,[0]]
    if variant=="H-only": return x[:,:,[1]]
    if variant=="T-only": return x[:,:,[2]]
    if variant=="H+T": return x[:,:,[1,2]]
    if variant=="Additive": return x
    static=np.stack([x[:,:,0]*x[:,:,1],x[:,:,1]*x[:,:,2]],axis=-1)
    if variant=="Static interaction": return np.concatenate([x,static],axis=-1)
    quality=np.concatenate([x,u,a],axis=-1)
    if variant in ["Quality static","HGB quality"]: return quality
    if variant=="Quality temporal": return np.concatenate([quality,gated,smooth,delta],axis=-1)
    if variant=="No temporal":
        ix=np.stack([gated[:,:,0]*gated[:,:,1],gated[:,:,1]*gated[:,:,2]],axis=-1)
        return np.concatenate([quality,gated,ix],axis=-1)
    ix=np.stack([smooth[:,:,0]*smooth[:,:,1],smooth[:,:,1]*smooth[:,:,2]],axis=-1)
    if variant=="No quality": return np.concatenate([x,smooth,delta,ix],axis=-1)
    if variant=="No interactions": return np.concatenate([quality,gated,smooth,delta],axis=-1)
    if variant in ["No driver","No telemetry"]:
        keep=[1,2] if variant=="No driver" else [0,1]
        arrays=[v[:,:,keep] for v in [x,u,a,gated,smooth]]
        product=smooth[:,:,keep[0]]*smooth[:,:,keep[1]]
        return np.concatenate(arrays+[delta,product[:,:,None]],axis=-1)
    if variant not in ["UATIF","HGB full","MLP full"]:
        raise ValueError(variant)
    return np.concatenate([quality,gated,smooth,delta,ix],axis=-1)

def score(estimator,X):
    if hasattr(estimator,"decision_function"): return estimator.decision_function(X)
    return logit(np.clip(estimator.predict_proba(X)[:,1],1e-7,1-1e-7))

def sigmoid_fit(scores,y):
    scores=np.asarray(scores); y=np.asarray(y)
    def loss(theta):
        p=np.clip(expit(theta[0]*scores+theta[1]),1e-9,1-1e-9)
        return -np.mean(y*np.log(p)+(1-y)*np.log1p(-p))
    opt=minimize(loss,[1.,0.],method="L-BFGS-B",bounds=[(1e-4,20),(-20,20)])
    if not opt.success: raise RuntimeError("sigmoid calibration failed: "+opt.message)
    return opt.x

def threshold_fit(y,p,min_recall=.75):
    best=None
    candidates=np.unique(np.r_[0,np.quantile(p,np.linspace(0,1,501)),1])
    for threshold in candidates:
        pred=p>=threshold; tp=np.sum(pred&(y==1)); fp=np.sum(pred&(y==0))
        recall=tp/max(1,np.sum(y==1)); precision=tp/max(1,tp+fp)
        f1=2*precision*recall/max(1e-12,precision+recall)
        if recall>=min_recall:
            candidate=(f1,float(threshold))
            if best is None or candidate>best: best=candidate
    return best[1]

def metrics(y,p,threshold):
    y=np.asarray(y,int); p=np.asarray(p); pred=p>=threshold
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    recall=tp/max(1,tp+fn); precision=tp/max(1,tp+fp)
    edges=np.linspace(0,1,11); ece=0.
    for i in range(10):
        mask=(p>=edges[i]) & ((p<edges[i+1]) if i<9 else (p<=edges[i+1]))
        if mask.any(): ece+=mask.mean()*abs(y[mask].mean()-p[mask].mean())
    return {"AUROC":roc_auc_score(y,p),"AP":average_precision_score(y,p),
            "Brier":brier_score_loss(y,p),"ECE":ece,
            "Sensitivity":recall,"Precision":precision,"FPR":fp/max(1,tn+fp),
            "F1":2*recall*precision/max(1e-12,recall+precision),
            "LogLoss":log_loss(y,p,labels=[0,1])}

class Fitted:
    def __init__(self,name):
        self.name=name; self.scaler=StandardScaler()
        if name.startswith("HGB"):
            self.estimator=HistGradientBoostingClassifier(max_iter=180,
                max_leaf_nodes=15,l2_regularization=1.,learning_rate=.07,
                early_stopping=False,random_state=41)
        elif name.startswith("MLP"):
            self.estimator=MLPClassifier(hidden_layer_sizes=(32,16),activation="relu",
                alpha=1.,max_iter=120,batch_size=512,learning_rate_init=.001,
                early_stopping=False,tol=1e-4,n_iter_no_change=15,random_state=41)
        else:
            self.estimator=LogisticRegression(C=1.,max_iter=2000,random_state=41)
    def fit(self,train,y,cal,ycal,threshold_data,ythreshold):
        train=self.scaler.fit_transform(train)
        self.estimator.fit(train,y)
        self.cal=sigmoid_fit(score(self.estimator,self.scaler.transform(cal)),ycal)
        p=self.predict(threshold_data)
        self.threshold=threshold_fit(ythreshold,p)
        return self
    def predict(self,X):
        s=score(self.estimator,self.scaler.transform(X))
        return expit(self.cal[0]*s+self.cal[1])
    def export(self,path):
        if not isinstance(self.estimator,LogisticRegression): return
        obj={"variant":self.name,"alpha":.35,
             "mean":self.scaler.mean_.tolist(),"scale":self.scaler.scale_.tolist(),
             "coefficient":self.estimator.coef_[0].tolist(),
             "intercept":float(self.estimator.intercept_[0]),
             "calibration":self.cal.tolist(),"threshold":self.threshold}
        Path(path).write_text(json.dumps(obj,indent=2))

class Streaming:
    def __init__(self,model):
        self.model=json.loads(Path(model).read_text()) if isinstance(model,(str,Path)) else model
        if self.model["variant"]!="UATIF": raise ValueError("Only UATIF is portable here")
        self.reset()
    def reset(self): self.state=np.zeros(3)
    def step(self,x,u,a):
        x,u,a=normalized(x,u,a)
        if x.shape!=(3,): raise ValueError("one three-modality frame is required")
        g=a*(1-u)*x; old=self.state.copy(); alpha=self.model["alpha"]
        self.state=np.where(a>0,alpha*g+(1-alpha)*self.state,self.state)
        delta=max(0,self.state[1]-old[1])
        z=np.r_[x,u,a,g,self.state,delta,
                self.state[0]*self.state[1],self.state[1]*self.state[2]]
        s=((z-np.array(self.model["mean"]))/np.array(self.model["scale"]))@np.array(self.model["coefficient"])+self.model["intercept"]
        p=float(expit(self.model["calibration"][0]*s+self.model["calibration"][1]))
        missing=bool(np.all(a==0))
        return {"probability":p,"threshold_flag":bool(p>=self.model["threshold"]),
                "all_missing":missing,"status":"unavailable" if missing else "observed"}
