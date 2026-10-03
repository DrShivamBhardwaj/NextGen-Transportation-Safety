"""Regenerate every manuscript figure solely from archived executed outputs."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch,Rectangle
from sklearn.metrics import precision_recall_curve
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"output"/"figures";OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({"font.family":"serif","font.serif":["Times New Roman","DejaVu Serif"],
    "font.size":10,"axes.labelweight":"bold","axes.spines.top":False,
    "axes.spines.right":False,"axes.grid":False,"savefig.dpi":600})
COLORS={"Additive":"#7a7a7a","Quality temporal":"#b87333","HGB full":"#3873ac",
        "MLP full":"#8a589e","UATIF":"#153c63","Interaction":"#153c63","HGB":"#3873ac","MLP":"#8a589e"}
def save(fig,name):
    fig.savefig(OUT/(name+".png"),dpi=600,bbox_inches="tight",facecolor="white")
    fig.savefig(OUT/(name+".svg"),bbox_inches="tight",facecolor="white")
    fig.savefig(OUT/(name+".pdf"),bbox_inches="tight",facecolor="white")
    plt.close(fig)
def box(ax,x,y,w,h,label,fill="#f4f6f8",edge="#526170",dashed=False):
    p=FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.015,rounding_size=0.025",
        fc=fill,ec=edge,lw=1.1,linestyle="--" if dashed else "-")
    ax.add_patch(p);ax.text(x+w/2,y+h/2,label,ha="center",va="center",fontsize=9,fontfamily="sans-serif")
def arrow(ax,start,end):
    ax.add_patch(FancyArrowPatch(start,end,arrowstyle="-|>",mutation_scale=10,lw=1,color="#425260"))
fig,ax=plt.subplots(figsize=(7.05,4.65));ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis("off")
ax.add_patch(Rectangle((.015,.59),.97,.39,fill=False,ls="--",ec="#909ba6",lw=1))
ax.text(.035,.952,"Perception interfaces specified; raw-camera models are outside this evaluation",
        ha="left",va="center",fontsize=9,fontfamily="sans-serif")
xs=[.04,.355,.67]
for x,l1,l2,l3 in zip(xs,["Driver camera","Road camera","Vehicle signals"],
 ["Emotion and driver-state\nnetwork","Vehicle detection\nand tracking","Kinematic estimation"],
 ["Driver score\nand quality","Hazard score\nand quality","Telemetry score\nand quality"]):
    box(ax,x,.805,.29,.082,l1,dashed=True)
    box(ax,x,.687,.29,.087,l2,dashed=True)
    arrow(ax,(x+.145,.802),(x+.145,.776))
    box(ax,x,.545,.29,.09,l3,fill="#e7eef5")
    arrow(ax,(x+.145,.683),(x+.145,.637))
ax.text(.035,.495,"Executed causal fusion and prospective evaluation",
        ha="left",va="center",fontsize=9,fontfamily="sans-serif",fontweight="bold")
box(ax,.05,.33,.28,.11,"Availability and quality\nnormalization",fill="#e7eef5")
box(ax,.36,.33,.28,.11,"Causal memory and\ncross-modal interactions",fill="#e7eef5")
box(ax,.67,.33,.28,.11,"Calibrated logistic fusion\nor matched neural fusion",fill="#e7eef5")
for x in xs:
    ax.plot([x+.145,x+.145],[.541,.465],color="#425260",lw=1)
ax.plot([.19,.815],[.465,.465],color="#425260",lw=1)
arrow(ax,(.19,.465),(.19,.443))
arrow(ax,(.335,.385),(.355,.385));arrow(ax,(.645,.385),(.665,.385))
box(ax,.36,.16,.28,.11,"Probability, numerical flag\nand availability status",fill="#e7eef5")
arrow(ax,(.81,.326),(.81,.213));arrow(ax,(.81,.213),(.645,.213))
ax.text(.5,.075,"Evidence: synthetic score sequences + participant-disjoint simulator event logs",
        ha="center",va="center",fontsize=9,fontfamily="sans-serif")
save(fig,"Figure_1_Framework")

s=pd.read_csv(ROOT/"output/synthetic/summary.csv")
clean=s[(s.scenario=="clean")&(s.horizon_frames==0)].set_index("model")
names=["Additive","HGB quality","Quality temporal","HGB full","UATIF","MLP full"]
fig,ax=plt.subplots(figsize=(3.5,3.15))
for i,name in enumerate(names):
    r=clean.loc[name];col=COLORS.get(name,"#3873ac")
    ax.errorbar(r.AP,i,xerr=[[r.AP-r.AP_low],[r.AP_high-r.AP]],fmt="o",color=col,capsize=3,ms=5)
ax.set_yticks(range(len(names)),names);ax.set_xlabel("Average precision")
ax.set_xlim(.675,.745);ax.grid(axis="x",ls="--",alpha=.25);fig.tight_layout()
save(fig,"Figure_2_Clean_AP")

fig,ax=plt.subplots(figsize=(3.5,3.1))
for name,marker in [("Additive","s"),("HGB full","^"),("MLP full","D"),("UATIF","o")]:
    q=s[(s.horizon_frames==0)&(s.model==name)&s.scenario.isin(["clean","iid10","iid20","iid30"])]
    q=q.set_index("scenario").loc[["clean","iid10","iid20","iid30"]]
    ax.plot([0,10,20,30],q.AP,marker=marker,color=COLORS[name],lw=1.6,ms=4,label=name)
    if name=="UATIF":ax.fill_between([0,10,20,30],q.AP_low,q.AP_high,color=COLORS[name],alpha=.1)
ax.set_ylim(.53,.77);ax.set_xticks([0,10,20,30]);ax.set_xlabel("Additional independent dropout (%)")
ax.set_ylabel("Average precision");ax.grid(axis="y",ls="--",alpha=.25)
ax.legend(ncol=2,loc="upper center",bbox_to_anchor=(.5,1.22),frameon=False,fontsize=8,columnspacing=1)
fig.tight_layout();save(fig,"Figure_4_Dropout")

fig,axs=plt.subplots(1,2,figsize=(7.05,2.85))
cases=["clean","burst30","noise","quality_bias"];labels=["Clean","Burst loss","Extra noise","Quality bias"]
for ax,metric,title in zip(axs,["AP","ECE"],["(a) Discrimination","(b) Calibration error"]):
    for j,name in enumerate(["Additive","MLP full","UATIF"]):
        q=s[(s.horizon_frames==0)&(s.model==name)].set_index("scenario").loc[cases]
        ax.plot(range(4),q[metric],marker=["s","D","o"][j],lw=1.5,ms=4,color=COLORS[name],label=name)
    ax.set_xticks(range(4),labels,rotation=20,ha="right")
    ax.set_title(title,loc="left",fontweight="bold",fontsize=10)
    ax.set_ylabel("Average precision" if metric=="AP" else "Expected calibration error")
    ax.grid(axis="y",ls="--",alpha=.25)
axs[0].set_ylim(.53,.76);axs[1].set_ylim(0,.21)
handles,labs=axs[0].get_legend_handles_labels()
fig.legend(handles,labs,ncol=3,loc="upper center",bbox_to_anchor=(.5,1.08),frameon=False,fontsize=9)
fig.tight_layout();save(fig,"Figure_3_Failure_Cases")

d=pd.read_csv(ROOT/"output/events/oof_h8.csv");y=d.y.to_numpy()
fig,axs=plt.subplots(1,2,figsize=(7.05,3.05))
for name in ["Additive","Interaction","HGB","MLP"]:
    p=d["p_"+name].to_numpy();prec,rec,_=precision_recall_curve(y,p)
    axs[0].step(rec,prec,where="post",label=name,color=COLORS[name],lw=1.3)
axs[0].axhline(y.mean(),color="#888888",ls=":",lw=1.2)
axs[0].set_xlabel("Recall");axs[0].set_ylabel("Precision")
axs[0].set_xlim(0,1);axs[0].set_ylim(0,.4);axs[0].grid(ls="--",alpha=.2)
axs[0].text(.97,.054,"Prevalence 1.95%",ha="right",fontsize=8,color="#666666")
table=pd.read_csv(ROOT/"output/events/model_comparison.csv");ci=pd.read_csv(ROOT/"output/events/confidence_intervals.csv").set_index("model")
for i,name in enumerate(["Driver history","Context only","Additive","Interaction","HGB","MLP"]):
    r=table[(table.horizon_seconds==8)&(table.model==name)].iloc[0];c=ci.loc[name]
    axs[1].errorbar(r.AP,i,xerr=[[r.AP-c.AP_low],[c.AP_high-r.AP]],fmt="o",
                   color=COLORS.get(name,"#7a7a7a"),capsize=3,ms=4)
axs[1].axvline(y.mean(),color="#888888",ls=":",lw=1.2)
axs[1].set_yticks(range(6),["Driver history","Context only","Additive","Interaction","HGB","MLP"])
axs[1].set_xlabel("Average precision");axs[1].set_xlim(0,.13)
axs[1].grid(axis="x",ls="--",alpha=.2)
axs[0].set_title("(a) Out-of-participant predictions",loc="left",fontsize=10,fontweight="bold")
axs[1].set_title("(b) Participant bootstrap intervals",loc="left",fontsize=10,fontweight="bold")
handles,labs=axs[0].get_legend_handles_labels()
fig.legend(handles,labs,ncol=4,loc="upper center",bbox_to_anchor=(.5,1.09),frameon=False,fontsize=9)
fig.tight_layout();save(fig,"Figure_5_Event_Validation")
print("Five figures generated from executed desktop results.",flush=True)
