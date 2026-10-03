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
def draw_framework():
    """Draw the specified interfaces and the evaluated scalar decision flow."""
    with plt.rc_context({"svg.fonttype": "none"}):
        fig, ax = plt.subplots(figsize=(7.05, 4.18))
        ax.set_xlim(0, 10.4)
        ax.set_ylim(0, 6.25)
        ax.axis("off")
        ink = "#263a4a"
        muted = "#717b84"
        blue = "#eaf0f5"

        def node(x, y, width, height, label, specified=False, size=9.2):
            ax.add_patch(Rectangle((x, y), width, height,
                facecolor="white" if specified else blue,
                edgecolor=muted if specified else ink, linewidth=0.9,
                linestyle=(0, (4, 3)) if specified else "-"))
            ax.text(x + width / 2, y + height / 2, label,
                ha="center", va="center", fontsize=size,
                fontfamily="DejaVu Sans", color=ink, linespacing=1.18)

        def link(start, end, specified=False):
            ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>",
                mutation_scale=9, linewidth=0.9,
                color=muted if specified else ink,
                linestyle=(0, (3, 2)) if specified else "-",
                shrinkA=0, shrinkB=0))

        ax.add_patch(Rectangle((0.15, 4.74), 10.1, 1.30, fill=False,
            edgecolor=muted, linewidth=0.75, linestyle=(0, (5, 4))))
        ax.text(0.35, 5.76, "Specified perception interfaces", fontsize=9.2,
            fontfamily="DejaVu Sans", fontweight="bold", color=ink)
        starts = [0.40, 3.825, 7.25]
        labels = ["Driver camera\nEmotion / state\nnetwork",
                  "Road camera\nVehicle detection\nand tracking",
                  "Vehicle signals\nKinematic estimation"]
        scores = ["Driver score E\nUncertainty u\nAvailability a",
                  "Hazard score H\nUncertainty u\nAvailability a",
                  "Telemetry score T\nUncertainty u\nAvailability a"]
        for x, label, score_label in zip(starts, labels, scores):
            node(x, 4.84, 2.75, 0.78, label, specified=True, size=8.0)
            node(x, 3.80, 2.75, 0.80, score_label, size=8.0)
            link((x + 1.375, 4.83), (x + 1.375, 4.61), specified=True)
            link((x + 1.375, 3.79), (x + 1.375, 3.58))

        node(1.20, 2.99, 8.00, 0.58,
             "Normalize inputs + apply quality gates", size=9.5)
        node(1.20, 2.14, 8.00, 0.58,
             "Causal memory + contextual products", size=9.5)
        node(1.20, 1.29, 8.00, 0.58,
             "Calibrated scalar fusion\nUATIF / matched MLP or HGB", size=8.2)
        node(1.20, 0.44, 8.00, 0.58,
             "Probability + threshold flag + availability status", size=9.2)
        for y_top, y_bottom in [(2.98, 2.73), (2.13, 1.88), (1.28, 1.03)]:
            link((5.20, y_top), (5.20, y_bottom))
        save(fig, "Figure_1_Framework")

draw_framework()


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

