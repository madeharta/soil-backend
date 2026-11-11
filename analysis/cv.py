import sys, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import mean_absolute_error as MAE, mean_squared_error as MSE, r2_score as R2
F=["latitude","longitude","temperature","humidity","dew_point_2m","precipitation","rain","weather_code","evapotranspiration","wind_speed","wind_direction"]
W=F[2:]
def run(df,target,feats,split,model="rf"):
    X=df[feats].values; y=df[target].values; g=df.combine_loc.values
    oof=np.zeros(len(y)); fm=[]
    for tr,te in split.split(X,y,g):
        if model=="rf": p=RandomForestRegressor(n_estimators=100,random_state=0,n_jobs=-1).fit(X[tr],y[tr]).predict(X[te])
        elif model=="mean": p=np.full(len(te),y[tr].mean())
        elif model=="locmean":
            mu=pd.Series(y[tr]).groupby(g[tr]).mean(); p=np.array([mu.get(k,y[tr].mean()) for k in g[te]])
        elif model=="nn":  # nearest training location's mean
            d=df.iloc[tr].groupby("combine_loc")[["latitude","longitude",target]].mean()
            p=[]
            for la,lo in X[te][:,:2]:
                i=((d.latitude-la)**2+(d.longitude-lo)**2).idxmin(); p.append(d.loc[i,target])
            p=np.array(p)
        oof[te]=p; fm.append((MAE(y[te],p),MSE(y[te],p),R2(y[te],p) if len(te)>1 else np.nan))
        
    fm=np.array(fm)
    return dict(MAE_f=fm[:,0].mean(),MSE_f=fm[:,1].mean(),R2_f=np.nanmean(fm[:,2]),
                MAE=MAE(y,oof),RMSE=MSE(y,oof)**.5,R2_pooled=R2(y,oof))
out=[]
for name,target in [("moisture","moisture"),("suction","suction_pressure")]:
    df=pd.read_csv(f"{sys.argv[1]}/{name}.csv")
    ng=df.combine_loc.nunique()
    splits={"simple K=7":KFold(7),"shuffled K=7":KFold(7,shuffle=True,random_state=42),"grouped K=6":GroupKFold(6)}
    for sn,sp in splits.items():
        for label,feats,model in [("RF all features",F,"rf"),("RF coords only",F[:2],"rf"),("RF weather only",W,"rf"),
                                  ("Global mean",F,"mean"),("Location mean",F,"locmean"),("Nearest location",F,"nn")]:
            if model=="nn" and sn!="grouped K=6": continue
            r=run(df,target,feats,sp,model); r.update(dataset=name,split=sn,model=label); out.append(r)
    # dedup shuffled
    d2=df.drop_duplicates()
    r=run(d2,target,F,KFold(7,shuffle=True,random_state=42)); r.update(dataset=name,split="shuffled K=7, duplicates removed (n=%d)"%len(d2),model="RF all features"); out.append(r)
res=pd.DataFrame(out)[["dataset","split","model","MAE_f","MSE_f","R2_f","MAE","RMSE","R2_pooled"]]
pd.set_option("display.width",250); print(res.round(4).to_string(index=False))
res.to_csv("/home/claude/analysis/cv_results.csv",index=False)
