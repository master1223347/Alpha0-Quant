"""Cross-sectional momentum/reversal baseline with costs, 1-bar lag and a 70/30 chronological split.

Usage: python scripts/xs_reversal_baseline.py <raw_dir> <bars_per_day>
"""

import glob
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
fr={}
for f in glob.glob(sys.argv[1]+"/*.txt"):
    d=pd.read_csv(f); ts=pd.to_datetime(d["<DATE>"].astype(str)+d["<TIME>"].astype(str).str.zfill(6),format="%Y%m%d%H%M%S")
    fr[d["<TICKER>"][0]]=pd.Series(d["<CLOSE>"].values,index=ts)
px=pd.DataFrame(fr).sort_index().ffill(limit=2)
ret=np.log(px).diff()
day=px.index.date
ret[pd.Series(day,index=px.index).ne(pd.Series(day,index=px.index).shift())]=np.nan  # drop overnight returns
n=len(px); cut=int(n*0.7); COST=1.5e-4  # per unit turnover (1bp cost + 0.5bp slippage)
def run(lb,hold,sign):
    sig=sign*ret.rolling(lb).sum()
    z=sig.sub(sig.mean(1),axis=0).div(sig.std(1),axis=0)
    w=z.div(z.abs().sum(1),axis=0).fillna(0)
    w=w.iloc[::hold].reindex(w.index).ffill()
    w=w.shift(1)  # 1-bar execution lag
    pnl=(w*ret.shift(-0)).sum(1)  # w at t (decided t-1) earns ret at t
    to=w.diff().abs().sum(1)
    return pnl-COST*to, pnl
res=[]
for lb in (1,3,6,12,24):
  for hold in (1,3,6,12):
    for sign in (1,-1):
      net,gross=run(lb,hold,sign)
      tr,te=net.iloc[:cut],net.iloc[cut:]
      sh=lambda x: x.mean()/x.std()*np.sqrt(float(sys.argv[2])*252) if x.std()>0 else 0
      res.append((lb,hold,"mom" if sign>0 else "rev",sh(tr),sh(te),gross.iloc[cut:].mean()*1e4,te.sum()))
r=pd.DataFrame(res,columns=["lb","hold","type","train_sh","test_sh","test_gross_bps_bar","test_net_cum"])
print(r.sort_values("train_sh",ascending=False).head(8).to_string())
print("trials:",len(r))
