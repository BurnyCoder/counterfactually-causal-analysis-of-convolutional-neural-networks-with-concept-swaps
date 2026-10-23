# Pilot H1 (jitter seeds 0 = pilot-selection, 100 = pilot-test; both discarded later).
import time, numpy as np, torch
from concept_swap.config import Config
from concept_swap import model as M, stimuli as S, families as Fm, readouts as R, interventions as I, stats as St
cfg = Config(); dev='cuda'; m = M.load_inceptionv1(dev,'tf'); t0=time.time()
fam3b = Fm.load_family(cfg.data_dir,'mixed3b').neurons; fam4a = np.array(Fm.load_family(cfg.data_dir,'mixed4a').neurons)
gs, gt = S.grid(cfg, 0), S.grid(cfg, 100)
cs = I.collect(m, S.render(gs, cfg.arc_span_deg), 'mixed4a', dev, 64); ct = I.collect(m, S.render(gt, cfg.arc_span_deg), 'mixed4a', dev, 64)
print('collect', time.time()-t0)
z = R.ZScore().fit(cs.readout.numpy()[:, fam4a])
dec = R.TemplateDecoder().fit(z(cs.readout.numpy()[:, fam4a]), gs.theta.values, gs.cell.values)
th_clean = dec.predict(z(ct.readout.numpy()[:, fam4a]), gt.cell.values)
acc = np.abs(R.circ_dist(th_clean, gt.theta.values)) <= cfg.tol_deg
print('clean template acc', acc.mean(), 'by radius', gt.assign(a=acc).groupby('radius').a.mean().round(3).to_dict())
rid = R.RidgeDecoder().fit(cs.readout.numpy(), gs.theta.values)
print('clean ridge acc (all 508)', (np.abs(R.circ_dist(rid.predict(ct.readout.numpy()), gt.theta.values))<=cfg.tol_deg).mean())
p = S.minimal_pairs(gt, cfg, 100); b, s = p.base.values, p.source.values
elig = acc[b] & acc[s]; print('pairs', len(p), 'eligible', elig.sum())
def reduce(Rr): return np.abs(R.circ_dist(dec.predict(z(Rr), gt.cell.values[b]), th_clean[s])) <= cfg.tol_deg
# delta per channel from selection-split pairs
ps = S.minimal_pairs(gs, cfg, 0); W = cs.windows
delta = ((W[ps.source.values]-W[ps.base.values])**2).mean((0,2,3)).numpy()
print('delta rank of family (pct)', [round((delta<delta[n]).mean(),2) for n in fam3b])
rng = np.random.default_rng(0)
nul = Fm.delta_matched_sets(delta, fam3b, 200, rng); br = Fm.pool_sets(np.setdiff1d(np.arange(320,416), fam3b), 10, 200, rng)
fis = np.argsort(-Fm.fisher_scores(cs.readout.new_tensor(0).numpy() if False else W[:, :, 5, 5].numpy(), gs.theta.values))[:10]
print('fisher top10', fis.tolist(), 'jaccard', len(set(fis)&set(fam3b))/len(set(fis)|set(fam3b)))
t1=time.time()
sets = [np.array(fam3b), np.arange(480), np.setdiff1d(np.arange(480), fam3b), fis]
out = I.patched_readouts(m,'mixed4a', ct.windows, b, s, sets, fam4a, reduce)
print('curve, all, complement, fisher IIA:', [round(o[elig].mean(),3) for o in out], 'time', time.time()-t1)
t1=time.time(); on = I.patched_readouts(m,'mixed4a', ct.windows, b, s, list(nul), fam4a, reduce)[:, elig]; print('null 200 sets time', time.time()-t1)
ob = I.patched_readouts(m,'mixed4a', ct.windows, b, s, list(br), fam4a, reduce)[:, elig]
print('delta-null mean/q95', on.mean().round(3), np.quantile(on.mean(1),0.95).round(3), 'branch mean/q95', ob.mean().round(3), np.quantile(ob.mean(1),0.95).round(3))
Wt = St.pigeonhole_weights(b[elig], s[elig], 500, rng)
r = St.margin_vs_null(out[0][elig], on, Wt); print({k:(np.round(v,3) if not isinstance(v,tuple) else tuple(np.round(v,3))) for k,v in r.items()})
print('dtheta bins', p[elig].assign(o=out[0][elig]).groupby(pd.cut(np.abs(p[elig].dtheta),[29,60,90,120,150,170,180])).o.mean().round(3).to_dict() if False else '')
import pandas as pd
pe = p[elig].assign(o=out[0][elig]); print(pe.groupby(pd.cut(np.abs(pe.dtheta),[29,60,90,120,150,175,180])).o.agg(['mean','size']))
print('total', time.time()-t0)
