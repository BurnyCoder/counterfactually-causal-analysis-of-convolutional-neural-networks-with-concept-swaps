# Pilot (seed 0, discarded): tuning of 3a/3b/4a families vs published, per radius.
import json, numpy as np, torch
from concept_swap.config import Config
from concept_swap import model as M, stimuli as S
cfg = Config(radii=(10,15,20,30,45,60,80), widths=(3,5), n_theta=72)
dev='cuda'; m = M.load_inceptionv1(dev,'tf')
g = S.grid(cfg, seed=0)
X = S.render(g, cfg.arc_span_deg)
outs = {l: [] for l in ['mixed3a','mixed3b','mixed4a']}
with torch.no_grad():
    for i in range(0, len(X), 256):
        with M.capture(m, ['mixed3a','mixed3b','mixed4a']) as a:
            m(M.preprocess(X[i:i+256].to(dev)))
        outs['mixed3a'].append(a['mixed3a'][:,:,13,13].cpu()); outs['mixed3b'].append(a['mixed3b'][:,:,13,13].cpu()); outs['mixed4a'].append(a['mixed4a'][:,:,6,6].cpu())
outs = {k: torch.cat(v).numpy() for k,v in outs.items()}
pub = {l: json.load(open(f'data/circuits/curves-{l}.json')) for l in outs}
def tune(A, mask): # mean over cells -> (72, C)
    d = g[mask]; return np.stack([A[mask.values][d.theta.values==t].mean(0) for t in np.arange(72)*5])
for l in outs:
    neurons = pub[l]['neurons']; P = np.array([pub[l]['activations'][str(n)] for n in neurons])[:, ::10]  # 72 pts at 5deg
    for r in cfg.radii:
        T = tune(outs[l], g.radius==r)[:, neurons].T  # (N,72)
        best=(-9,None)
        for refl in (1,-1):
            for sh in range(72):
                idx = (refl*np.arange(72)+sh)%72
                rr = np.mean([np.corrcoef(T[i][idx], P[i])[0,1] if T[i].std()>0 else 0 for i in range(len(neurons))])
                if rr>best[0]: best=(rr,(refl,sh*5))
        print(l, 'r=',r, 'meanAct', T.mean().round(2), 'max', T.max().round(1), 'bestCorr', round(best[0],3), best[1], 'argmax-theta per neuron', (T.argmax(1)*5).tolist())
