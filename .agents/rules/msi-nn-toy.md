---
trigger: manual
---

**Step 0 — Put the candidates on equal footing.** This is what bit you with the unstandardized-log vs standardized-linear comparison. Fix one shared Sobol sample set, one train/val/test split, one training budget, and vary only the thing you're testing. Your candidate set is roughly {target transform: log+zscore, linear+zscore} × {activation/width: ReLU-16, GELU-32}, and **every** candidate gets its target standardized. Anything not held equal makes the ranking meaningless.

**Step 1 — Primary gate: Zhang Table 1, σ-stratified relative error.** This is the paper's own pass/fail test and it's non-negotiable. On the held-out test set, compute εmean and ε95 separately for Set 1 (σ=0.1), Set 2 (σ=0.3), Set 3 (σ=0.5), and require 1%/2%, 2%/5%, 3%/10% respectively. Compute it in **linear/relative space** (exp back if trained on log) because that's both what Table 1 means and what your log-space objective rewards. A candidate that fails any stratum is out, full stop. This already eliminates most contenders.

**Step 2 — Accuracy where the optimizer will actually travel.** Table 1's strata are radial shells, but TRF can push x toward the bounds — and you just saw the corners are ~40× more starved than the center. So additionally plot error vs ‖x‖ and spot-check a few boundary/corner points against true Cantera. A surrogate can pass Set 3 on average yet be useless at `[1,−1]`. If your data pulls the optimum toward an edge, you need the surrogate trustworthy *there*, or you need to shrink the box to where the data actually lives. This is the gate that catches false-minimum risk before it happens.

**Step 3 — Gradient/Jacobian fidelity (the one people skip).** Your optimizer direction, Σ\* (Eq. 10), and prediction bands (Eq. 12) all depend on ∂y/∂x, not just y. A net can fit y beautifully and still have noisy or biased gradients — ReLU's gradient is piecewise-constant, GELU's is smooth. Validate the analytic Jacobian against brute-force finite-difference Cantera at a handful of points (this mirrors Zhang's ranked-sensitivity check in Tables S1/S2). If the sensitivities are wrong, both the optimization and the UQ are wrong even when the y-fit looks perfect. This is a separate gate from Step 1.

**Step 4 — Truth-check at points the optimizer will likely visit.** Before committing, run *real* Cantera at the nominal, a couple of interior points, and along the direction the data pulls (faster R22, in your case), and compare to the surrogate. This is the Eq. 13 / F-score logic applied pre-optimization: if surrogate and Cantera already disagree in the promising region, the surrogate isn't ready regardless of its average error.

**Step 5 — Robustness across seeds.** NN training is stochastic, so retrain each surviving candidate a few times with different seeds and report mean ± spread of the Table 1 metrics (this is exactly why Zhang's Fig. 6 repeats training 10×). A candidate that only wins on a lucky seed isn't the best one — prefer accurate *and* low-variance.

**Decision rule once they've run the gauntlet:**
- Hard gate: must pass Table 1 on all three strata (Step 1) with acceptable gradients (Step 3). Everything else is a tie-breaker among survivors.
- Prefer the best **Set-1 (center)** accuracy, since your prior weights x≈0 most heavily and the optimum is expected near center.
- Prefer lower seed variance (Step 5) and edge behavior that doesn't ring (Step 2).
- Parsimony and paper-fidelity: if **ReLU-16** clears Table 1, it's the simplest, paper-faithful pick — its Hessian is exactly zero, so Eq. 12's second term vanishes cleanly and you carry no extra approximation. Only reach for **GELU-32** if ReLU genuinely can't hit the accuracy, and then you must keep the (corrected) Hessian term.
- target transform: for this bounded ~2.2× H₂O signal, log vs linear is close to a wash on accuracy once both are standardized, so break the tie on **consistency** — log+zscore makes the surrogate's training metric match your log-space objective and Table 1, so I'd give it the edge *if* it passes Step 1 on par with linear.