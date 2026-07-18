#!/usr/bin/env python
# coding: utf-8
#
# optimize_joint_generic.py
# Generalized joint MAP inference + experiment-comparison figure for the trained
# joint OH+H2O surrogates. Auto-detects 6- vs 8-parameter (k1/k2/k5 [+k4]) from
# the checkpoint. Mirrors the methodology of optimize_1398k_joint_nn_k5.ipynb:
#   - rate-param Gaussian prior (SIG_PRIOR_X) + OH-trigger tau nuisance
#   - analytic NN Jacobian for the solve
#   - Cantera-integrated profiles at nominal and x*, with FD-Jacobian +/-2sig band
#
# Usage:
#   python optimize_joint_generic.py --result result_1192k_joint_r5_joint_dopt.pt \
#       --T 1192 --P_atm 1.95 --xH2O2 2220e-6 --xH2O 1360e-6 --xO2 680e-6 \
#       --oh_csv hong_1192K_oh.csv --h2o_csv hong_1192K_h2o.csv \
#       --out fig_1192k_r5_16.png --tag "1192 K  k1/k2/k5  hidden16"
import argparse, numpy as np, torch, torch.nn as nn, pandas as pd, cantera as ct
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.ndimage import uniform_filter1d

p = argparse.ArgumentParser()
p.add_argument("--result", required=True)
p.add_argument("--oh_csv", required=True)
p.add_argument("--h2o_csv", required=True)
p.add_argument("--T", type=float, required=True)
p.add_argument("--P_atm", type=float, required=True)
p.add_argument("--xH2O2", type=float, required=True)
p.add_argument("--xH2O", type=float, required=True)
p.add_argument("--xO2", type=float, required=True)
p.add_argument("--out", required=True)
p.add_argument("--tag", default="")
p.add_argument("--yaml", default="chem_cti_toy_model_og.yaml")
args = p.parse_args()

SIG_LOG, LOG_EPS, NOISE_FLOOR = 0.05, 1e-12, 1e-12
SIG_PRIOR_X = 0.5
TAU_PRIOR_US, TAU_BOUND_US = 2.0, 6.0
YAML_FILE = args.yaml
T_INITIAL = args.T
P_INITIAL = args.P_atm * ct.one_atm
INITIAL_X = {'H2O2': args.xH2O2, 'H2O': args.xH2O, 'O2': args.xO2,
             'AR': 1.0 - (args.xH2O2 + args.xH2O + args.xO2)}

# ── Load checkpoint ───────────────────────────────────────────────────────────
ckpt      = torch.load(args.result, weights_only=False)
INPUT_DIM = int(ckpt["input_dim"])
HIDDEN    = int(ckpt["hidden_dim"])
N_OH      = int(ckpt["n_oh_targets"]); N_H2O = int(ckpt["n_h2o_targets"])
N_TOTAL   = N_OH + N_H2O
oh_times  = np.asarray(ckpt["oh_target_times"]); h2o_times = np.asarray(ckpt["h2o_target_times"])
LN_F      = float(ckpt["ln_f"]); SIGMA_E = float(ckpt["sigma_e"])
PARAM_NAMES = list(ckpt["param_names"])
N_Z = INPUT_DIM + 1

class SurrogateNN(nn.Module):
    def __init__(self, hidden, n_out):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(INPUT_DIM, hidden), nn.ReLU(), nn.Linear(hidden, n_out))
    def forward(self, x): return self.net(x)

model = SurrogateNN(HIDDEN, N_TOTAL); model.load_state_dict(ckpt["model_state"]); model.eval()
print(f"Loaded {args.result}: {INPUT_DIM}->{HIDDEN}->{N_TOTAL}  (OH {N_OH}, H2O {N_H2O})")
print(f"OH  target times (ms): {np.round(oh_times*1e3,4)}")
print(f"H2O target times (ms): {np.round(h2o_times*1e3,4)}")

# ── Nominal rate params for the ACTIVE reactions (order = param vector) ────────
mol_units = ct.UnitSystem({"length":"cm","mass":"g","time":"s","quantity":"mol",
    "pressure":"dyn / cm^2","energy":"erg","temperature":"K","current":"A",
    "activation-energy":"cal / mol"})
_g = ct.Solution(YAML_FILE)
# param order is [R1, R2, R5, (R4)] -> falloff flag True only for R1
_specs = [("R1", int(ckpt["idx_r1"]), True), ("R2", int(ckpt["idx_r2"]), False),
          ("R5", int(ckpt["idx_r5"]), False)]
if INPUT_DIM >= 8 and "idx_r4" in ckpt:
    _specs.append(("R4", int(ckpt["idx_r4"]), False))
RXN = []   # (name, idx, is_falloff, A, b, Ea_cal)
for name, idx, fall in _specs:
    rate = _g.reaction(idx).rate
    base = rate.low_rate if fall else rate
    Ea = mol_units.convert_activation_energy_to(f"{base.activation_energy} J/kmol", "cal / mol")
    RXN.append((name, idx, fall, base.pre_exponential_factor, base.temperature_exponent, Ea))
    print(f"  {name} idx{idx} {'(falloff)' if fall else ''}: A={base.pre_exponential_factor:.3e} "
          f"b={base.temperature_exponent:+.2f} Ea={Ea:.0f} cal/mol")
del _g
assert len(RXN) * 2 == INPUT_DIM, f"{len(RXN)} reactions x2 != INPUT_DIM {INPUT_DIM}"

# ── Experimental data ─────────────────────────────────────────────────────────
df_h2o = pd.read_csv(args.h2o_csv, skipinitialspace=True)
df_oh  = pd.read_csv(args.oh_csv,  skipinitialspace=True)
h2o_agg = df_h2o.groupby("Time [ms]")["[H2O] ppm"].mean().reset_index()
t_h2o = h2o_agg["Time [ms]"].values*1e-3; y_h2o = h2o_agg["[H2O] ppm"].values*1e-6
oh_agg  = df_oh.groupby("Time [ms]")["[OH] ppm"].mean().reset_index()
t_oh  = oh_agg["Time [ms]"].values*1e-3; y_oh = oh_agg["[OH] ppm"].values*1e-6
y_oh_at_tgt  = np.interp(oh_times,  t_oh,  y_oh)
y_h2o_at_tgt = np.interp(h2o_times, t_h2o, y_h2o)
logH2O_obs = np.log(np.clip(y_h2o_at_tgt + LOG_EPS, NOISE_FLOOR, None))

# ── NN forward / Jacobian, residual, solve (tau nuisance on OH) ───────────────
def nn_log_all(x):
    with torch.no_grad():
        return model(torch.tensor(x, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
def nn_jac(x):
    xt = torch.tensor(x, dtype=torch.float32, requires_grad=True).unsqueeze(0)
    J = torch.autograd.functional.jacobian(lambda xx: model(xx).squeeze(0), xt)
    return J.detach().numpy().reshape(N_TOTAL, INPUT_DIM)
def logOH_obs_at(tau_us):
    y = np.interp(oh_times + tau_us*1e-6, t_oh, y_oh)
    return np.log(np.clip(y + LOG_EPS, NOISE_FLOOR, None))
def residual(z):
    x, tau = z[:INPUT_DIM], z[INPUT_DIM]
    lp = nn_log_all(x)
    return np.concatenate([(lp[:N_OH]-logOH_obs_at(tau))/SIG_LOG,
                           (lp[N_OH:]-logH2O_obs)/SIG_LOG,
                           x/SIG_PRIOR_X, [tau/TAU_PRIOR_US]])
def residual_jac(z):
    x, tau = z[:INPUT_DIM], z[INPUT_DIM]
    J = nn_jac(x)/SIG_LOG
    h = 1e-3
    dobs = (logOH_obs_at(tau+h)-logOH_obs_at(tau-h))/(2*h)
    return np.vstack([
        np.hstack([J[:N_OH], (-dobs/SIG_LOG)[:,None]]),
        np.hstack([J[N_OH:], np.zeros((N_H2O,1))]),
        np.hstack([np.eye(INPUT_DIM)/SIG_PRIOR_X, np.zeros((INPUT_DIM,1))]),
        np.hstack([np.zeros((1,INPUT_DIM)), np.array([[1.0/TAU_PRIOR_US]])])])
sol = least_squares(residual, x0=np.zeros(N_Z), jac=residual_jac,
                    bounds=([-1.0]*INPUT_DIM+[-TAU_BOUND_US], [1.0]*INPUT_DIM+[TAU_BOUND_US]),
                    method="trf")
z_opt = sol.x; cov = np.linalg.inv(residual_jac(z_opt).T @ residual_jac(z_opt))
x_opt = z_opt[:INPUT_DIM]; tau_opt = z_opt[INPUT_DIM]
Sigma_star = cov[:INPUT_DIM,:INPUT_DIM]; sig_opt = np.sqrt(np.diag(cov))[:INPUT_DIM]
print(f"\nJOINT MAP  cost={0.5*np.sum(sol.fun**2):.4g}  tau={tau_opt:+.3f} us")
for n, zv, sv in zip(PARAM_NAMES, x_opt, sig_opt): print(f"  {n:<9}{zv:>8.4f}  +/-{sv:.4f}")
# Diagnostic: eigen-decomposition of Sigma_star reveals which parameter COMBINATION
# is weakly identified (large eigenvalue) and therefore dominates the propagated
# +/-2sig band noise/width -- e.g. k4 (Burke R4) is expected to show up here given
# its small (~0.13) OH sensitivity vs k1's ~0.92.
_evals, _evecs = np.linalg.eigh(Sigma_star)
_order = np.argsort(_evals)[::-1]
print(f"  Sigma_star cond number: {_evals[_order[0]]/max(_evals[_order[-1]],1e-30):.3g}")
print("  Weakest-identified directions (largest posterior variance eigenvalues):")
for i in _order[:min(3, INPUT_DIM)]:
    dom = PARAM_NAMES[int(np.argmax(np.abs(_evecs[:, i])))]
    print(f"    eigval={_evals[i]:.4g}  dominant param: {dom:<9}  vec={np.round(_evecs[:,i],2)}")
_p = nn_log_all(x_opt)
print(f"  OH  resid (sig): {np.round((_p[:N_OH]-logOH_obs_at(tau_opt))/SIG_LOG,2)}")
print(f"  H2O resid (sig): {np.round((_p[N_OH:]-logH2O_obs)/SIG_LOG,2)}")

# ── Cantera profiles at nominal & x*, + FD posterior band ─────────────────────
DT_SIM, N_STEPS = 1e-6, 1000
T_SIM = np.linspace(DT_SIM, DT_SIM*N_STEPS, N_STEPS)
T_OH = np.concatenate([np.linspace(1e-7, 2e-4, 600), np.linspace(2e-4+5e-6, 1e-3, 200)])
def perturb_gas(x):
    gas = ct.Solution(YAML_FILE)
    for j,(name,idx,fall,A,b,Ea) in enumerate(RXN):
        An = A*np.exp(x[2*j]*LN_F); Ean = (Ea + x[2*j+1]*SIGMA_E)*4184.0
        r = gas.reaction(idx)
        if fall: r.rate.low_rate = ct.Arrhenius(An, b, Ean)
        else:    r.rate = ct.Arrhenius(An, b, Ean)
        gas.modify_reaction(idx, r)
    return gas
def run_profiles(x):
    gas = perturb_gas(x); gas.TPX = T_INITIAL, P_INITIAL, INITIAL_X
    rr = ct.IdealGasConstPressureReactor(gas, energy="on"); net = ct.ReactorNet([rr])
    hi = gas.species_index("H2O"); oi = gas.species_index("OH")
    allt = np.concatenate([T_SIM, T_OH]); order = np.argsort(allt, kind="stable")
    hv = np.empty(len(allt)); ov = np.empty(len(allt))
    for k,t in enumerate(allt[order]):
        net.advance(t); hv[order[k]] = rr.thermo.X[hi]; ov[order[k]] = rr.thermo.X[oi]
    return hv[:N_STEPS], ov[N_STEPS:]
print("Cantera nominal / x* ...")
h2o_nom, oh_nom = run_profiles(np.zeros(INPUT_DIM))
h2o_opt, oh_opt = run_profiles(x_opt)
# Central differences (not one-sided): the residual-Jacobian's tau column already
# uses central FD (logOH_obs_at(tau+h)-logOH_obs_at(tau-h))/(2h); this Cantera FD
# Jacobian previously used one-sided forward FD, which is O(EPS) vs O(EPS^2) accurate.
# That extra truncation noise gets amplified precisely in weakly-identified parameter
# directions (large posterior variance in Sigma_star, e.g. k4) -> visibly jittery
# +/-2sig bands. Central differences remove that as the dominant noise source.
EPS = 1e-3
Jh = np.zeros((N_STEPS,INPUT_DIM)); Jo = np.zeros((len(T_OH),INPUT_DIM))
for k in range(INPUT_DIM):
    dx = np.zeros(INPUT_DIM); dx[k]=EPS
    hp, op   = run_profiles(x_opt+dx)
    hm, om   = run_profiles(x_opt-dx)
    Jh[:,k] = (np.log(np.clip(hp,1e-30,None))-np.log(np.clip(hm,1e-30,None)))/(2*EPS)
    Jo[:,k] = (np.log(np.clip(op,1e-30,None))-np.log(np.clip(om,1e-30,None)))/(2*EPS)
Jo = uniform_filter1d(Jo,20,axis=0); Jh = uniform_filter1d(Jh,20,axis=0)
h2o_lv = np.einsum("ti,ij,tj->t", Jh, Sigma_star, Jh)
oh_lv  = np.einsum("ti,ij,tj->t", Jo, Sigma_star, Jo)
h2o_up, h2o_lo = h2o_opt*np.exp(+2*np.sqrt(h2o_lv)), h2o_opt*np.exp(-2*np.sqrt(h2o_lv))
oh_up,  oh_lo  = oh_opt*np.exp(+2*np.sqrt(oh_lv)),  oh_opt*np.exp(-2*np.sqrt(oh_lv))

# ── Figure ────────────────────────────────────────────────────────────────────
yeh_u = y_h2o_at_tgt*1e6*(np.exp(2*SIG_LOG)-1); yeh_d = y_h2o_at_tgt*1e6*(1-np.exp(-2*SIG_LOG))
yeo_u = y_oh_at_tgt*1e6*(np.exp(2*SIG_LOG)-1);  yeo_d = y_oh_at_tgt*1e6*(1-np.exp(-2*SIG_LOG))
h2o_xlim = max(t_h2o.max()*1e3*1.05, 0.06); oh_xlim = max(t_oh.max()*1e3*1.05, 0.06)
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(16,6))
ax1.plot(df_h2o["Time [ms]"], df_h2o["[H2O] ppm"],"o",mfc="none",mec="k",ms=4,alpha=0.4,label="Hong et al.")
ax1.errorbar(h2o_times*1e3, y_h2o_at_tgt*1e6, yerr=[yeh_d,yeh_u],fmt="none",ecolor="k",elinewidth=1.5,capsize=4)
ax1.plot(T_SIM*1e3,h2o_nom*1e6,"r--",lw=1.5,alpha=0.7,label="Nominal")
ax1.plot(T_SIM*1e3,h2o_opt*1e6,"b-",lw=2.5,label="MAP x*")
ax1.fill_between(T_SIM*1e3,h2o_lo*1e6,h2o_up*1e6,color="steelblue",alpha=0.2,label="+/-2sig")
ax1.set(xlabel="Time [ms]",ylabel="H2O [ppm]",xlim=[0,h2o_xlim],title="H2O")
ax1.legend(frameon=False,fontsize=9); ax1.grid(True,ls="--",alpha=0.3)
ax2.plot(df_oh["Time [ms]"], df_oh["[OH] ppm"],"o",mfc="none",mec="k",ms=4,alpha=0.4,label="Hong et al.")
ax2.errorbar(oh_times*1e3, y_oh_at_tgt*1e6, yerr=[yeo_d,yeo_u],fmt="none",ecolor="k",elinewidth=1.5,capsize=4)
ax2.plot(T_OH*1e3,oh_nom*1e6,"r--",lw=1.5,alpha=0.7,label="Nominal")
ax2.plot(T_OH*1e3,oh_opt*1e6,"b-",lw=2.5,label="MAP x*")
ax2.fill_between(T_OH*1e3,oh_lo*1e6,oh_up*1e6,color="steelblue",alpha=0.2,label="+/-2sig")
ax2.set(xlabel="Time [ms]",ylabel="OH [ppm]",xlim=[0,oh_xlim],title="OH")
ax2.legend(frameon=False,fontsize=9); ax2.grid(True,ls="--",alpha=0.3)
_sub = "  |  ".join(f"{RXN[j][0]}: Ax{np.exp(x_opt[2*j]*LN_F):.3f}, Ea{x_opt[2*j+1]*SIGMA_E:+.0f}"
                    for j in range(len(RXN)))
fig.suptitle(f"Joint MAP -- {args.tag}   (tau={tau_opt:+.2f} us)\n{_sub}", fontsize=10)
plt.tight_layout(); plt.savefig(args.out, dpi=110, bbox_inches="tight")
print(f"Saved figure: {args.out}")
