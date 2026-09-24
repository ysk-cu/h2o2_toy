#!/usr/bin/env python

# Multi-condition optimization of H2O2 rate parameters against Hong et al. shock tube data

import os, sys, json, time, datetime, itertools
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import cantera as ct
from scipy.optimize import least_squares
from scipy.stats import qmc, chi2
import warnings
warnings.filterwarnings("ignore", message=".*Arrhenius.*")
warnings.filterwarnings("ignore", message=".*ReactorBase.*")
warnings.filterwarnings("ignore", category=UserWarning, module="scipy.stats._qmc")

#   CONFIGURATION
MECHANISM = "chem_cti_toy_model_ogog.yaml"

# Which conditions enter the optimization.
FIT = ["1192", "1398"]

CONDITIONS = {
    "1192": dict(
        label   = "1192 K / 1.95 atm",
        result  = "result_1192k_train_final4.pt",
        T       = 1192.0,
        P_atm   = 1.95,
        X0      = {"H2O2": 2220e-6, "H2O": 1360e-6, "O2": 680e-6},   # rest is AR
        oh_csv  = "hong_1192K_oh.csv",
        h2o_csv = "hong_1192K_h2o.csv",
    ),
    "1398": dict(
        label   = "1398 K / 1.91 atm",
        result  = "result_1398k_train_final4.pt",
        T       = 1398.0,
        P_atm   = 1.91,
        X0      = {"H2O2": 2540e-6, "H2O": 1234e-6, "O2": 617e-6},
        oh_csv  = "hong_1398K_oh.csv",
        h2o_csv = "hong_1398K_h2o.csv",
    )
}

# Which analysis stages to run. 
ANALYSIS = dict(
    setup_checks          = True,   # compatibility, mechanism, perturbation, probes
    consistency_test      = True,   # F scores from the surrogate and from Cantera
    surrogate_free_check  = True,   # repeat the fit with Cantera in place of the NN
    freezing_ablation     = True,   # significance of each reaction
    leave_one_out         = True,   # fit N-1 conditions, predict the remaining one
    rate_constant_table   = True,   # k(T) with uncertainty, and the pivot temperature
    figures               = True,   # OH / H2O time history comparison
    uncertainty_band      = True,   # 2-sigma band on the figures (adds Cantera runs)
)

# Priors and experimental uncertainty
SIG_LOG       = 0.05     # experimental uncertainty
SIG_PRIOR_X   = 0.5      # prior sd of the normalized parameters -> lambda = 4
TAU_PRIOR_US  = 2.0      # prior sd of the OH trigger offset, microseconds
TAU_BOUND_US  = 6.0

# Zhang's thresholds
F_THRESHOLD   = 1.0      # |F| > 1 : inconsistent data point
CHI_X         = 0.05     # freezing test on the A factor multiplier
CHI_2SIGMA    = 0.96     # freezing test on retained uncertainty

N_RESTARTS    = 16       # multi-start count for the surrogate fit
SEED          = 0


CAL_PER_MOL = 4184.0
R_CAL       = ct.gas_constant / CAL_PER_MOL      # cal / (mol K)
LOG_EPS, NOISE_FLOOR = 1e-12, 1e-12
MOL_UNITS = ct.UnitSystem({
    "length": "cm", "mass": "g", "time": "s", "quantity": "mol",
    "pressure": "dyn / cm^2", "energy": "erg", "temperature": "K",
    "current": "A", "activation-energy": "cal / mol"})


class Tee:
    """Write to the terminal and to the log file at the same time."""
    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8")
        self.stdout = sys.stdout
    def write(self, s):
        self.stdout.write(s); self.f.write(s); self.f.flush()
    def flush(self):
        self.stdout.flush(); self.f.flush()
    def close(self):
        self.f.close()


def head(n, title):
    print("\n" + "=" * 78)
    print(f"  [{n}]  {title}")
    print("=" * 78)


def sub(title):
    print(f"\n  --- {title} " + "-" * max(0, 68 - len(title)))


class SurrogateNN(nn.Module):
    def __init__(self, n_in, hidden, n_out):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(),
                                 nn.Linear(hidden, n_out))
    def forward(self, x):
        return self.net(x)


class Condition:
    def __init__(self, key, mech=MECHANISM):
        cfg = CONDITIONS[key]
        self.key, self.label = key, cfg["label"]
        self.T, self.P = cfg["T"], cfg["P_atm"] * ct.one_atm
        self.P_atm = cfg["P_atm"]
        self.X0 = dict(cfg["X0"]); self.X0["AR"] = 1.0 - sum(cfg["X0"].values())
        self.mech = mech
        self.result_path = cfg["result"]

        ck = torch.load(cfg["result"], weights_only=False)
        self.input_dim = int(ck["input_dim"]); self.hidden = int(ck["hidden_dim"])
        self.n_oh  = int(ck["n_oh_targets"]);  self.n_h2o = int(ck["n_h2o_targets"])
        self.n_tot = self.n_oh + self.n_h2o
        self.oh_times  = np.asarray(ck["oh_target_times"],  float).ravel()
        self.h2o_times = np.asarray(ck["h2o_target_times"], float).ravel()
        self.ln_f    = float(ck["ln_f"]); self.sigma_e = float(ck["sigma_e"])
        self.param_names = list(ck["param_names"])
        self.best_val = float(ck.get("best_val", np.nan))

        # Reaction table
        def branches_of(idxs):
            idxs = [idxs] if np.isscalar(idxs) else list(idxs)
            return idxs
        spec = [("R1", branches_of(ck["idx_r1"]), True)]
        r2 = ck["idx_r2_all"] if "idx_r2_all" in ck else ck["idx_r2"]
        spec.append(("R2", branches_of(r2), False))
        spec.append(("R5", branches_of(ck["idx_r5"]), False))
        if self.input_dim >= 8 and ("idx_r4" in ck):
            spec.append(("R4", branches_of(ck["idx_r4"]), False))
        assert 2 * len(spec) == self.input_dim, \
            f"{len(spec)} reactions x 2 != input_dim {self.input_dim}"

        g = ct.Solution(self.mech)
        self.RXN = []
        for name, idxs, fall in spec:
            bl = []
            for i in idxs:
                rate = g.reaction(i).rate
                base = rate.low_rate if fall else rate
                ea = MOL_UNITS.convert_activation_energy_to(
                    f"{base.activation_energy} J/kmol", "cal / mol")
                bl.append((i, base.pre_exponential_factor,
                           base.temperature_exponent, ea))
            self.RXN.append((name, fall, bl))
        self.equations = {n: g.reaction(b[0][0]).equation for n, f, b in self.RXN}
        del g

        self.model = SurrogateNN(self.input_dim, self.hidden, self.n_tot)
        self.model.load_state_dict(ck["model_state"]); self.model.eval()

        self._load_hong(cfg["oh_csv"], cfg["h2o_csv"])
        self.mask = np.ones(self.n_tot, dtype=bool)
        self.use_tau = True
        self.use_surrogate = True
        self.active_idx = list(range(self.input_dim))
        self._gas = None

    # experimental data
    def _load_hong(self, oh_csv, h2o_csv):
        self.df_oh  = pd.read_csv(oh_csv,  skipinitialspace=True)
        self.df_h2o = pd.read_csv(h2o_csv, skipinitialspace=True)
        a = self.df_oh.groupby("Time [ms]")["[OH] ppm"].mean().reset_index()
        self.t_oh, self.y_oh = a["Time [ms]"].values * 1e-3, a["[OH] ppm"].values * 1e-6
        b = self.df_h2o.groupby("Time [ms]")["[H2O] ppm"].mean().reset_index()
        self.t_h2o, self.y_h2o = b["Time [ms]"].values * 1e-3, b["[H2O] ppm"].values * 1e-6
        y = np.interp(self.h2o_times, self.t_h2o, self.y_h2o)
        self.logH2O_obs = np.log(np.clip(y + LOG_EPS, NOISE_FLOOR, None))

    def logOH_obs_at(self, tau_us):
        y = np.interp(self.oh_times + tau_us * 1e-6, self.t_oh, self.y_oh)
        return np.log(np.clip(y + LOG_EPS, NOISE_FLOOR, None))

    def log_obs(self, tau_us):
        return np.concatenate([self.logOH_obs_at(tau_us), self.logH2O_obs])

    def labels(self):
        return ([f"OH  {t*1e3:.4f} ms"  for t in self.oh_times] +
                [f"H2O {t*1e3:.4f} ms" for t in self.h2o_times])

    def tau_leverage(self):
        """A priori forecast of Zhang's chi_2sigma test for the trigger offset:
        how many sigma_obs of OH residual one prior sigma of tau can produce."""
        h = 0.01 * TAU_PRIOR_US
        d = (self.logOH_obs_at(+h) - self.logOH_obs_at(-h)) / (2 * h)
        return float(np.mean(np.abs(d)) * TAU_PRIOR_US / SIG_LOG)

    def tau_leverage_crit(self):
        n = max(int(self.mask[:self.n_oh].sum()), 1)
        return float(np.sqrt((1.0 / CHI_2SIGMA ** 2 - 1.0) / n))

    # surrogate
    def nn_log(self, x):
        if not self.use_surrogate:
            return self.cantera_targets(x)
        with torch.no_grad():
            xt = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
            return self.model(xt).squeeze(0).numpy().astype(float)

    def nn_jac(self, x):
        if not self.use_surrogate:
            h = 2e-3
            J = np.empty((self.n_tot, self.input_dim))
            for k in range(self.input_dim):
                d = np.zeros(self.input_dim); d[k] = h
                J[:, k] = (self.cantera_targets(x + d)
                           - self.cantera_targets(x - d)) / (2 * h)
            return J
        xt = torch.tensor(x, dtype=torch.float32, requires_grad=True).unsqueeze(0)
        J = torch.autograd.functional.jacobian(lambda z: self.model(z).squeeze(0), xt)
        return J.detach().numpy().reshape(self.n_tot, self.input_dim).astype(float)

    # full model
    def perturb_gas(self, x, cached=True):
        """Every active reaction is rewritten from its stored nominal values on
        each call, so one Solution can be reused without drift. All duplicate
        branches get the SAME factor and the SAME Ea shift."""
        if cached:
            if self._gas is None:
                self._gas = ct.Solution(self.mech)
            gas = self._gas
        else:
            gas = ct.Solution(self.mech)
        for j, (name, fall, branches) in enumerate(self.RXN):
            fA  = np.exp(x[2 * j] * self.ln_f)
            dEa = x[2 * j + 1] * self.sigma_e
            for (i, A, b, Ea) in branches:
                r = gas.reaction(i)
                An, Ean = A * fA, (Ea + dEa) * CAL_PER_MOL
                if fall:
                    r.rate.low_rate = ct.Arrhenius(An, b, Ean)
                else:
                    r.rate = ct.Arrhenius(An, b, Ean)
                gas.modify_reaction(i, r)
        return gas

    def _phase(self, reactor):
        return reactor.phase if hasattr(reactor, "phase") else reactor.thermo

    def cantera_targets(self, x):
        gas = self.perturb_gas(x)
        gas.TPX = self.T, self.P, self.X0
        rr = ct.IdealGasConstPressureReactor(gas, energy="on")
        net = ct.ReactorNet([rr])
        oi, hi = gas.species_index("OH"), gas.species_index("H2O")
        allt = np.concatenate([self.oh_times, self.h2o_times])
        order = np.argsort(allt, kind="stable")
        ov = np.empty(len(allt)); hv = np.empty(len(allt))
        for k, t in enumerate(allt[order]):
            net.advance(float(t)); ph = self._phase(rr)
            ov[order[k]] = ph.X[oi]; hv[order[k]] = ph.X[hi]
        n = self.n_oh
        return np.concatenate([np.log(np.clip(ov[:n], 1e-30, None)),
                               np.log(np.clip(hv[n:], 1e-30, None))])

    def profile_grids(self):
        te_h, te_o = float(self.t_h2o.max()), float(self.t_oh.max())
        t_sim = np.linspace(te_h / 600.0, te_h, 600)
        knee = 0.25 * te_o
        t_oh = np.concatenate([np.linspace(1e-7, knee, 500),
                               np.linspace(knee * 1.001, te_o, 200)])
        return t_sim, t_oh

    def run_profiles(self, x, t_sim, t_oh):
        gas = self.perturb_gas(x)
        gas.TPX = self.T, self.P, self.X0
        rr = ct.IdealGasConstPressureReactor(gas, energy="on")
        net = ct.ReactorNet([rr])
        hi, oi = gas.species_index("H2O"), gas.species_index("OH")
        allt = np.concatenate([t_sim, t_oh]); order = np.argsort(allt, kind="stable")
        hv = np.empty(len(allt)); ov = np.empty(len(allt))
        for k, t in enumerate(allt[order]):
            net.advance(float(t)); ph = self._phase(rr)
            hv[order[k]] = ph.X[hi]; ov[order[k]] = ph.X[oi]
        n = len(t_sim)
        return hv[:n], ov[n:]

    def k_of(self, gas, name, fall, branches, T):
        """Rate constant of one reaction. For a falloff reaction this is the
        low pressure limit, which is what the active parameters describe; the
        blended effective k does not scale with k0 and must not be used here."""
        gas.TP = T, ct.one_atm
        if fall:
            tot = 0.0
            for (i, *_r) in branches:
                lr = gas.reaction(i).rate.low_rate
                ea = MOL_UNITS.convert_activation_energy_to(
                    f"{lr.activation_energy} J/kmol", "cal / mol")
                tot += (lr.pre_exponential_factor * T ** lr.temperature_exponent
                        * np.exp(-ea / (R_CAL * T)))
            return tot
        return sum(gas.forward_rate_constants[i] for (i, *_r) in branches)

    def check_perturbation(self):
        gas0 = ct.Solution(self.mech)
        worst = 0.0
        for j, (name, fall, br) in enumerate(self.RXN):
            x = np.zeros(self.input_dim); x[2*j], x[2*j+1] = 0.7, -0.4
            gas1 = self.perturb_gas(x, cached=False)
            for T in sorted({CONDITIONS[k]["T"] for k in FIT}):
                k0 = self.k_of(gas0, name, fall, br, T)
                k1 = self.k_of(gas1, name, fall, br, T)
                want = np.exp(x[2*j]*self.ln_f - x[2*j+1]*self.sigma_e/(R_CAL*T))
                rel = abs(k1/k0/want - 1.0); worst = max(worst, rel)
                tag = " (low-P limit)" if fall else ""
                print(f"      {name} @{T:.0f} K{tag:15s} k'/k = {k1/k0:.8f}  "
                      f"expected {want:.8f}  dev {rel:.1e}  "
                      f"{'ok' if rel < 1e-8 else 'MISMATCH'}")
        return worst


# residual, Jacobian, solver
def make_residual(fit_conds):
    """
    z = [ x_active , tau_c for conditions carrying one ]

    r = [ per-condition (log y_pred - log y_obs)/SIG_LOG , masked ]
        [ x_active / SIG_PRIOR_X ]      <-- ONCE, not once per condition
        [ tau_c / TAU_PRIOR_US ]

    least_squares minimizes 0.5*||r||^2, so 2*cost is Zhang Eq. 8 with
    lambda = 1/SIG_PRIOR_X^2 = 4. Repeating the prior block per condition would
    double lambda and shrink the posterior covariance for no reason.
    """
    NX  = fit_conds[0].input_dim
    act = fit_conds[0].active_idx
    NA  = len(act)
    tau_conds = [c for c in fit_conds if c.use_tau]
    NT = len(tau_conds)
    slot = {c.key: i for i, c in enumerate(tau_conds)}

    def expand(za):
        x = np.zeros(NX); x[act] = za; return x

    def tau_of(c, z):
        return float(z[NA + slot[c.key]]) if c.use_tau else 0.0

    def residual(z):
        x = expand(z[:NA]); parts = []
        for c in fit_conds:
            r = (c.nn_log(x) - c.log_obs(tau_of(c, z))) / SIG_LOG
            parts.append(r[c.mask])
        parts.append(x[act] / SIG_PRIOR_X)
        if NT:
            parts.append(z[NA:NA + NT] / TAU_PRIOR_US)
        return np.concatenate(parts)

    def jac(z):
        x = expand(z[:NA]); rows = []; h = 0.01 * TAU_PRIOR_US
        for c in fit_conds:
            J = c.nn_jac(x)[:, act] / SIG_LOG
            T = np.zeros((c.n_tot, NT))
            if c.use_tau:
                t = tau_of(c, z)
                d = (c.logOH_obs_at(t + h) - c.logOH_obs_at(t - h)) / (2 * h)
                T[:c.n_oh, slot[c.key]] = -d / SIG_LOG
            rows.append(np.hstack([J, T])[c.mask])
        rows.append(np.hstack([np.eye(NA) / SIG_PRIOR_X, np.zeros((NA, NT))]))
        if NT:
            rows.append(np.hstack([np.zeros((NT, NA)), np.eye(NT) / TAU_PRIOR_US]))
        return np.vstack(rows)

    return residual, jac, tau_conds, act, expand


def solve(fit_conds, n_restart=N_RESTARTS, seed=SEED, x0=None, quiet=False):
    res, jac, tau_conds, act, expand = make_residual(fit_conds)
    NA, NT = len(act), len(tau_conds)
    NZ = NA + NT
    lo = np.array([-1.0]*NA + [-TAU_BOUND_US]*NT)
    hi = np.array([+1.0]*NA + [+TAU_BOUND_US]*NT)

    z0 = np.zeros(NZ)
    if x0 is not None:
        z0[:NA] = np.asarray(x0, float)[act]
    starts = [z0]
    if n_restart > 1:
        u = qmc.Sobol(d=NZ, scramble=True, seed=seed).random(n_restart - 1)
        for row in u:
            s = np.empty(NZ)
            s[:NA] = -0.5 + row[:NA]
            s[NA:] = -2.0 + 4.0 * row[NA:]
            starts.append(s)

    best, costs = None, []
    for s in starts:
        sol = least_squares(res, s, jac=jac, bounds=(lo, hi), method="trf",
                            xtol=1e-12, ftol=1e-12, gtol=1e-12)
        costs.append(sol.cost)
        if best is None or sol.cost < best.cost:
            best = sol
    costs = np.array(costs)
    if not quiet:
        near = costs < costs.min() * 1.01 + 1e-12
        print(f"      multi-start: {len(starts)} runs, best 2*cost = {2*costs.min():.6g}, "
              f"{near.sum()}/{len(costs)} within 1%")
        if near.sum() < len(costs) and np.ptp(costs) > 0.05 * max(costs.min(), 1e-12):
            print("      note: restarts disagree; compare the x* they reach before "
                  "trusting one of them")
    return best, costs, jac, act, tau_conds


def unpack(sol, fit_conds, act, tau_conds, jac):
    NX = fit_conds[0].input_dim
    NA, NT = len(act), len(tau_conds)
    J = jac(sol.x)
    cov = np.linalg.inv(J.T @ J)
    Sigma = np.zeros((NX, NX)); Sigma[np.ix_(act, act)] = cov[:NA, :NA]
    x = np.zeros(NX); x[act] = sol.x[:NA]
    taus = {c.key: float(sol.x[NA + i]) for i, c in enumerate(tau_conds)}
    for c in fit_conds:
        taus.setdefault(c.key, 0.0)
    return x, Sigma, cov, taus


def f_scores(cond, x, tau, source):
    """Zhang Eq. 13:  F = (y_obs - y_opt) / (2 sigma_obs). |F| > 1 = inconsistent.
    Relative to the residual r = (y_pred - y_obs)/SIG_LOG this is simply F = -r/2."""
    y = cond.nn_log(x) if source == "nn" else cond.cantera_targets(x)
    return (cond.log_obs(tau) - y) / (2.0 * SIG_LOG)


def fit_tau_only(cond, x, source="cantera"):
    y = cond.nn_log(x) if source == "nn" else cond.cantera_targets(x)
    r = lambda t: np.concatenate([(y - cond.log_obs(t[0])) / SIG_LOG,
                                  [t[0] / TAU_PRIOR_US]])
    s = least_squares(r, [0.0], bounds=([-TAU_BOUND_US], [TAU_BOUND_US]), method="trf")
    return float(s.x[0])


def k_ratio_and_sd(x, Sigma, j, ln_f, sigma_e, T):
    """Rate constant relative to nominal, and the standard deviation of its
    logarithm, for reaction slot j at temperature T.
        ln(k/k_nom) = x_A*ln_f - x_Ea*sigma_e/(R T)
        Var = ln_f^2 S_AA - 2 ln_f (sigma_e/RT) S_AE + (sigma_e/RT)^2 S_EE
    """
    g = sigma_e / (R_CAL * T)
    m = np.exp(x[2*j] * ln_f - x[2*j+1] * g)
    S = Sigma[2*j:2*j+2, 2*j:2*j+2]
    var = ln_f**2 * S[0,0] - 2*ln_f*g*S[0,1] + g**2 * S[1,1]
    return m, float(np.sqrt(max(var, 0.0)))


#  reporting
def report_parameters(conds, fit_conds, x, Sigma, taus, tau_conds):
    names = fit_conds[0].param_names
    ln_f, sig_e = fit_conds[0].ln_f, fit_conds[0].sigma_e
    act = fit_conds[0].active_idx
    sd = np.sqrt(np.diag(Sigma))

    sub("optimized rate parameters")
    print(f"      {'parameter':<10}{'x*':>9}{'posterior sd':>14}{'sd/prior':>10}"
          f"   physical value")
    for k, n in enumerate(names):
        if k not in act:
            print(f"      {n:<10}{'frozen':>9}{'-':>14}{'-':>10}   at nominal")
            continue
        phys = (f"A factor x {np.exp(x[k]*ln_f):.4f}" if k % 2 == 0
                else f"Ea {x[k]*sig_e:+.0f} cal/mol")
        print(f"      {n:<10}{x[k]:>+9.4f}{sd[k]:>14.4f}{sd[k]/SIG_PRIOR_X:>10.2f}"
              f"   {phys}")

    sub("OH trigger offset")
    for c in fit_conds:
        L, Lc = c.tau_leverage(), c.tau_leverage_crit()
        if c.use_tau:
            print(f"      {c.label:<20} tau = {taus[c.key]:+.3f} us   "
                  f"leverage {L:.3f} (threshold {Lc:.3f})")
        else:
            print(f"      {c.label:<20} frozen at 0   leverage {L:.3f} "
                  f"(threshold {Lc:.3f}, below it this parameter is not determinable)")

    sub("Zhang freezing test on the rate parameters")
    x_crit = np.log1p(CHI_X) / ln_f
    print(f"      criterion: |x| < {x_crit:.4f} AND posterior sd > "
          f"{CHI_2SIGMA*SIG_PRIOR_X:.3f}  (did not move and did not tighten)")
    flagged = [names[k] for k in act
               if abs(x[k]) < x_crit and sd[k] > CHI_2SIGMA * SIG_PRIOR_X]
    print("      flagged for freezing: " + (", ".join(flagged) if flagged else "none"))

    sub("posterior structure")
    Sa = Sigma[np.ix_(act, act)]
    ev, evec = np.linalg.eigh(Sa)
    order = np.argsort(ev)[::-1]
    sds = np.sqrt(ev[order])
    print(f"      sd along the eigen-directions (prior = {SIG_PRIOR_X}):")
    print("      " + "  ".join(f"{v:.4f}" for v in sds))
    n_at_prior = int(np.sum(sds > CHI_2SIGMA * SIG_PRIOR_X))
    print(f"      directions still at the prior level: {n_at_prior} of {len(act)}"
          "   (these are not determined by the data)")
    print(f"      condition number of the posterior covariance: "
          f"{ev[order[0]]/max(ev[order[-1]],1e-30):.4g}")
    for i in order[:3]:
        dom = names[act[int(np.argmax(np.abs(evec[:, i])))]]
        print(f"        sd = {np.sqrt(ev[i]):.4f}   largest component: {dom:<8} "
              f"vector = {np.round(evec[:, i], 2)}")
    print("      correlation between ln A and Ea:")
    for j in range(len(names)//2):
        if 2*j in act and 2*j+1 in act:
            rho = Sigma[2*j,2*j+1]/np.sqrt(Sigma[2*j,2*j]*Sigma[2*j+1,2*j+1])
            print(f"        {names[2*j][4:]:<4} {rho:+.3f}"
                  + ("   (still on the A / Ea line: only the combination is determined)"
                     if abs(rho) > 0.9 else ""))
    return dict(sd=sd.tolist(), eigen_sd=sds.tolist(), n_at_prior=n_at_prior)


def report_rate_constants(fit_conds, x, Sigma):
    c0 = fit_conds[0]
    names = c0.param_names
    ln_f, sig_e = c0.ln_f, c0.sigma_e
    act = c0.active_idx
    temps = sorted({c.T for c in fit_conds})
    out = {}
    sub("rate constants relative to nominal, at the experimental temperatures")
    print(f"      {'reaction':<6}{'T [K]':>8}{'k/k_nom':>10}{'2 sigma range':>22}")
    for j in range(len(names)//2):
        nm = names[2*j][4:]
        if 2*j not in act:
            print(f"      {nm:<6}{'':>8}{'frozen at nominal':>32}")
            continue
        for T in temps:
            m, s = k_ratio_and_sd(x, Sigma, j, ln_f, sig_e, T)
            print(f"      {nm:<6}{T:>8.0f}{m:>10.3f}"
                  f"{f'{m*np.exp(-2*s):.3f} .. {m*np.exp(+2*s):.3f}':>22}")
            out[f"{nm}_{int(T)}"] = dict(ratio=m, sd_lnk=s)

    sub("pivot temperature (where this fit is most certain)")
    print("      Away from the pivot the A factor and Ea trade off and the band widens.")
    print("      Quote k at the pivot, not at the edges of an extrapolated range.")
    Tg = np.linspace(300.0, 2500.0, 2201)
    for j in range(len(names)//2):
        nm = names[2*j][4:]
        if 2*j not in act:
            continue
        sds = np.array([k_ratio_and_sd(x, Sigma, j, ln_f, sig_e, T)[1] for T in Tg])
        i = int(np.nanargmin(sds))
        m, s = k_ratio_and_sd(x, Sigma, j, ln_f, sig_e, Tg[i])
        print(f"      {nm:<6} pivot = {Tg[i]:6.0f} K   k/k_nom = {m:.3f} "
              f"+/- {s:.3f} (1 sd of ln k)   [at 2500 K the sd is {sds[-1]:.3f}]")
        out[f"{nm}_pivot"] = dict(T=float(Tg[i]), ratio=m, sd_lnk=s,
                                  sd_lnk_2500=float(sds[-1]))
    return out


def report_consistency(conds, fit_conds, eval_keys, fit_keys, x, taus, with_cantera):
    worst = 0.0; store = {}
    for k in eval_keys:
        c = conds[k]
        held = k not in fit_keys
        tau = taus.get(k, 0.0)
        if held:
            tau = fit_tau_only(c, x, "cantera" if with_cantera else "nn")
        Fn = f_scores(c, x, tau, "nn")
        Fc = f_scores(c, x, tau, "cantera") if with_cantera else None
        sub(f"{c.label}" + ("   [HELD OUT: this is a prediction]" if held else "")
            + f"   tau = {tau:+.3f} us")
        hdr = f"      {'target':<16}{'F (surrogate)':>15}"
        if with_cantera:
            hdr += f"{'F (Cantera)':>14}{'difference':>12}{'obs/pred':>10}"
        print(hdr + "   status")
        for i, lab in enumerate(c.labels()):
            ref = Fc[i] if with_cantera else Fn[i]
            st = "INCONSISTENT" if abs(ref) > F_THRESHOLD else "ok"
            line = f"      {lab:<16}{Fn[i]:>+15.3f}"
            if with_cantera:
                line += (f"{Fc[i]:>+14.3f}{Fc[i]-Fn[i]:>+12.3f}"
                         f"{np.exp(2*ref*SIG_LOG):>10.3f}")
            print(line + f"   {st}" + ("" if c.mask[i] else "  [removed]"))
        ref = Fc if with_cantera else Fn
        mx = float(np.abs(ref).max())
        print(f"      largest |F| = {mx:.3f}"
              + ("   all data points consistent" if mx < F_THRESHOLD
                 else "   AT LEAST ONE DATA POINT CANNOT BE RECONCILED"))
        if with_cantera:
            d = float(np.abs(Fc - Fn).max())
            print(f"      largest surrogate contribution to |F| = {d:.3f} "
                  f"({100*(np.exp(2*d*SIG_LOG)-1):.1f}% error in the predicted value)")
            if d > 0.2 * F_THRESHOLD:
                print("      warning: the surrogate accounts for a noticeable part of "
                      "the F value; judge consistency on the Cantera column only")
        if k in fit_keys:
            worst = max(worst, mx)
        store[k] = dict(tau_us=tau, labels=c.labels(), F_nn=Fn.tolist(),
                        F_cantera=(Fc.tolist() if Fc is not None else None),
                        held_out=held, max_abs_F=mx)
    return worst, store


def report_chi2(fit_conds, sol, n_free):
    n_data = sum(int(c.mask.sum()) for c in fit_conds)
    S = float(np.sum(sol.fun[:n_data] ** 2))
    lo = max(n_data - n_free, 1); hi = n_data
    sub("overall goodness of fit")
    print(f"      sum of squared data residuals = {S:.3f} over {n_data} data points")
    print(f"      effective degrees of freedom between {lo} (prior weak) and "
          f"{hi} (prior strong)")
    print(f"      chi-square 95% critical value: {chi2.ppf(0.95, lo):.2f} .. "
          f"{chi2.ppf(0.95, hi):.2f}")
    v = ("consistent" if S <= chi2.ppf(0.95, hi)
         else "NOT consistent under either convention" if S > chi2.ppf(0.95, lo)
         else "borderline, depends on how informative the prior is judged to be")
    print(f"      verdict: {v}")
    return S, n_data


def make_figure(conds, eval_keys, fit_keys, x, Sigma, taus, path, band):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.ndimage import uniform_filter1d
    NX = conds[eval_keys[0]].input_dim
    fig, axes = plt.subplots(len(eval_keys), 2,
                             figsize=(16, 6*len(eval_keys)), squeeze=False)
    for row, key in enumerate(eval_keys):
        c = conds[key]
        t_sim, t_oh = c.profile_grids()
        h_nom, o_nom = c.run_profiles(np.zeros(NX), t_sim, t_oh)
        h_opt, o_opt = c.run_profiles(x, t_sim, t_oh)
        if band:
            EPS = 1e-3
            Jh = np.zeros((len(t_sim), NX)); Jo = np.zeros((len(t_oh), NX))
            lg = lambda v: np.log(np.clip(v, 1e-30, None))
            for k in range(NX):
                d = np.zeros(NX); d[k] = EPS
                hp, op = c.run_profiles(x + d, t_sim, t_oh)
                hm, om = c.run_profiles(x - d, t_sim, t_oh)
                Jh[:, k] = (lg(hp) - lg(hm)) / (2*EPS)
                Jo[:, k] = (lg(op) - lg(om)) / (2*EPS)
            Jh = uniform_filter1d(Jh, 20, axis=0); Jo = uniform_filter1d(Jo, 20, axis=0)
            hs = np.sqrt(np.einsum("ti,ij,tj->t", Jh, Sigma, Jh))
            os_ = np.sqrt(np.einsum("ti,ij,tj->t", Jo, Sigma, Jo))
        tau = taus.get(key, 0.0)
        yo, yh = np.exp(c.logOH_obs_at(tau)), np.exp(c.logH2O_obs)
        eu = lambda y: y*1e6*(np.exp(2*SIG_LOG)-1); ed = lambda y: y*1e6*(1-np.exp(-2*SIG_LOG))
        held = key not in fit_keys
        panels = [(axes[row][0], t_sim, h_nom, h_opt, c.df_h2o, "[H2O] ppm",
                   c.h2o_times, yh, "H2O", (hs if band else None)),
                  (axes[row][1], t_oh, o_nom, o_opt, c.df_oh, "[OH] ppm",
                   c.oh_times, yo, "OH", (os_ if band else None))]
        for ax, tg, pn, po, df, yl, tt, yt, sp, sd in panels:
            ax.plot(df["Time [ms]"], df[yl], "o", mfc="none", mec="k", ms=4,
                    alpha=0.4, label="Hong et al.")
            ax.errorbar(tt*1e3, yt*1e6, yerr=[ed(yt), eu(yt)], fmt="none", ecolor="k",
                        elinewidth=1.5, capsize=4, label=r"targets $\pm2\sigma$")
            ax.plot(tg*1e3, pn*1e6, "r--", lw=1.5, alpha=0.7, label="nominal")
            ax.plot(tg*1e3, po*1e6, "b-", lw=2.5,
                    label="prediction" if held else "optimized")
            if sd is not None:
                ax.fill_between(tg*1e3, po*1e6*np.exp(-2*sd), po*1e6*np.exp(2*sd),
                                color="steelblue", alpha=0.2, label=r"$\pm2\sigma$")
            ax.set(xlabel="Time [ms]", ylabel=yl, xlim=[0, tg.max()*1e3*1.02],
                   title=f"{c.label}  --  {sp}" +
                         ("   [HELD OUT]" if held else "   [fitted]"))
            ax.legend(frameon=False, fontsize=9); ax.grid(True, ls="--", alpha=0.3)
    names = conds[eval_keys[0]].param_names
    ln_f, sig_e = conds[eval_keys[0]].ln_f, conds[eval_keys[0]].sigma_e
    st = "  |  ".join(f"{names[2*j][4:]}: Ax{np.exp(x[2*j]*ln_f):.3f}, "
                      f"Ea{x[2*j+1]*sig_e:+.0f}" for j in range(len(names)//2))
    fig.suptitle(f"fit on {', '.join(fit_keys)} K\n{st}", fontsize=10)
    plt.tight_layout(); plt.savefig(path, dpi=110, bbox_inches="tight"); plt.close(fig)
    print(f"      saved {path}")


def main():
    t_start = time.time()
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = f"analysis_{'_'.join(FIT)}_{stamp}"
    os.makedirs(outdir, exist_ok=True)
    tee = Tee(os.path.join(outdir, "analysis.log"))
    sys.stdout = tee

    print("=" * 78)
    print("  H2O2 RATE PARAMETER OPTIMIZATION -- FULL ANALYSIS")
    print(f"  run at        : {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"  mechanism     : {MECHANISM}")
    print(f"  conditions    : {', '.join(FIT)} K")
    print(f"  output        : {outdir}/")
    print("=" * 78)
    print(f"\n  experimental uncertainty (log space)  : {SIG_LOG}")
    print(f"  prior sd of normalized parameters     : {SIG_PRIOR_X}  "
          f"(lambda = {1/SIG_PRIOR_X**2:.0f})")
    print(f"  prior sd of the OH trigger offset     : {TAU_PRIOR_US} us")
    print(f"  inconsistency threshold               : |F| > {F_THRESHOLD}")

    conds = {k: Condition(k) for k in FIT}
    fit_conds = [conds[k] for k in FIT]
    results = dict(run=stamp, mechanism=MECHANISM, fit=FIT,
                   sig_log=SIG_LOG, sig_prior_x=SIG_PRIOR_X,
                   lam=1/SIG_PRIOR_X**2, tau_prior_us=TAU_PRIOR_US)

    #  1. setup 
    if ANALYSIS["setup_checks"]:
        head(1, "SETUP AND VALIDATION")
        sub("surrogate files")
        ref = fit_conds[0]
        for c in fit_conds:
            print(f"      {c.label:<20} {c.result_path}   "
                  f"{c.n_oh} OH + {c.n_h2o} H2O targets, hidden {c.hidden}, "
                  f"best val loss {c.best_val:.3e}")
            assert c.param_names == ref.param_names, "parameter order differs"
            assert abs(c.ln_f - ref.ln_f) < 1e-10, "LN_F differs; x is not the same variable"
            assert abs(c.sigma_e - ref.sigma_e) < 1e-6, "SIGMA_E differs"
        print(f"      shared parameters: {ref.param_names}")
        print(f"      LN_F = {ref.ln_f:.6f}, SIGMA_E = {ref.sigma_e:.0f} cal/mol")

        sub("mechanism content")
        g = ct.Solution(MECHANISM)
        for name, fall, br in ref.RXN:
            print(f"      {name}: {ref.equations[name]:<32} "
                  f"{len(br)} branch(es) at {[b[0] for b in br]}"
                  + ("   falloff, low pressure limit is the active parameter"
                     if fall else ""))
            if name == "R2" and len(br) < 2:
                print("      *** WARNING: R2 has a single branch. If the surrogates "
                      "were trained on the corrected mechanism this file is wrong.")
        for T in sorted({c.T for c in fit_conds}):
            ks = ", ".join(f"k({n}) = {ref.k_of(g, n, f, b, T):.4g}"
                           for n, f, b in ref.RXN)
            print(f"      at {T:.0f} K: {ks}")
        del g

        sub("perturbation is an exact multiplicative factor on k(T)")
        print("      A common A factor and a common Ea shift on every duplicate branch")
        print("      must give  k'(T)/k(T) = exp(x_A*LN_F - x_Ea*SIGMA_E/RT) exactly.")
        w = ref.check_perturbation()
        if w > 1e-8:
            print(f"      *** FAILED (worst deviation {w:.2e}). A duplicate branch is "
                  "being missed. Stopping.")
            sys.stdout = tee.stdout; tee.close(); return
        print("      all reactions pass")

        sub("surrogate against the full model")
        print("      At x = 0 both must agree. A probe with one parameter displaced")
        print("      catches a training / optimization mismatch that x = 0 hides,")
        print("      for example a duplicate pair perturbed on one branch only.")
        NX = ref.input_dim
        probes = [("x = 0", np.zeros(NX))]
        for j, (nm, _f, _b) in enumerate(ref.RXN):
            p = np.zeros(NX); p[2*j] = 0.3
            probes.append((f"lnA_{nm} = +0.3", p))
        worst_probe = 0.0
        for c in fit_conds:
            print(f"      {c.label}")
            for lab, z in probes:
                gap = np.abs(np.exp(c.nn_log(z) - c.cantera_targets(z)) - 1.0) * 100
                worst_probe = max(worst_probe, gap.max())
                print(f"        {lab:<16} largest {gap.max():5.2f}%   "
                      + " ".join(f"{v:.2f}" for v in gap)
                      + ("   <-- CHECK" if gap.max() > 10 else ""))
        results["worst_probe_percent"] = worst_probe
        if worst_probe > 10:
            print("      *** WARNING: the surrogate and the full model disagree by more")
            print("      *** than 10%. Check that the mechanism file matches the one used")
            print("      *** for training, and that both scripts perturb the duplicate")
            print("      *** branches the same way.")
        else:
            print(f"      consistent at every probe (largest {worst_probe:.2f}%)")

    #  decide which conditions carry a trigger offset
    for c in fit_conds:
        c.use_tau = c.tau_leverage() >= c.tau_leverage_crit()

    #  2. joint optimization 
    head(2, "OPTIMIZATION")
    n_data = sum(c.n_tot for c in fit_conds)
    n_tau = sum(1 for c in fit_conds if c.use_tau)
    print(f"\n  {n_data} data points, {fit_conds[0].input_dim} rate parameters "
          f"+ {n_tau} trigger offsets = {fit_conds[0].input_dim + n_tau} unknowns")
    if n_data <= fit_conds[0].input_dim + n_tau:
        print("  WARNING: there are no more data points than unknowns. The result is")
        print("  determined mostly by the prior and the consistency test has no power.")
    sol, costs, jac, act, tau_conds = solve(fit_conds)
    x, Sigma, cov, taus = unpack(sol, fit_conds, act, tau_conds, jac)
    print(f"      objective value (Zhang Eq. 8) = {2*sol.cost:.6f}")
    pinfo = report_parameters(conds, fit_conds, x, Sigma, taus, tau_conds)
    S_data, n_pts = report_chi2(fit_conds, sol, len(act) + n_tau)
    results.update(param_names=fit_conds[0].param_names, x_opt=x.tolist(),
                   Sigma=Sigma.tolist(), tau_us=taus, objective=2*sol.cost,
                   sum_sq_data_residuals=S_data, n_data_points=n_pts, **pinfo)

    if ANALYSIS["rate_constant_table"]:
        results["rate_constants"] = report_rate_constants(fit_conds, x, Sigma)

    #  3. consistency
    if ANALYSIS["consistency_test"]:
        head(3, "CONSISTENCY TEST  (Zhang Eq. 13)")
        print("\n  F = (observed - predicted) / (2 x experimental uncertainty).")
        print("  |F| > 1 means that data point cannot be reconciled with the model.")
        print("  The Cantera column is the one to judge on: it uses the full model")
        print("  rather than the surrogate, so it is not affected by surrogate error.")
        worst, fst = report_consistency(conds, fit_conds, FIT, FIT, x, taus, True)
        results["f_scores"] = fst
        results["max_abs_F"] = worst

    #  4. surrogate-free confirmation 
    if ANALYSIS["surrogate_free_check"]:
        head(4, "CONFIRMATION WITHOUT THE SURROGATE")
        print("\n  The Cantera column of the consistency test checks the prediction at")
        print("  a given x. It does not check x itself, because x came out of the")
        print("  surrogate. Repeating the fit with Cantera in place of the network is")
        print("  the only check on the optimized parameters. Starting from the")
        print("  surrogate optimum, so this converges in a few iterations.")
        for c in fit_conds:
            c.use_surrogate = False
        sol2, _, jac2, act2, tc2 = solve(fit_conds, n_restart=1, x0=x, quiet=True)
        x2, Sig2, _, taus2 = unpack(sol2, fit_conds, act2, tc2, jac2)
        for c in fit_conds:
            c.use_surrogate = True
        names = fit_conds[0].param_names
        sub("optimized parameters with and without the surrogate")
        print(f"      {'parameter':<10}{'surrogate':>12}{'Cantera':>12}"
              f"{'difference':>13}{'in sd':>9}   physical difference")
        worst_sd = 0.0
        for k, n in enumerate(names):
            if k not in act: continue
            d = x2[k] - x[k]
            sdk = np.sqrt(Sigma[k, k]); rel = abs(d)/sdk if sdk > 0 else np.nan
            worst_sd = max(worst_sd, rel)
            phys = (f"A factor x {np.exp(d*fit_conds[0].ln_f):.4f}" if k % 2 == 0
                    else f"{d*fit_conds[0].sigma_e:+.0f} cal/mol")
            print(f"      {n:<10}{x[k]:>+12.4f}{x2[k]:>+12.4f}{d:>+13.4f}"
                  f"{rel:>9.2f}   {phys}")
        print(f"      largest difference is {worst_sd:.2f} posterior standard deviations")
        if worst_sd < 0.3:
            print("      the optimized parameters are not an artifact of the surrogate")
        else:
            print("      *** the surrogate is shifting the optimum; improve it before")
            print("      *** reporting these parameters")
        results["x_opt_cantera"] = x2.tolist()
        results["surrogate_shift_in_sd"] = worst_sd

    #  5. freezing ablation 
    if ANALYSIS["freezing_ablation"] and len(fit_conds[0].RXN) > 1:
        head(5, "SIGNIFICANCE OF EACH REACTION")
        print("\n  Each reaction is fixed at its nominal value and the fit repeated.")
        print("  The rise in the sum of squared data residuals is compared against the")
        print("  chi-square distribution with 2 degrees of freedom (one A factor and")
        print("  one activation energy). Critical values: 5.99 at 95%, 9.21 at 99%.")
        print("  A reaction that does not raise the residual is not determined by the")
        print("  data and should be frozen (Zhang Eqs. 15-16).")
        base_names = [n for n, f, b in fit_conds[0].RXN]
        full_idx = list(range(fit_conds[0].input_dim))
        abl = {}
        print(f"\n      {'frozen':<10}{'residual sum':>14}{'rise':>10}"
              f"{'p value':>11}   verdict")
        print(f"      {'none':<10}{S_data:>14.3f}{'':>10}{'':>11}   reference")
        for j, nm in enumerate(base_names):
            for c in fit_conds:
                c.active_idx = [k for k in full_idx if k // 2 != j]
            s_, _, j_, a_, t_ = solve(fit_conds, n_restart=max(4, N_RESTARTS//4),
                                      quiet=True)
            nd = sum(int(c.mask.sum()) for c in fit_conds)
            Sf = float(np.sum(s_.fun[:nd] ** 2))
            rise = Sf - S_data
            p = 1.0 - chi2.cdf(max(rise, 0.0), 2)
            verd = ("significant" if p < 0.05 else "NOT significant, freeze it")
            print(f"      {nm:<10}{Sf:>14.3f}{rise:>10.3f}{p:>11.4f}   {verd}")
            abl[nm] = dict(residual_sum=Sf, rise=rise, p_value=p,
                           significant=bool(p < 0.05))
        for c in fit_conds:
            c.active_idx = full_idx
        results["ablation"] = abl
        weak = [n for n, v in abl.items() if not v["significant"]]
        if weak:
            print(f"\n      Recommendation: freeze {', '.join(weak)} and re-run. This")
            print("      removes parameters the data does not determine and tightens")
            print("      the uncertainty on the remaining ones.")

    #  6. leave one out 
    if ANALYSIS["leave_one_out"] and len(FIT) >= 2:
        head(6, "PREDICTIVE TEST  (leave one condition out)")
        print("\n  Each condition is removed from the fit in turn and then predicted.")
        print("  This is the only test of predictive capability: the consistency test")
        print("  above uses every condition in the fit, so it can only show internal")
        print("  agreement. A prediction that fails while the joint fit passes means")
        print("  the conditions are being reconciled by parameters rather than agreeing.")
        loo = {}
        for held in FIT:
            keep = [k for k in FIT if k != held]
            sub(f"fit on {', '.join(keep)} K, predict {held} K")
            sub_conds = [conds[k] for k in keep]
            s_, _, j_, a_, t_ = solve(sub_conds, quiet=True)
            xl, Sl, _, taul = unpack(s_, sub_conds, a_, t_, j_)
            print(f"      x* = {np.round(xl, 4)}")
            w, fs = report_consistency(conds, sub_conds, FIT, keep, xl, taul, True)
            loo[held] = dict(fit_on=keep, x_opt=xl.tolist(),
                             max_abs_F_held=fs[held]["max_abs_F"], f_scores=fs)
            m = fs[held]["max_abs_F"]
            print(f"      prediction for {held} K: largest |F| = {m:.3f}   "
                  + ("passes" if m < F_THRESHOLD else "FAILS"))
        results["leave_one_out"] = loo

    #  7. figures 
    if ANALYSIS["figures"]:
        head(7, "FIGURES")
        p = os.path.join(outdir, "profiles.png")
        make_figure(conds, FIT, FIT, x, Sigma, taus, p, ANALYSIS["uncertainty_band"])

    #  summary 
    head(8, "SUMMARY")
    print(f"\n  conditions fitted            : {', '.join(FIT)} K")
    print(f"  free rate parameters         : {len(act)} of {fit_conds[0].input_dim}")
    print(f"  objective value              : {2*sol.cost:.4f}")
    print(f"  sum of squared data residuals: {S_data:.3f} over {n_pts} points")
    if "max_abs_F" in results:
        m = results["max_abs_F"]
        print(f"  largest |F| (Cantera)        : {m:.3f}   "
              + ("all data points consistent" if m < F_THRESHOLD else "INCONSISTENT"))
    if "surrogate_shift_in_sd" in results:
        print(f"  surrogate shift in optimum   : {results['surrogate_shift_in_sd']:.2f} sd")
    print(f"  directions still at prior    : {results['n_at_prior']} of {len(act)}")
    if "ablation" in results:
        for n, v in results["ablation"].items():
            print(f"  {n} significance              : p = {v['p_value']:.4f}  "
                  + ("significant" if v["significant"] else "not significant"))
    if "leave_one_out" in results:
        for h, v in results["leave_one_out"].items():
            print(f"  prediction of {h} K          : largest |F| = "
                  f"{v['max_abs_F_held']:.3f}  "
                  + ("passes" if v["max_abs_F_held"] < F_THRESHOLD else "FAILS"))
    print(f"\n  elapsed: {time.time()-t_start:.1f} s")

    with open(os.path.join(outdir, "result.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"  written: {outdir}/analysis.log")
    print(f"           {outdir}/result.json")
    if ANALYSIS["figures"]:
        print(f"           {outdir}/profiles.png")

    sys.stdout = tee.stdout
    tee.close()


if __name__ == "__main__":
    main()