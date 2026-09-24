#!/usr/bin/env python
# coding: utf-8
#
# optimize_joint_multicondition.py
#
# Two-condition (1192 K + 1398 K) joint MAP inference on ONE shared normalized
# rate-parameter vector x, plus Zhang's F-score consistency test (Eq. 13).
#
# Extends optimize_joint_generic.py.  Differences:
#   1. x is shared across conditions; each condition keeps its own OH-trigger
#      nuisance tau_c.  z = [x (INPUT_DIM), tau_c for c in fit_conditions].
#   2. The MUM-PCE prior term x/SIG_PRIOR_X appears EXACTLY ONCE in the residual
#      vector, not once per condition.  (lambda = 1/SIG_PRIOR_X^2 = 4, matching
#      Zhang Eq. 10 / FFCM-1.)  Concatenating two per-condition residual builders
#      would double lambda and spuriously shrink Sigma*.
#   3. F-score is computed from BOTH the surrogate and a direct Cantera run at x*.
#      The OH surrogate does not pass Zhang Table 1 (Set 1 p95 = 2.4-3.1% against
#      sigma_obs = 5%), so surrogate-based F carries up to ~0.25 of spurious
#      |F| against a threshold of 1.0.  Cull on the Cantera F only.
#   4. Multi-start TRF, because the lnA/Ea ridge makes a single start from x0 = 0
#      uninformative about whether the optimum is unique.
#
# Usage
# -----
#   # joint fit on both conditions, F reported for both, Cantera-verified
#   python optimize_joint_multicondition.py --fit 1192,1398 --verify_cantera
#
#   # reproduce a single-condition fit (sanity check against optimize_joint_generic)
#   python optimize_joint_multicondition.py --fit 1192 --eval 1192
#
#   # HOLD-OUT: fit on 1192 only, predict 1398 -- the test that actually counts
#   python optimize_joint_multicondition.py --fit 1192 --eval 1192,1398 --verify_cantera
#
#   # Zhang's iterative cull (see the warning printed by the loop before using)
#   python optimize_joint_multicondition.py --fit 1192,1398 --cull --verify_cantera

import argparse, json
import warnings
import numpy as np
import torch, torch.nn as nn
import pandas as pd
import cantera as ct
from scipy.optimize import least_squares
from scipy.stats import qmc, chi2

warnings.filterwarnings("ignore", category=DeprecationWarning, module="cantera")
warnings.filterwarnings("ignore", message=".*Arrhenius.*")
warnings.filterwarnings("ignore", message=".*ReactorBase.*")

# ── Fixed constants (do not tune) ─────────────────────────────────────────────
SIG_LOG               = 0.05        # sigma_m,obs, log-space, Zhang Eq. 8 & 13
LOG_EPS, NOISE_FLOOR  = 1e-12, 1e-12
SIG_PRIOR_X           = 0.5         # -> lambda = 4
TAU_PRIOR_US          = 2.0
TAU_BOUND_US          = 6.0
CAL_PER_MOL           = 4184.0
YAML_FILE             = "chem_cti_toy_model_ogog.yaml"   # override with --yaml
F_THRESHOLD           = 1.0         # Zhang: |F_m| > 1 => inconsistent
# Zhang Sec. 3.4 freezing thresholds (FFCM-1a values).  chi_x is stated on the
# A-factor multiplier |A_k/A_k,0 - 1|; in normalized x with the ln_f convention
# that is |x_k| < ln(1 + chi_x)/ln_f  (= 0.021 for chi_x=0.05, ln_f=ln 10).
# tau has no A-multiplier, so chi_x is applied to it in normalized units instead
# -- that substitution is an extension, not something Zhang specifies.
CHI_X                 = 0.05
CHI_2SIGMA            = 0.96

COND_LIB = {
    "1192": dict(
        label="1192 K / 1.95 atm",
        result="result_1192k_train_final.pt",
        T=1192.0, P_atm=1.95,
        X0={"H2O2": 2220e-6, "H2O": 1360e-6, "O2": 680e-6},
        oh_csv="hong_1192K_oh.csv", h2o_csv="hong_1192K_h2o.csv"),
    "1398": dict(
        label="1398 K / 1.91 atm",
        result="result_1398k_train_final.pt",
        T=1398.0, P_atm=1.91,
        X0={"H2O2": 2540e-6, "H2O": 1234e-6, "O2": 617e-6},
        oh_csv="hong_1398K_oh.csv", h2o_csv="hong_1398K_h2o.csv"),
}

MOL_UNITS = ct.UnitSystem({
    "length": "cm", "mass": "g", "time": "s", "quantity": "mol",
    "pressure": "dyn / cm^2", "energy": "erg", "temperature": "K",
    "current": "A", "activation-energy": "cal / mol"})


class SurrogateNN(nn.Module):
    def __init__(self, n_in, hidden, n_out):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(),
                                 nn.Linear(hidden, n_out))

    def forward(self, x):
        return self.net(x)


class Condition:
    """One shock-tube condition: its surrogate, its Hong data, its Cantera setup."""

    def __init__(self, key, yaml_file=YAML_FILE):
        cfg = COND_LIB[key]
        self.key, self.label = key, cfg["label"]
        self.T, self.P = cfg["T"], cfg["P_atm"] * ct.one_atm
        self.X0 = dict(cfg["X0"])
        self.X0["AR"] = 1.0 - sum(cfg["X0"].values())
        self.yaml = yaml_file

        ck = torch.load(cfg["result"], weights_only=False)
        self.result_path = cfg["result"]
        self.input_dim = int(ck["input_dim"])
        self.hidden    = int(ck["hidden_dim"])
        self.n_oh      = int(ck["n_oh_targets"])
        self.n_h2o     = int(ck["n_h2o_targets"])
        self.n_tot     = self.n_oh + self.n_h2o
        self.oh_times  = np.asarray(ck["oh_target_times"],  dtype=float).ravel()
        self.h2o_times = np.asarray(ck["h2o_target_times"], dtype=float).ravel()
        self.ln_f      = float(ck["ln_f"])
        self.sigma_e   = float(ck["sigma_e"])
        self.param_names = list(ck["param_names"])

        specs = [("R1", int(ck["idx_r1"]), True),
                 ("R2", int(ck["idx_r2"]), False),
                 ("R5", int(ck["idx_r5"]), False)]
        if self.input_dim >= 8 and "idx_r4" in ck:
            specs.append(("R4", int(ck["idx_r4"]), False))
        assert 2 * len(specs) == self.input_dim, \
            f"{len(specs)} reactions x2 != input_dim {self.input_dim}"

        g = ct.Solution(self.yaml)
        self.RXN = []
        for name, idx, fall in specs:
            # A reaction may now be a DUPLICATE PAIR (R26 after the fix). Perturbing
            # only ckpt's idx would scale one branch and leave the other at nominal,
            # i.e. optimize half of k2. Collect every branch with the same
            # reactants/products and perturb them together: a common A-factor and a
            # common Ea shift factor out of the sum, giving an exact multiplicative
            # perturbation of the total k(T).
            ref = g.reaction(idx)
            sib = [i for i, r in enumerate(g.reactions())
                   if set(r.reactants) == set(ref.reactants)
                   and set(r.products) == set(ref.products)]
            branches = []
            for i in sib:
                rate = g.reaction(i).rate
                base = rate.low_rate if fall else rate
                Ea = MOL_UNITS.convert_activation_energy_to(
                    f"{base.activation_energy} J/kmol", "cal / mol")
                branches.append((i, base.pre_exponential_factor,
                                 base.temperature_exponent, Ea))
            if len(branches) > 1:
                print(f"  {name}: duplicate pair, {len(branches)} branches at "
                      f"indices {[b[0] for b in branches]} -- perturbed together")
            self.RXN.append((name, fall, branches))
        del g

        self.model = SurrogateNN(self.input_dim, self.hidden, self.n_tot)
        self.model.load_state_dict(ck["model_state"])
        self.model.eval()

        self._load_hong(cfg["oh_csv"], cfg["h2o_csv"])
        # Target mask: True = target participates in the objective.
        # Zhang's cull loop flips entries to False.
        self.mask = np.ones(self.n_tot, dtype=bool)
        # Whether this condition carries an OH-trigger nuisance tau.
        # Set from --tau; see tau_leverage() for whether it is identifiable.
        self.use_tau = True

    def tau_leverage(self):
        """
        A PRIORI screen: how many sigma_obs of OH residual one prior-sigma of tau
        can produce.  This is NOT Zhang's criterion -- Zhang's freezing test
        (Sec. 3.4) is post-hoc, after optimization.  This is a cheap forecast of
        what that test will say, valid in the linearized limit.

        Link to Zhang: with n_OH targets of equal leverage L and a unit-precision
        prior on tau, the posterior sd relative to prior is 1/sqrt(1 + n_OH*L^2).
        Setting that equal to chi_2sigma gives the threshold in tau_leverage_crit().

        tau enters only through the interpolated EXPERIMENTAL trace, so its
        Jacobian column is the slope of the data, not of the model.  Where the OH
        targets sit on a flat, coarsely-sampled plateau that slope is digitization
        noise and tau is unidentifiable however it is fitted.
        """
        h = max(1e-3, 0.01 * TAU_PRIOR_US)
        d = (self.logOH_obs_at(+h) - self.logOH_obs_at(-h)) / (2 * h)
        return float(np.mean(np.abs(d)) * TAU_PRIOR_US / SIG_LOG)

    def tau_leverage_crit(self):
        """Leverage below which Zhang's chi_2sigma test will call tau unconstrained."""
        n = max(int(self.mask[:self.n_oh].sum()), 1)
        return float(np.sqrt((1.0 / CHI_2SIGMA ** 2 - 1.0) / n))

    # ── data ──────────────────────────────────────────────────────────────────
    def _load_hong(self, oh_csv, h2o_csv):
        df_oh  = pd.read_csv(oh_csv,  skipinitialspace=True)
        df_h2o = pd.read_csv(h2o_csv, skipinitialspace=True)
        self.df_oh, self.df_h2o = df_oh, df_h2o
        a = df_oh.groupby("Time [ms]")["[OH] ppm"].mean().reset_index()
        self.t_oh = a["Time [ms]"].values * 1e-3
        self.y_oh = a["[OH] ppm"].values * 1e-6
        b = df_h2o.groupby("Time [ms]")["[H2O] ppm"].mean().reset_index()
        self.t_h2o = b["Time [ms]"].values * 1e-3
        self.y_h2o = b["[H2O] ppm"].values * 1e-6
        y = np.interp(self.h2o_times, self.t_h2o, self.y_h2o)
        self.logH2O_obs = np.log(np.clip(y + LOG_EPS, NOISE_FLOOR, None))

    def logOH_obs_at(self, tau_us):
        y = np.interp(self.oh_times + tau_us * 1e-6, self.t_oh, self.y_oh)
        return np.log(np.clip(y + LOG_EPS, NOISE_FLOOR, None))

    def log_obs(self, tau_us):
        return np.concatenate([self.logOH_obs_at(tau_us), self.logH2O_obs])

    def target_labels(self):
        return ([f"OH  t={t*1e3:.4f} ms"  for t in self.oh_times] +
                [f"H2O t={t*1e3:.4f} ms" for t in self.h2o_times])

    # ── surrogate ─────────────────────────────────────────────────────────────
    def nn_log(self, x):
        if not getattr(self, "use_surrogate", True):
            return self.cantera_log_targets(x)
        with torch.no_grad():
            xt = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
            return self.model(xt).squeeze(0).numpy().astype(float)

    def nn_jac(self, x):
        if not getattr(self, "use_surrogate", True):
            # central differences on the full model; h is in normalized-x units
            h = 2e-3
            J = np.empty((self.n_tot, self.input_dim))
            for k in range(self.input_dim):
                dx = np.zeros(self.input_dim); dx[k] = h
                J[:, k] = (self.cantera_log_targets(x + dx)
                           - self.cantera_log_targets(x - dx)) / (2 * h)
            return J
        xt = torch.tensor(x, dtype=torch.float32, requires_grad=True).unsqueeze(0)
        J = torch.autograd.functional.jacobian(
            lambda xx: self.model(xx).squeeze(0), xt)
        return J.detach().numpy().reshape(self.n_tot, self.input_dim).astype(float)

    # ── Cantera ───────────────────────────────────────────────────────────────
    def perturb_gas(self, x, cached=True):
        # Every active reaction is rewritten from the stored nominal A/b/Ea on each
        # call, so reusing one Solution cannot accumulate drift. Re-parsing the YAML
        # each time dominates the cost of a surrogate-free fit (~10^3 calls).
        if cached:
            if getattr(self, "_gas_cache", None) is None:
                self._gas_cache = ct.Solution(self.yaml)
            gas = self._gas_cache
        else:
            gas = ct.Solution(self.yaml)
        for j, (name, fall, branches) in enumerate(self.RXN):
            fA  = np.exp(x[2 * j] * self.ln_f)          # common to all branches
            dEa = x[2 * j + 1] * self.sigma_e           # common to all branches
            for (idx, A, b, Ea) in branches:
                r = gas.reaction(idx)
                An, Ean = A * fA, (Ea + dEa) * CAL_PER_MOL
                if fall:
                    r.rate.low_rate = ct.Arrhenius(An, b, Ean)
                else:
                    r.rate = ct.Arrhenius(An, b, Ean)
                gas.modify_reaction(idx, r)
        return gas

    def check_perturbation(self, verbose=True):
        """Verify that each active parameter produces an EXACT multiplicative
        perturbation of the rate constant it is defined on:
            k'(T)/k(T) = exp(x_A*ln_f - x_Ea*sigma_E/RT)

        Two subtleties this check must respect, or it reports false failures:
          * R is Cantera's gas constant, not 1.987. The 4th digit shows up as a
            ~1e-4 deviation in the exponent and looks like a real mismatch.
          * For a FALLOFF reaction the active parameters are the LOW-PRESSURE
            limit, so k0 must be compared directly. forward_rate_constants returns
            the blended effective k, which legitimately does NOT scale with k0
            (raising k0 pushes the reaction toward the unchanged high-P limit).
        """
        R = ct.gas_constant / CAL_PER_MOL          # cal / (mol K)

        def k_of(gas, name_fall, branches, T):
            gas.TP = T, ct.one_atm
            if name_fall:                          # low-pressure limit, P-independent
                tot = 0.0
                for (i, *_rest) in branches:
                    lr = gas.reaction(i).rate.low_rate
                    Ea_c = MOL_UNITS.convert_activation_energy_to(
                        f"{lr.activation_energy} J/kmol", "cal / mol")
                    tot += (lr.pre_exponential_factor * T ** lr.temperature_exponent
                            * np.exp(-Ea_c / (R * T)))
                return tot
            return sum(gas.forward_rate_constants[i] for (i, *_rest) in branches)

        gas0 = ct.Solution(self.yaml)
        worst = 0.0
        for j, (name, fall, branches) in enumerate(self.RXN):
            x = np.zeros(self.input_dim)
            x[2 * j], x[2 * j + 1] = 0.7, -0.4      # arbitrary non-trivial probe
            gas1 = self.perturb_gas(x, cached=False)
            for T in (1192.0, 1398.0):
                k0 = k_of(gas0, fall, branches, T)
                k1 = k_of(gas1, fall, branches, T)
                want = np.exp(x[2*j] * self.ln_f - x[2*j+1] * self.sigma_e / (R * T))
                rel = abs(k1 / k0 / want - 1.0)
                worst = max(worst, rel)
                if verbose:
                    tag = " (k0, low-P limit)" if fall else ""
                    flag = "ok" if rel < 1e-8 else "MISMATCH"
                    print(f"    {name} @{T:.0f} K{tag}  k'/k = {k1/k0:.8f}  "
                          f"expected {want:.8f}  rel dev {rel:.2e}  {flag}")
        if worst > 1e-8:
            raise SystemExit(
                f"\n  perturbation is NOT a clean multiplicative factor "
                f"(worst rel dev {worst:.2e}).\n"
                f"  Most likely only one branch of a duplicate pair is being scaled. "
                f"Fix before optimizing.")
        return worst

    def profile_grids(self):
        """Time grids for the figure, sized to each condition's own data range.
        OH is denser early where the rise and peak live."""
        t_h2o_end = float(self.t_h2o.max())
        t_oh_end  = float(self.t_oh.max())
        t_sim = np.linspace(t_h2o_end / 600.0, t_h2o_end, 600)
        knee  = 0.25 * t_oh_end
        t_oh  = np.concatenate([np.linspace(1e-7, knee, 500),
                                np.linspace(knee * 1.001, t_oh_end, 200)])
        return t_sim, t_oh

    def run_profiles(self, x, t_sim, t_oh):
        """H2O on t_sim and OH on t_oh from one Cantera integration."""
        gas = self.perturb_gas(x)
        gas.TPX = self.T, self.P, self.X0
        rr  = ct.IdealGasConstPressureReactor(gas, energy="on")
        net = ct.ReactorNet([rr])
        hi, oi = gas.species_index("H2O"), gas.species_index("OH")
        allt  = np.concatenate([t_sim, t_oh])
        order = np.argsort(allt, kind="stable")
        hv = np.empty(len(allt)); ov = np.empty(len(allt))
        for k, t in enumerate(allt[order]):
            net.advance(float(t))
            ph = rr.phase if hasattr(rr, "phase") else rr.thermo
            hv[order[k]] = ph.X[hi]; ov[order[k]] = ph.X[oi]
        n = len(t_sim)
        return hv[:n], ov[n:]

    def cantera_log_targets(self, x):
        """log OH at oh_times and log H2O at h2o_times, from the full model."""
        gas = self.perturb_gas(x)
        gas.TPX = self.T, self.P, self.X0
        rr  = ct.IdealGasConstPressureReactor(gas, energy="on")
        net = ct.ReactorNet([rr])
        oi, hi = gas.species_index("OH"), gas.species_index("H2O")
        allt  = np.concatenate([self.oh_times, self.h2o_times])
        order = np.argsort(allt, kind="stable")
        ov = np.empty(len(allt)); hv = np.empty(len(allt))
        for k, t in enumerate(allt[order]):
            net.advance(float(t))
            ov[order[k]] = rr.thermo.X[oi]
            hv[order[k]] = rr.thermo.X[hi]
        n = self.n_oh
        return np.concatenate([
            np.log(np.clip(ov[:n], 1e-30, None)),
            np.log(np.clip(hv[n:], 1e-30, None))])


def assert_compatible(conds):
    """x must mean the same thing in every surrogate, or the joint fit is nonsense."""
    ref = conds[0]
    for c in conds[1:]:
        assert c.param_names == ref.param_names, (
            f"param order differs:\n  {ref.key}: {ref.param_names}\n  {c.key}: {c.param_names}")
        assert c.input_dim == ref.input_dim, "input_dim differs"
        assert abs(c.ln_f - ref.ln_f) < 1e-10, (
            f"LN_F differs ({ref.ln_f} vs {c.ln_f}): x is a different variable in each net")
        assert abs(c.sigma_e - ref.sigma_e) < 1e-6, (
            f"SIGMA_E differs ({ref.sigma_e} vs {c.sigma_e}): x is a different variable")
        assert [r[0] for r in c.RXN] == [r[0] for r in ref.RXN], "reaction set differs"
    print(f"Compatibility OK: shared x = {ref.param_names}, "
          f"LN_F={ref.ln_f:.6f}, SIGMA_E={ref.sigma_e:.0f} cal/mol")


# ── Residual / Jacobian ───────────────────────────────────────────────────────
def make_residual(fit_conds):
    """
    z = [x (NX), tau_0 ... tau_{NT-1}]

    r = [ per-condition (log y_pred - log y_obs)/SIG_LOG, masked ]   <- data
        [ x / SIG_PRIOR_X ]                                          <- prior, ONCE
        [ tau_c / TAU_PRIOR_US for each c ]                          <- tau prior

    least_squares' .cost = 0.5*||r||^2, so 2*cost = Zhang Eq. 8 with lambda = 4.
    """
    NX  = fit_conds[0].input_dim
    act = fit_conds[0].active_idx                          # free components of x
    NA  = len(act)
    def expand(za):
        x = np.zeros(NX); x[act] = za; return x            # frozen entries stay at 0
    tau_conds = [c for c in fit_conds if c.use_tau]       # only these get a tau
    NT = len(tau_conds)
    slot = {c.key: i for i, c in enumerate(tau_conds)}

    def tau_of(c, z):
        """tau for condition c; exactly 0 (no free parameter) if frozen."""
        return float(z[NA + slot[c.key]]) if c.use_tau else 0.0

    def residual(z):
        x = expand(z[:NA])
        parts = []
        for c in fit_conds:
            r = (c.nn_log(x) - c.log_obs(tau_of(c, z))) / SIG_LOG
            parts.append(r[c.mask])
        parts.append(x[act] / SIG_PRIOR_X)                  # exactly once, active only
        if NT:
            parts.append(z[NA:NA + NT] / TAU_PRIOR_US)
        return np.concatenate(parts)

    def jac(z):
        x = expand(z[:NA])
        rows = []
        h = max(1e-3, 0.01 * TAU_PRIOR_US)
        for c in fit_conds:
            J = c.nn_jac(x)[:, act] / SIG_LOG               # (n_tot, NA)
            Tcol = np.zeros((c.n_tot, NT))
            if c.use_tau:
                t = tau_of(c, z)
                dobs = (c.logOH_obs_at(t + h) - c.logOH_obs_at(t - h)) / (2 * h)
                Tcol[:c.n_oh, slot[c.key]] = -dobs / SIG_LOG  # tau shifts OH only
            rows.append(np.hstack([J, Tcol])[c.mask])
        rows.append(np.hstack([np.eye(NA) / SIG_PRIOR_X, np.zeros((NA, NT))]))
        if NT:
            rows.append(np.hstack([np.zeros((NT, NA)), np.eye(NT) / TAU_PRIOR_US]))
        return np.vstack(rows)

    return residual, jac, tau_conds, act, expand


def solve(fit_conds, n_restart=16, seed=0, verbose=True, x0=None):
    res, jac, tau_conds, act, expand = make_residual(fit_conds)
    NA, NT = len(act), len(tau_conds)
    NZ = NA + NT
    lo = np.array([-1.0] * NA + [-TAU_BOUND_US] * NT)
    hi = np.array([+1.0] * NA + [+TAU_BOUND_US] * NT)

    z_start = np.zeros(NZ)
    if x0 is not None:
        z_start[:NA] = np.asarray(x0, float)[act]
    starts = [z_start]
    if n_restart > 1:
        u = qmc.Sobol(d=NZ, scramble=True, seed=seed).random(n_restart - 1)
        for row in u:
            z0 = np.empty(NZ)
            z0[:NA] = -0.5 + row[:NA]          # x0 in [-0.5, +0.5]
            z0[NA:] = -2.0 + 4.0 * row[NA:]    # tau0 in [-2, +2] us
            starts.append(z0)

    best, costs = None, []
    for z0 in starts:
        s = least_squares(res, z0, jac=jac, bounds=(lo, hi), method="trf",
                          xtol=1e-12, ftol=1e-12, gtol=1e-12)
        costs.append(s.cost)
        if best is None or s.cost < best.cost:
            best = s
    costs = np.array(costs)

    if verbose:
        spread = costs.max() - costs.min()
        print(f"\nMulti-start: {len(starts)} runs, cost min={costs.min():.6g} "
              f"max={costs.max():.6g} spread={spread:.3g}")
        near = costs < costs.min() * 1.01 + 1e-12
        print(f"  {near.sum()}/{len(costs)} restarts within 1% of the best cost")
        if spread > 0.05 * max(costs.min(), 1e-12) and near.sum() < len(costs):
            print("  NOTE: restarts do not all agree -> either multiple minima or a "
                  "flat ridge. Compare the x* they reach before trusting a single one.")
    return best, costs


# ── F-score (Zhang Eq. 13) ────────────────────────────────────────────────────
def f_scores(cond, x, tau, source="nn"):
    """
    F_m = (y_m,obs - y_m,opt) / (2 sigma_m).   |F_m| > 1 => inconsistent.

    source="nn"      -> y_opt from the surrogate  (contaminated by surrogate error)
    source="cantera" -> y_opt from the full model (the one to cull on)

    Sign convention: F > 0 means the model UNDER-predicts the observation.
    Relative to optimize_joint_generic.py's residual r = (y_pred - y_obs)/SIG_LOG,
    this is simply F = -r/2.
    """
    y_pred = cond.nn_log(x) if source == "nn" else cond.cantera_log_targets(x)
    y_obs  = cond.log_obs(tau)
    return (y_obs - y_pred) / (2.0 * SIG_LOG)


def fit_tau_only(cond, x, source="nn"):
    """For an eval-only (held-out) condition: tau is an experimental nuisance,
    so refit it alone at fixed x rather than pretending it is zero."""
    y_pred = cond.nn_log(x) if source == "nn" else cond.cantera_log_targets(x)

    def r(t):
        return np.concatenate([(y_pred - cond.log_obs(t[0])) / SIG_LOG,
                               [t[0] / TAU_PRIOR_US]])

    s = least_squares(r, [0.0], bounds=([-TAU_BOUND_US], [TAU_BOUND_US]),
                      method="trf")
    return float(s.x[0])


def print_f_table(cond, x, tau, verify_cantera):
    labels = cond.target_labels()
    F_nn = f_scores(cond, x, tau, "nn")
    F_ct = f_scores(cond, x, tau, "cantera") if verify_cantera else None
    print(f"\n  {cond.label}   (tau = {tau:+.3f} us)")
    hdr = f"    {'target':<20}{'F(NN)':>9}"
    if verify_cantera:
        hdr += f"{'F(Cantera)':>12}{'dF(surr)':>10}"
    hdr += "   status"
    print(hdr)
    for k, lab in enumerate(labels):
        used = "" if cond.mask[k] else "  [culled]"
        Fref = F_ct[k] if verify_cantera else F_nn[k]
        flag = "INCONSISTENT" if abs(Fref) > F_THRESHOLD else "ok"
        line = f"    {lab:<20}{F_nn[k]:>+9.3f}"
        if verify_cantera:
            line += f"{F_ct[k]:>+12.3f}{F_ct[k]-F_nn[k]:>+10.3f}"
        print(line + f"   {flag}{used}")
    if verify_cantera:
        d = np.abs(F_ct - F_nn)
        print(f"    surrogate contribution to |F|: max {d.max():.3f}, mean {d.mean():.3f}"
              f"   (threshold is {F_THRESHOLD:.1f})")
        if d.max() > 0.2 * F_THRESHOLD:
            print("    WARNING: surrogate error is a non-negligible fraction of the F "
                  "threshold. Cull on F(Cantera) only.")
    return F_nn, F_ct


def chi2_report(fit_conds, sol, n_x, n_tau):
    """Global goodness-of-fit on the data block alone."""
    r = sol.fun
    n_data = sum(int(c.mask.sum()) for c in fit_conds)
    r_data = r[:n_data]
    S = float(np.sum(r_data ** 2))
    dof_lo = max(n_data - (n_x + n_tau), 1)   # prior uninformative
    dof_hi = n_data                           # prior fully determines x
    print(f"\nGlobal fit:  sum(r_data^2) = {S:.3f} over {n_data} data residuals")
    print(f"  effective dof between {dof_lo} (prior weak) and {dof_hi} (prior strong)")
    print(f"  chi2 95% critical: {chi2.ppf(0.95, dof_lo):.2f} .. "
          f"{chi2.ppf(0.95, dof_hi):.2f}")
    verdict = ("data and model are mutually consistent"
               if S <= chi2.ppf(0.95, dof_hi) else
               "NOT consistent at 95% under either dof convention"
               if S > chi2.ppf(0.95, dof_lo) else
               "borderline -- depends on how informative you call the prior")
    print(f"  -> {verdict}")


def report_posterior(sol, fit_conds, jac, tau_conds):
    NX  = fit_conds[0].input_dim
    act = fit_conds[0].active_idx
    NA, NT = len(act), len(tau_conds)
    J = jac(sol.x)
    cov = np.linalg.inv(J.T @ J)
    # Expand back to the full 8-dim space. Frozen entries: x = 0 and zero
    # covariance -- that IS Zhang Eq. 16, since the conditional covariance of a
    # Gaussian equals the inverse of the corresponding block of the precision.
    Sigma = np.zeros((NX, NX)); Sigma[np.ix_(act, act)] = cov[:NA, :NA]
    x = np.zeros(NX); x[act] = sol.x[:NA]
    sd = np.sqrt(np.diag(Sigma))
    names = fit_conds[0].param_names
    frozen = [k for k in range(NX) if k not in act]
    if frozen:
        print("  frozen at nominal (--freeze): " + ", ".join(names[k] for k in frozen))
    print(f"\nJOINT MAP   2*cost (Zhang Eq. 8) = {2*sol.cost:.6g}")
    print(f"  {'param':<10}{'x*':>9}{'sd':>9}{'physical shift':>22}")
    ln_f, sig_e = fit_conds[0].ln_f, fit_conds[0].sigma_e
    for k, n in enumerate(names):
        phys = (f"A x {np.exp(x[k]*ln_f):.4f}" if k % 2 == 0
                else f"Ea {x[k]*sig_e:+.0f} cal/mol")
        print(f"  {n:<10}{x[k]:>+9.4f}{sd[k]:>9.4f}{phys:>22}")
    for i, c in enumerate(tau_conds):
        t_i, sd_t = sol.x[NA + i], np.sqrt(cov[NA + i, NA + i])
        # Zhang Sec. 3.4, post-hoc: unconstrained iff it did not move AND kept
        # ~prior uncertainty.  chi_x applied in normalized units (see header).
        frozen = (abs(t_i) < CHI_X * TAU_PRIOR_US) and (sd_t > CHI_2SIGMA * TAU_PRIOR_US)
        verdict = "FREEZE (Zhang chi_x & chi_2sigma both met)" if frozen else "live"
        print(f"  tau[{c.key}] = {t_i:+.3f} us  (sd {sd_t:.3f} = "
              f"{sd_t/TAU_PRIOR_US:.1%} of prior)  screen L={c.tau_leverage():.3f} "
              f"vs L_crit={c.tau_leverage_crit():.3f}  -> {verdict}")
    for c in fit_conds:
        if not c.use_tau:
            print(f"  tau[{c.key}] frozen at 0 (--tau)   "
                  f"screen L={c.tau_leverage():.3f} vs L_crit={c.tau_leverage_crit():.3f}")

    # Zhang's freezing test on the RATE parameters -- this is what the criterion
    # was actually written for.  A parameter flagged here contributed nothing and
    # should be frozen to nominal via Eqs. 15-16 (i.e. dropped and refitted).
    x_crit = np.log1p(CHI_X) / ln_f
    print(f"\n  Zhang freezing test on rate params "
          f"(|x| < {x_crit:.4f} and sd > {CHI_2SIGMA*SIG_PRIOR_X:.3f}):")
    any_frozen = False
    for k, n in enumerate(names):
        if abs(x[k]) < x_crit and sd[k] > CHI_2SIGMA * SIG_PRIOR_X:
            print(f"    {n:<10} x={x[k]:+.4f} sd={sd[k]:.4f}  -> FREEZE (unconstrained)")
            any_frozen = True
    if not any_frozen:
        print("    none -- every rate parameter either moved or tightened")

    ev, evec = np.linalg.eigh(Sigma[np.ix_(act, act)])
    order = np.argsort(ev)[::-1]
    print(f"\n  Sigma* condition number: {ev[order[0]]/max(ev[order[-1]],1e-30):.4g}")
    print("  posterior sd along eigen-directions (prior = "
          f"{SIG_PRIOR_X}): {np.round(np.sqrt(ev[order]),4)}")
    for i in order[:3]:
        dom = names[act[int(np.argmax(np.abs(evec[:, i])))]]
        print(f"    sd={np.sqrt(ev[i]):.4f}  dominant: {dom:<9} vec={np.round(evec[:,i],2)}")
    # lnA/Ea correlation per reaction -- the ridge diagnostic
    print("  rho(lnA, Ea):", end=" ")
    for j in range(NX // 2):
        if 2*j not in act or 2*j+1 not in act: continue
        rho = Sigma[2*j, 2*j+1] / np.sqrt(Sigma[2*j, 2*j] * Sigma[2*j+1, 2*j+1])
        print(f"{names[2*j][4:]}={rho:+.3f}", end="  ")
    print()
    return x, Sigma, cov


def make_figure(conds, eval_keys, fit_keys, x, Sigma, taus, out_path, band=True):
    """One row per condition: H2O and OH time histories, Hong data, nominal, MAP x*,
    and the +/-2sigma posterior band propagated through a central-difference Cantera
    Jacobian (same construction as optimize_joint_generic.py).

    Held-out conditions are drawn identically but labelled, because that is exactly
    where the plot is worth looking at -- the MAP curve there is a prediction, not a fit.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.ndimage import uniform_filter1d

    NX = conds[eval_keys[0]].input_dim
    n = len(eval_keys)
    fig, axes = plt.subplots(n, 2, figsize=(16, 6 * n), squeeze=False)

    for row, key in enumerate(eval_keys):
        c = conds[key]
        t_sim, t_oh_grid = c.profile_grids()
        print(f"  {c.label}: Cantera profiles ...", flush=True)
        h2o_nom, oh_nom = c.run_profiles(np.zeros(NX), t_sim, t_oh_grid)
        h2o_opt, oh_opt = c.run_profiles(x,           t_sim, t_oh_grid)

        if band:
            EPS = 1e-3
            Jh = np.zeros((len(t_sim), NX)); Jo = np.zeros((len(t_oh_grid), NX))
            for k in range(NX):
                dx = np.zeros(NX); dx[k] = EPS
                hp, op = c.run_profiles(x + dx, t_sim, t_oh_grid)
                hm, om = c.run_profiles(x - dx, t_sim, t_oh_grid)
                lg = lambda v: np.log(np.clip(v, 1e-30, None))
                Jh[:, k] = (lg(hp) - lg(hm)) / (2 * EPS)
                Jo[:, k] = (lg(op) - lg(om)) / (2 * EPS)
            Jo = uniform_filter1d(Jo, 20, axis=0); Jh = uniform_filter1d(Jh, 20, axis=0)
            hv = np.einsum("ti,ij,tj->t", Jh, Sigma, Jh)
            ov = np.einsum("ti,ij,tj->t", Jo, Sigma, Jo)
            h2o_up, h2o_lo = h2o_opt*np.exp(+2*np.sqrt(hv)), h2o_opt*np.exp(-2*np.sqrt(hv))
            oh_up,  oh_lo  = oh_opt *np.exp(+2*np.sqrt(ov)), oh_opt *np.exp(-2*np.sqrt(ov))

        tau = taus.get(key, 0.0)
        y_oh_t  = np.exp(c.logOH_obs_at(tau))
        y_h2o_t = np.exp(c.logH2O_obs)
        eu = lambda y: y * 1e6 * (np.exp(2 * SIG_LOG) - 1)
        ed = lambda y: y * 1e6 * (1 - np.exp(-2 * SIG_LOG))

        held = key not in fit_keys
        for col, (ax, tg, prof_n, prof_o, df, xcol, ycol, tt, yt, lbl) in enumerate([
            (axes[row][0], t_sim,      h2o_nom, h2o_opt, c.df_h2o,
             "Time [ms]", "[H2O] ppm", c.h2o_times, y_h2o_t, "H2O"),
            (axes[row][1], t_oh_grid,  oh_nom,  oh_opt,  c.df_oh,
             "Time [ms]", "[OH] ppm",  c.oh_times,  y_oh_t,  "OH")]):
            ax.plot(df[xcol], df[ycol], "o", mfc="none", mec="k", ms=4, alpha=0.4,
                    label="Hong et al.")
            ax.errorbar(tt * 1e3, yt * 1e6, yerr=[ed(yt), eu(yt)], fmt="none",
                        ecolor="k", elinewidth=1.5, capsize=4,
                        label=r"targets $\pm 2\sigma_{obs}$")
            ax.plot(tg * 1e3, prof_n * 1e6, "r--", lw=1.5, alpha=0.7, label="Nominal")
            ax.plot(tg * 1e3, prof_o * 1e6, "b-", lw=2.5,
                    label="MAP prediction" if held else "MAP $x^*$")
            if band:
                lo, up = (h2o_lo, h2o_up) if col == 0 else (oh_lo, oh_up)
                ax.fill_between(tg * 1e3, lo * 1e6, up * 1e6,
                                color="steelblue", alpha=0.2, label=r"$\pm 2\sigma$")
            ax.set(xlabel="Time [ms]", ylabel=ycol,
                   xlim=[0, tg.max() * 1e3 * 1.02],
                   title=f"{c.label}  --  {lbl}" +
                         ("   [HELD OUT]" if held else "   [fitted]"))
            ax.legend(frameon=False, fontsize=9)
            ax.grid(True, ls="--", alpha=0.3)

    names = conds[eval_keys[0]].param_names
    ln_f, sig_e = conds[eval_keys[0]].ln_f, conds[eval_keys[0]].sigma_e
    sub = "  |  ".join(
        f"{names[2*j][4:]}: Ax{np.exp(x[2*j]*ln_f):.3f}, Ea{x[2*j+1]*sig_e:+.0f}"
        for j in range(len(names) // 2))
    ttl = ("tau: " + ", ".join(f"{k}={v:+.2f}us" for k, v in taus.items())) if taus else ""
    fig.suptitle(f"Joint MAP -- fit on {','.join(fit_keys)} K   {ttl}\n{sub}", fontsize=10)
    plt.tight_layout(); plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit",  default="1192,1398",
                    help="conditions entering the objective")
    ap.add_argument("--eval", default=None,
                    help="conditions to report F for (default: same as --fit). "
                         "Conditions in --eval but not --fit are HELD OUT.")
    ap.add_argument("--verify_cantera", action="store_true",
                    help="also compute F from a full Cantera run at x* (slow, "
                         "but required before culling)")
    ap.add_argument("--cull", action="store_true",
                    help="Zhang's iterative inconsistent-target removal")
    ap.add_argument("--tau", default="auto",
                    help="which conditions carry an OH-trigger nuisance tau: "
                         "'auto' (keep only where the pre-screen L >= L_crit "
                         "derived from Zhang's chi_2sigma), "
                         "'all', 'none', or a comma list e.g. '1398'")
    ap.add_argument("--yaml", default=YAML_FILE,
                    help="mechanism file; must match what the surrogates were trained on")
    ap.add_argument("--result", default=None,
                    help="override surrogate files, e.g. "
                         "'1192=result_a.pt,1398=result_b.pt'. Must be the RESULT_PATH "
                         "files (model_state, param_names, ...), NOT the ckpt_*.pt "
                         "sample-generation checkpoints, which contain no model.")
    ap.add_argument("--ckpt", default=None, help=argparse.SUPPRESS)   # old name
    ap.add_argument("--freeze", default="",
                    help="freeze these reactions' lnA and Ea at nominal, e.g. 'R5' or "
                         "'R5,R4'. Zhang Eqs. 15-16. Use to test whether a parameter "
                         "earns its place: freeze it and see how much the DATA term "
                         "of the objective rises.")
    ap.add_argument("--no_surrogate", action="store_true",
                    help="replace the NN with direct Cantera integration in the "
                         "objective. This is the ONLY check on x* itself: "
                         "--verify_cantera validates the forward prediction at a "
                         "given x*, not the x* the surrogate produced. Slow; pair "
                         "with --x0 and a small --restarts.")
    ap.add_argument("--x0", default=None,
                    help="JSON result file whose x_opt is used as the starting point")
    ap.add_argument("--verify_setup", action="store_true",
                    help="preflight only: compatibility + duplicate-branch perturbation check")
    ap.add_argument("--restarts", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="joint_multicondition_result.json")
    ap.add_argument("--fig", default=None,
                    help="write the OH/H2O comparison figure here "
                         "(default: fig_joint_<fit keys>.png; use 'none' to skip)")
    ap.add_argument("--no_band", action="store_true",
                    help="skip the +/-2sigma band (saves 2*INPUT_DIM Cantera runs per condition)")
    args = ap.parse_args()

    fit_keys  = [k.strip() for k in args.fit.split(",") if k.strip()]
    eval_keys = ([k.strip() for k in args.eval.split(",")] if args.eval
                 else list(fit_keys))
    all_keys  = list(dict.fromkeys(fit_keys + eval_keys))

    override = args.result or args.ckpt
    if args.ckpt and not args.result:
        print("note: --ckpt is the old name, use --result")
    if override:
        for item in override.split(","):
            k, path = item.split("=", 1)
            COND_LIB[k.strip()]["result"] = path.strip()
    conds = {k: Condition(k, yaml_file=args.yaml) for k in all_keys}
    froz = {t.strip().upper() for t in args.freeze.split(",") if t.strip()}
    for c in conds.values():
        c.use_surrogate = not args.no_surrogate
        rn = [r[0] for r in c.RXN]
        for r in froz:
            assert r in rn, f"--freeze {r}: not one of {rn}"
        c.active_idx = [k for k in range(c.input_dim) if rn[k // 2] not in froz]
        assert c.active_idx, "--freeze left no free parameters"
    if args.no_surrogate:
        print("\n*** SURROGATE OFF: objective evaluated by direct Cantera integration.")
    fit_conds = [conds[k] for k in fit_keys]
    assert_compatible(list(conds.values()))
    print(f"\nmechanism: {args.yaml}")
    for k, c in conds.items():
        print(f"  checkpoint[{k}] = {c.result_path}")
    print("\nmechanism content check:")
    for c in conds.values():
        for name, fall, branches in c.RXN:
            if name == "R2":
                nb = len(branches)
                gg = ct.Solution(args.yaml)
                ks = []
                for T in (1192.0, 1398.0):
                    gg.TP = T, ct.one_atm
                    ks.append(sum(gg.forward_rate_constants[i] for i, *_ in branches))
                print(f"  {c.label}: R2 has {nb} branch(es), "
                      f"k2(1192)={ks[0]:.4g}  k2(1398)={ks[1]:.4g}")
                if nb < 2:
                    print("    *** WARNING: R2 is SINGLE-BRANCH. This is the UNCORRECTED "
                          "mechanism.\n    *** If the surrogates were trained with the "
                          "restored R26 duplicate, --yaml is wrong.")
        break

    print("\nperturbation check (k'/k must be an exact multiplicative factor):")
    for c in conds.values():
        print(f"  {c.label}")
        c.check_perturbation()
    print("  all reactions OK")

    # End-to-end wiring check: at x = 0 the surrogate and the full model must agree
    # to within the surrogate's own Set-1 accuracy. A large gap means --yaml is not
    # the mechanism the surrogate was trained on, or the target times are stale.
    print("\nsurrogate vs Cantera at x = 0 (catches a mechanism/target mismatch):")
    bad = False
    for c in conds.values():
        z = np.zeros(c.input_dim)
        gap = np.abs(np.exp(c.nn_log(z) - c.cantera_log_targets(z)) - 1.0) * 100
        print(f"  {c.label}: " + "  ".join(f"{g:.2f}%" for g in gap)
              + f"   (max {gap.max():.2f}%)")
        if gap.max() > 10.0:
            bad = True
    if bad:
        print("  *** WARNING: >10% disagreement at nominal. Check that --yaml matches the\n"
              "  *** training mechanism and that the result file is the right one.")
    else:
        print("  consistent")
    if args.verify_setup:
        return

    # ── tau selection ─────────────────────────────────────────────────────────
    print("\ntau pre-screen (a priori forecast of Zhang's post-hoc chi_2sigma test):")
    for c in conds.values():
        lv, lc = c.tau_leverage(), c.tau_leverage_crit()
        print(f"  {c.label:<18} L={lv:6.3f}  L_crit={lc:.3f}   "
              f"{'identifiable' if lv >= lc else 'NOT identifiable'}")
    if args.tau == "all":
        keep = set(all_keys)
    elif args.tau == "none":
        keep = set()
    elif args.tau == "auto":
        keep = {k for k, c in conds.items() if c.tau_leverage() >= c.tau_leverage_crit()}
    else:
        keep = {k.strip() for k in args.tau.split(",") if k.strip()}
    for k, c in conds.items():
        c.use_tau = k in keep
    print(f"  --tau {args.tau} -> tau active for: {sorted(keep) if keep else 'none'}")
    if not keep:
        print("  NOTE: with no tau anywhere, a time-zero offset that differs between "
              "conditions can only be absorbed as an apparent temperature dependence, "
              "i.e. into Ea. Run once with --tau all and compare x*.")

    NX = fit_conds[0].input_dim
    NT = sum(1 for c in fit_conds if c.use_tau)
    NAn = len(fit_conds[0].active_idx)
    n_data = sum(c.n_tot for c in fit_conds)
    print(f"\nFit conditions: {fit_keys}   ({n_data} data residuals, "
          f"{NAn} free rate params + {NT} tau = {NAn+NT} unknowns"
          + (f"; {NX-NAn} frozen" if NAn < NX else "") + ")")
    if n_data <= NAn + NT:
        print("  WARNING: data residuals <= unknowns. The fit is determined by the "
              "prior, and the F-score test has essentially no power here.")
    held = [k for k in eval_keys if k not in fit_keys]
    if held:
        print(f"HELD OUT (not in objective, F reported): {held}")

    X0 = None
    if args.x0:
        with open(args.x0) as f:
            X0 = np.asarray(json.load(f)["x_opt"], float)
        print(f"starting from x_opt in {args.x0}: {np.round(X0, 4)}")

    _, jac, tau_conds, act, _exp = make_residual(fit_conds)
    NA = len(act)
    tau_slot = {c.key: i for i, c in enumerate(tau_conds)}

    for it in range(20):
        sol, _ = solve(fit_conds, n_restart=args.restarts, seed=args.seed, x0=X0)
        x, Sigma, cov = report_posterior(sol, fit_conds, jac, tau_conds)
        chi2_report(fit_conds, sol, NAn, NT)

        print("\n" + "=" * 72)
        print(f"  F-SCORE CONSISTENCY TEST  (Zhang Eq. 13, |F| > {F_THRESHOLD} "
              f"= inconsistent)" + (f"   [cull pass {it+1}]" if args.cull else ""))
        print("=" * 72)
        F_store, tau_used, worst, worst_ref = {}, {}, 0.0, None
        for k in eval_keys:
            c = conds[k]
            if not c.use_tau:
                tau = 0.0                                   # frozen
            elif k in fit_keys:
                tau = float(sol.x[NA + tau_slot[k]])
            else:
                tau = fit_tau_only(c, x, "cantera" if args.verify_cantera else "nn")
            F_nn, F_ct = print_f_table(c, x, tau, args.verify_cantera)
            tau_used[k] = tau
            F_store[k] = dict(tau_us=tau, F_nn=F_nn.tolist(),
                              F_cantera=(F_ct.tolist() if F_ct is not None else None),
                              labels=c.target_labels(), mask=c.mask.tolist())
            if k in fit_keys:
                Fref = F_ct if args.verify_cantera else F_nn
                live = np.where(c.mask, np.abs(Fref), -np.inf)
                if live.max() > worst:
                    worst, worst_ref = live.max(), (k, int(np.argmax(live)))

        if not args.cull or worst <= F_THRESHOLD:
            if args.cull and worst <= F_THRESHOLD:
                print(f"\nAll fitted targets consistent (max |F| = {worst:.3f}). "
                      f"Cull converged after {it} removal(s).")
            break

        k, idx = worst_ref
        c = conds[k]
        remaining = sum(int(cc.mask.sum()) for cc in fit_conds) - 1
        _unk = NAn + NT
        print(f"\n  Worst: {c.label}  {c.target_labels()[idx]}  |F| = {worst:.3f}")
        if remaining <= _unk:
            print("  REFUSING to cull further: removing this target would leave "
                  f"{remaining} residuals for {_unk} unknowns, i.e. a prior-determined "
                  "fit. Stop and treat the inconsistency as a data-model problem "
                  "(add a nuisance parameter) rather than deleting the datum.")
            break
        c.mask[idx] = False
        print(f"  Culled. {remaining} targets remain. Re-optimizing ...")

    out = dict(fit=fit_keys, eval=eval_keys, surrogate=not args.no_surrogate,
               frozen=sorted(froz),
               mechanism=args.yaml,
               param_names=fit_conds[0].param_names,
               x_opt=x.tolist(), Sigma_star=Sigma.tolist(),
               cost_2x=float(2 * sol.cost), f_scores=F_store,
               ln_f=fit_conds[0].ln_f, sigma_e=fit_conds[0].sigma_e,
               sig_log=SIG_LOG, sig_prior_x=SIG_PRIOR_X, lam=1.0 / SIG_PRIOR_X ** 2,
               surrogates={k: conds[k].result_path for k in all_keys})
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {args.out}")

    figpath = args.fig or f"fig_joint_{'_'.join(fit_keys)}.png"
    if str(figpath).lower() != "none":
        print("\nBuilding comparison figure ...")
        make_figure(conds, eval_keys, fit_keys, x, Sigma, tau_used,
                    figpath, band=not args.no_band)


if __name__ == "__main__":
    main()
