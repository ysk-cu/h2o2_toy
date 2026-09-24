#!/usr/bin/env python
# coding: utf-8
#
# check_r26_duplicate.py
#
# Surrogate-free, optimization-free test of one question:
#
#   Is the irreducible inter-condition residual D caused by the missing
#   duplicate branch of R26 (H2O2 + OH <=> HO2 + H2O)?
#
# WHY D AND NOT "how much did OH change"
# --------------------------------------
# Restoring the branch raises k2 and lowers OH at BOTH conditions.  A change
# that is common to both temperatures is absorbable by a single A-factor, so it
# cannot explain a hold-out that fails in OPPOSITE directions at the two
# conditions.  Only the DIFFERENCE between conditions is diagnostic:
#
#     r(T) = ln[ OH_Hong(t_target) / OH_model(t_target) ]
#     D    = r(1192 K) - r(1398 K)
#
# From the two hold-out runs, D = -0.204 (at x* from the 1192 K fit) and
# -0.191 (at x* from the 1398 K fit).  Two quite different parameter vectors,
# essentially the same D -> D is not reachable by the 8 active parameters.
#
# PASS  : |D| < 4*sigma_obs = 0.20.  NOT 0.10.  An optimal common A-factor shift
#         splits D evenly between the two conditions, leaving |D|/2 at each, so
#         consistency needs |D|/2 < 2*sigma_obs.  The script reports the residuals
#         after that shift, which is the number to judge.
# FAIL  : OH moves a lot but D stays near -0.19  -> the duplicate is not the cause
# OVERSHOOT : D crosses zero -> the duplicate is part of it but not all of it
#
# Everything is evaluated at NOMINAL x = 0.  Do not re-optimize: the point is
# whether the structural residual disappears, not whether a fit can hide it.

import numpy as np
import pandas as pd
import cantera as ct

YAML_IN  = "chem_cti_toy_model_og.yaml"
YAML_OUT = "chem_cti_toy_model_r26dup.yaml"

# ── The restored low-T branch ────────────────────────────────────────────────
# VERIFY THESE AGAINST YOUR OWN SOURCE before trusting the result.
# The commonly used duplicate pair for H2O2 + OH (GRI-Mech / Hong et al. 2010) is
#     7.59e13 exp(-7269/RT)   and   1.7e12 exp(-318/RT)      [cm3/mol/s, cal/mo#     1.70e12 exp(-318/RT)
# Note exp(-318/RT) means Ea = +318 cal/mol (nearly flat), NOT -318.  The two# Note exp(-160/RT) means Ea = +160 cal/mol (nearly flat), NOT -160.  The two
# candidates differ enough to matter here, so run both.
DUP_A, DUP_B, DUP_EA = 1.74e12, 0.0, 160.0     # try also (1.7e12, 0.0, 318.0)


CONDS = {
    "1192": dict(T=1192.0, P_atm=1.95,
                 X0={"H2O2": 2220e-6, "H2O": 1360e-6, "O2": 680e-6},
                 oh_csv="hong_1192K_oh.csv",
                 targets=[775.2e-6, 797.4e-6]),      # the two failing OH targets
    "1398": dict(T=1398.0, P_atm=1.91,
                 X0={"H2O2": 2540e-6, "H2O": 1234e-6, "O2": 617e-6},
                 oh_csv="hong_1398K_oh.csv",
                 targets=[48.9e-6, 98.3e-6]),
}
SIG_LOG = 0.05


def write_dup_yaml():
    """Insert the second R26 branch by exact text substitution, leaving the rest
    of the mechanism byte-identical."""
    src = open(YAML_IN).read()
    old = ("- equation: H2O2 + OH <=> HO2 + H2O  # Reaction 26\n"
           "  rate-constant: {A: 7.59e+13, b: 0.0, Ea: 7270.0}\n"
           "  # duplicate: true\n")
    if old not in src:
        raise SystemExit("R26 block not found verbatim -- check YAML_IN formatting")
    new = ("- equation: H2O2 + OH <=> HO2 + H2O  # Reaction 26a (high-T branch)\n"
           "  rate-constant: {A: 7.59e+13, b: 0.0, Ea: 7270.0}\n"
           "  duplicate: true\n"
           f"- equation: H2O2 + OH <=> HO2 + H2O  # Reaction 26b (low-T branch, restored)\n"
           f"  rate-constant: {{A: {DUP_A:.3e}, b: {DUP_B}, Ea: {DUP_EA}}}\n"
           "  duplicate: true\n")
    open(YAML_OUT, "w").write(src.replace(old, new))
    print(f"wrote {YAML_OUT}")


def r26_indices(gas):
    """Match by reactant/product composition.  Do NOT compare equation strings:
    Cantera canonicalizes species order, so 'HO2 + H2O' is rendered as
    'H2O + HO2' and a string match silently returns an empty list."""
    want_r, want_p = {"H2O2", "OH"}, {"HO2", "H2O"}
    idx = [i for i, r in enumerate(gas.reactions())
           if set(r.reactants) == want_r and set(r.products) == want_p]
    if not idx:
        raise SystemExit("R26 not found by composition -- check species names")
    return idx


def run_oh(yaml_file, cfg, times, k2_scale=1.0):
    """OH mole fraction at the requested times, nominal params, k2 optionally scaled."""
    gas = ct.Solution(yaml_file)
    if k2_scale != 1.0:
        for i in r26_indices(gas):                  # scale ALL branches together
            rx = gas.reaction(i)
            a = rx.rate
            rx.rate = ct.Arrhenius(a.pre_exponential_factor * k2_scale,
                                   a.temperature_exponent, a.activation_energy)
            gas.modify_reaction(i, rx)
    X = dict(cfg["X0"]); X["AR"] = 1.0 - sum(cfg["X0"].values())
    gas.TPX = cfg["T"], cfg["P_atm"] * ct.one_atm, X
    r = ct.IdealGasConstPressureReactor(gas, energy="on", clone=False)
    net = ct.ReactorNet([r])
    oi = gas.species_index("OH")
    out = []
    for t in sorted(times):
        net.advance(float(t))
        out.append((r.phase if hasattr(r,"phase") else r.thermo).X[oi])
    order = np.argsort(np.argsort(times))
    return np.array(out)[order]


def hong_oh(cfg, times):
    df = pd.read_csv(cfg["oh_csv"], skipinitialspace=True)
    g = df.groupby("Time [ms]")["[OH] ppm"].mean().reset_index()
    return np.interp(np.asarray(times), g["Time [ms]"] * 1e-3, g["[OH] ppm"] * 1e-6)


def k2_at(yaml_file, T):
    gas = ct.Solution(yaml_file); gas.TP = T, ct.one_atm
    return sum(gas.forward_rate_constants[i] for i in r26_indices(gas))


def report(yaml_file, label):
    print(f"\n=== {label} ===")
    print(f"  k2(1192 K) = {k2_at(yaml_file,1192.):.4g}   "
          f"k2(1398 K) = {k2_at(yaml_file,1398.):.4g}   "
          f"[cm^3/mol/s]")
    r_mean, detail = {}, {}
    for key, cfg in CONDS.items():
        t = cfg["targets"]
        oh = run_oh(yaml_file, cfg, t)
        obs = hong_oh(cfg, t)
        r = np.log(obs / oh)
        # local sensitivity d lnOH / d ln k2, central difference on a 2% scale
        up = run_oh(yaml_file, cfg, t, 1.02)
        dn = run_oh(yaml_file, cfg, t, 0.98)
        S = (np.log(up) - np.log(dn)) / (np.log(1.02) - np.log(0.98))
        detail[key] = (t, obs, oh, r, S)
        r_mean[key] = float(np.mean(r))
        print(f"  {key} K")
        for j, tt in enumerate(t):
            print(f"    t={tt*1e6:7.1f} us   OH_obs {obs[j]*1e6:7.1f} ppm   "
                  f"OH_model {oh[j]*1e6:7.1f} ppm   r={r[j]:+.4f}   "
                  f"|F|={abs(r[j])/(2*SIG_LOG):.2f}   dlnOH/dlnk2={S[j]:+.3f}")
        print(f"    mean r = {r_mean[key]:+.4f}")
    D = r_mean["1192"] - r_mean["1398"]
    print(f"\n  D = r(1192) - r(1398) = {D:+.4f}      "
          f"(consistency needs |D| < {4*SIG_LOG:.2f})")
    c = -(r_mean["1192"] + r_mean["1398"]) / 2.0
    a, b = r_mean["1192"] + c, r_mean["1398"] + c
    print(f"  after optimal common A-factor shift ({c:+.4f}): "
          f"|F|(1192)={abs(a)/(2*SIG_LOG):.2f}  |F|(1398)={abs(b)/(2*SIG_LOG):.2f}"
          f"  -> {'BOTH CONSISTENT' if max(abs(a),abs(b))<2*SIG_LOG else 'both inconsistent'}")
    return D, detail


if __name__ == "__main__":
    write_dup_yaml()
    D0, _ = report(YAML_IN,  "CURRENT mechanism (single-Arrhenius R26)")
    D1, _ = report(YAML_OUT, "R26 DUPLICATE RESTORED")

    print("\n" + "=" * 68)
    print(f"  D before = {D0:+.4f}")
    print(f"  D after  = {D1:+.4f}")
    print(f"  closed   = {abs(D0)-abs(D1):+.4f}  ({100*(1-abs(D1)/max(abs(D0),1e-12)):+.0f}%)")
    if abs(D1) < 4*SIG_LOG:
        v = ("PASS -- a common A-factor shift now puts both conditions inside |F|<1. "
             "The missing branch was the cause. Retrain both surrogates on "
             f"{YAML_OUT} and redo the hold-outs.")
    elif abs(D1) < 0.6 * abs(D0):
        v = ("PARTIAL -- most of D is the missing branch, but a residual "
             "temperature-dependence error remains. Fix the mechanism, then look "
             "for a second structural term.")
    elif D0 * D1 < 0:
        v = ("OVERSHOOT -- D crossed zero. The branch matters but the magnitude "
             "is wrong; check DUP_A/DUP_EA against your source and try the "
             "1.7e12 / 318 cal/mol variant.")
    else:
        v = ("FAIL -- D barely moved. OH may have changed a lot, but not "
             "DIFFERENTIALLY between conditions, so R26 is not the cause. "
             "Look elsewhere for the temperature-dependence defect.")
    print("  " + v)
    print("=" * 68)
    print("\nNOTE for retraining: inserting a reaction at position 26 shifts every\n"
          "reaction index above it by +1. Your checkpoints hardcode idx_r2=25;\n"
          "after the edit k2 lives in TWO reactions and the active parameter must\n"
          "perturb both branches together (as done in run_oh here), or you are\n"
          "only optimizing the high-T half of k2.")
