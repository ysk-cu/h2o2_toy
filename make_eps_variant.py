"""make_eps_variant.py -- regenerate nn_toy_train_lowP_eps.py from nn_toy_train_bothlimits.py.
The eps variant is the bothlimits script plus the eps_Ar knob; edit bothlimits and rerun this.
Every replacement must match exactly once."""
import sys
src = open('nn_toy_train_bothlimits.py').read()
R = [
("""# nn_toy_train_bothlimits.py
# Joint surrogate (single NN), active reactions k1/k2/k5 + k4 (HO2+OH, Burke R4)
# (+ k3, HO2 + HO2 <=> H2O2 + O2, with the 'r3' argument)
# TRUE joint D-optimal target selection, FIXED time windows
#
# VARIANT 1 -- R1 (H2O2 (+M) <=> 2 OH (+M), falloff) is perturbed on BOTH limits:
#   A0 and A_inf get the same factor exp(x_A ln f), Ea0 and Ea_inf the same shift
#   x_Ea * SIGMA_E.  P_r and the Troe factor are unchanged, so k1 is scaled by the
#   same factor at every pressure (the falloff shape is not perturbed).
#   Previous scripts (nn_toy_*_train_final.py) scaled the low-pressure limit only.
#
# R3 (optional) -- the duplicate pair 19/20 gets the SAME A factor and the SAME Ea
#   shift on both branches (as R2), inputs 8 and 9.
""",
"""# nn_toy_train_lowP_eps.py
# Joint surrogate (single NN), active reactions k1/k2/k5 + k4 (HO2+OH, Burke R4)
# (+ k3, HO2 + HO2 <=> H2O2 + O2, with the 'r3' argument)
# TRUE joint D-optimal target selection, FIXED time windows
#
# VARIANT 2 -- variant 1 (R1 A and Ea perturbed on BOTH limits together) PLUS an
#   independent low-pressure knob: the Ar collision efficiency of R1,
#       eps_Ar = eps_Ar,0 * F_EPS ** x_eps        (last input: index 8, or 10 with R3)
#   It moves [M]_eff and hence P_r, so it changes k1 in the falloff / low-P regime
#   only.  Ar is ~97 % of [M] in the Hong et al. mixtures.
#
#   Cantera (2.6 - 3.2) ignores efficiency changes made through modify_reaction(),
#   so every sample builds a new Solution with the edited R1 (r1_falloff.py).
#
#   At 1192 / 1398 K and ~1.9 atm, d ln k1 / d ln eps_Ar = 0.93 vs 1.00 for the A
#   factor: the data can only fix their combination.  Expect the difference
#   direction to stay at the prior unless targets at other pressures are added
#   (the Kappel et al. conditions at 3.0 - 3.6 atm).
#
# R3 (optional) -- the duplicate pair 19/20 gets the SAME A factor and the SAME Ea
#   shift on both branches (as R2), inputs 8 and 9; the eps knob moves to 10.
"""),
("""# usage:   python nn_toy_train_bothlimits.py 1192            (R1, R2, R5, R4)
#          python nn_toy_train_bothlimits.py 1398 r3         (R1, R2, R5, R4, R3)""",
"""# usage:   python nn_toy_train_lowP_eps.py 1192            (R1, R2, R5, R4, eps)
#          python nn_toy_train_lowP_eps.py 1398 r3         (R1, R2, R5, R4, R3, eps)"""),
("""AT_DATA   = 'dp' in FLAGS or WITH_TPM   # target times restricted to measured data times
""",
"""AT_DATA   = 'dp' in FLAGS or WITH_TPM   # target times restricted to measured data times

# Low-pressure knob: eps = eps0 * exp(x * LN_F_EPS) = eps0 * 2**x,
# so |x| <= 1 spans eps0/2 .. 2*eps0 (uncertainty factor f = 2)
EPS_SPECIES = 'AR'
LN_F_EPS    = np.log(2.0)
"""),
("""R1 = R1Falloff(YAML_FILE, IDX_R1)
_hi = R1.high
print(f'R1 (k1): {R1.equation}   both limits perturbed together')""",
"""R1 = R1Falloff(YAML_FILE, IDX_R1, eps_species=EPS_SPECIES)
_hi = R1.high
print(f'R1 (k1): {R1.equation}   both limits perturbed together, '
      f'eps_{EPS_SPECIES} = {R1.eps0:g} x {np.exp(LN_F_EPS):g}^x')"""),
("""    PARAM_NAMES += ['lnA_R3', 'Ea_R3']   # inputs 8, 9
INPUT_DIM   = len(PARAM_NAMES)""",
"""    PARAM_NAMES += ['lnA_R3', 'Ea_R3']   # inputs 8, 9
# EPS_SPECIES and LN_F_EPS (the efficiency knob) are set at the top of the file
IDX_EPS     = len(PARAM_NAMES)  # position of the efficiency knob in x (always last)
PARAM_NAMES += [f'lnEps_{EPS_SPECIES}_R1']
INPUT_DIM   = len(PARAM_NAMES)"""),
("""RUN_TAG         = ('bothlim' + ('_r3' if WITH_R3 else '') + ('_tpm' if WITH_TPM""",
"""RUN_TAG         = ('eps' + ('_r3' if WITH_R3 else '') + ('_tpm' if WITH_TPM"""),
("""S = {o: np.vstack([(p[o] - _nom[o]) / _nom[o] / d for p in _pert]).T for o in OBS}   # (N_FINE, 4 or 5)
""",
"""S = {o: np.vstack([(p[o] - _nom[o]) / _nom[o] / d for p in _pert]).T for o in OBS}   # (N_FINE, 4 or 5)
# The eps_Ar knob is deliberately NOT a column of the D-optimal design: at this
# pressure it acts only through k1, so its sensitivity row is ~0.93 x the k1 row at
# every time. Adding it would make S^T S almost singular and the greedy det() choice
# would chase numerical noise. Targets are chosen on the rate knobs as in variant 1.
"""),
("""        gas = ct.Solution(YAML_FILE)
        # R1 (k1) — falloff, BOTH limits: same A factor, same Ea shift
        R1.apply_rate(gas, fA=np.exp(x_vec[0] * LN_F), dEa_cal=x_vec[1] * SIGMA_E)""",
"""        # R1 (k1) — falloff, BOTH limits (same A factor, same Ea shift) and the
        # Ar efficiency. A NEW Solution is built: modify_reaction() would store the
        # efficiency but the kinetics would keep using the old one.
        gas = R1.build_solution(fA=np.exp(x_vec[0] * LN_F),
                                dEa_cal=x_vec[1] * SIGMA_E,
                                eps_factor=np.exp(x_vec[IDX_EPS] * LN_F_EPS))"""),
("""{INPUT_DIM} inputs (k1 both limits,k2,k5,k4{",k3" if WITH_R3 else ""}, TRUE JOINT D-OPT)""",
"""{INPUT_DIM} inputs (k1 both limits,k2,k5,k4{",k3" if WITH_R3 else ""} + eps_{EPS_SPECIES}, TRUE JOINT D-OPT)"""),
("""    'eps':              None,             # no efficiency knob in this variant""",
"""    'eps':              dict(species=EPS_SPECIES, ln_f_eps=float(LN_F_EPS),
                             eps0=R1.eps0, index=IDX_EPS, reaction=IDX_R1),"""),
]
for a, b in R:
    n = src.count(a)
    if n != 1: sys.exit(f'pattern found {n}x: {a[:60]!r}')
    src = src.replace(a, b)
open('nn_toy_train_lowP_eps.py', 'w').write(src)
print('ok')
