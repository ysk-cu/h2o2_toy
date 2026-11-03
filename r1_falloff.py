"""
r1_falloff.py -- perturbation of the falloff reaction R1, H2O2 (+M) <=> 2 OH (+M)

Shared by nn_toy_train_bothlimits.py, nn_toy_train_lowP_eps.py and optimize_final.py,
so the surrogate is trained and optimized against exactly the same perturbation.

Two knobs, following the treatment of pressure-dependent reactions in the
MUM-PCE papers:

  1. Rate knob (x_A, x_Ea): the SAME factor on A0 and A_inf, and the SAME shift
     on Ea0 and Ea_inf.  Then k0 and k_inf scale identically, P_r = k0[M]/k_inf
     and the Troe factor F(T, P_r) are unchanged, and
         k(T, P) -> k(T, P) * exp(x_A ln f - x_Ea sigma_E / RT)   at every P.

  2. Low-pressure knob (x_eps): a collision efficiency (here Ar, the bath gas of
     the Hong et al. experiments) is scaled as  eps = eps_0 * f_eps**x_eps.
     This moves [M]_eff, hence P_r, so it changes k only in the falloff / low-P
     regime and leaves the high-pressure limit untouched.

IMPORTANT (tested on Cantera 2.6.0, 3.0.1, 3.1.0 and 3.2.0): a change of
third-body efficiencies made through Solution.modify_reaction() is stored and
reads back correctly, but the kinetics keep using the ORIGINAL efficiencies,
so k is unchanged.  The efficiency knob is therefore applied by building a new
Solution from the edited reaction list (build_solution below).  The rate knob
works with modify_reaction() and does not need the rebuild.
"""

import numpy as np
import cantera as ct

CAL_PER_MOL = 4184.0                        # J/kmol per cal/mol
_HAS_THIRD_BODY = hasattr(ct, "ThirdBody")  # Cantera >= 3.0


def _arrhenius_class():
    """The Arrhenius class that TroeRate accepts: ArrheniusRate on Cantera >= 3.2,
    Arrhenius on 2.6 - 3.1 (where TroeRate rejects ArrheniusRate)."""
    for cls in (getattr(ct, "ArrheniusRate", None), getattr(ct, "Arrhenius", None)):
        if cls is None:
            continue
        try:
            ct.TroeRate(low=cls(1.0, 0.0, 0.0), high=cls(1.0, 0.0, 0.0),
                        falloff_coeffs=[0.5, 1e-30, 1e30])
            return cls
        except TypeError:
            continue
    raise RuntimeError("no Arrhenius class accepted by TroeRate in this Cantera version")


_ARR = _arrhenius_class()


def _get_eff(rxn):
    tb = getattr(rxn, "third_body", None)
    if tb is not None:
        return dict(tb.efficiencies), float(tb.default_efficiency)
    return dict(rxn.efficiencies), float(rxn.default_efficiency)


class R1Falloff:
    """Nominal parameters of one Troe falloff reaction and the two knobs."""

    def __init__(self, mech, idx, eps_species="AR"):
        g = ct.Solution(mech)
        r = g.reaction(idx)
        if not hasattr(r.rate, "low_rate"):
            raise TypeError(f"reaction {idx} ({r.equation}) is not a falloff reaction")
        self.mech, self.idx, self.equation = mech, idx, r.equation
        lo, hi = r.rate.low_rate, r.rate.high_rate
        # (A, b, Ea in J/kmol) -- Cantera SI units
        self.low  = (lo.pre_exponential_factor, lo.temperature_exponent, lo.activation_energy)
        self.high = (hi.pre_exponential_factor, hi.temperature_exponent, hi.activation_energy)
        self.falloff_coeffs = np.asarray(r.rate.falloff_coeffs, float)
        self.rate_cls = type(r.rate)                       # TroeRate in this mechanism
        self.eff0, self.default_eff = _get_eff(r)
        self.eps_species = eps_species
        if eps_species not in g.species_names:
            raise KeyError(f"{eps_species} is not a species of {mech}")
        # nominal efficiency of the knob species (falls back to the default, 1.0 here)
        self.eps0 = float(self.eff0.get(eps_species, self.default_eff))
        # pristine copies used by build_solution
        self._species = g.species()
        self._thermo_model, self._kinetics_model = g.thermo_model, g.kinetics_model

    # rate knob -------------------------------------------------------------
    def rate_object(self, fA=1.0, dEa_cal=0.0):
        """Troe rate with A0, A_inf multiplied by fA and Ea0, Ea_inf shifted by dEa_cal."""
        d = dEa_cal * CAL_PER_MOL
        low  = _ARR(self.low[0]  * fA, self.low[1],  self.low[2]  + d)
        high = _ARR(self.high[0] * fA, self.high[1], self.high[2] + d)
        return self.rate_cls(low=low, high=high, falloff_coeffs=self.falloff_coeffs)

    def apply_rate(self, gas, fA=1.0, dEa_cal=0.0):
        """In-place, both limits, same factor.  Efficiencies are NOT touched here."""
        r = gas.reaction(self.idx)
        r.rate = self.rate_object(fA, dEa_cal)
        gas.modify_reaction(self.idx, r)
        return gas

    # low-pressure knob --------------------------------------------------------
    def efficiencies(self, eps_factor=1.0):
        e = dict(self.eff0)
        e[self.eps_species] = self.eps0 * eps_factor
        return e

    def reaction_object(self, fA=1.0, dEa_cal=0.0, eps_factor=1.0):
        rate = self.rate_object(fA, dEa_cal)
        eff = self.efficiencies(eps_factor)
        if _HAS_THIRD_BODY:
            tb = ct.ThirdBody(efficiencies=eff, default_efficiency=self.default_eff)
            return ct.Reaction(equation=self.equation, rate=rate, third_body=tb)
        return ct.FalloffReaction(equation=self.equation, rate=rate,
                                  efficiencies=eff, kinetics=None)

    def build_solution(self, fA=1.0, dEa_cal=0.0, eps_factor=1.0):
        """A NEW Solution in which R1 carries both knobs.  Every other reaction is
        taken from a freshly loaded mechanism, so nothing carries over between calls;
        perturb the other reactions on the returned object with modify_reaction()."""
        rxns = ct.Solution(self.mech).reactions()
        rxns[self.idx] = self.reaction_object(fA, dEa_cal, eps_factor)
        return ct.Solution(thermo=self._thermo_model, kinetics=self._kinetics_model,
                           species=self._species, reactions=rxns)

    # diagnostics ----------------------------------------------------------------
    def check(self, T=1200.0, verbose=True):
        """Exact identities that must hold if the knobs are wired correctly.
        Returns the worst relative deviation."""
        worst = 0.0
        X = {self.eps_species: 1.0}

        def k_eff(g, P):
            g.TPX = T, P, X
            return g.forward_rate_constants[self.idx]

        # (a) rate knob: identical factor at every pressure
        fA, dEa = 1.7, -800.0
        want = fA * np.exp(-dEa / (ct.gas_constant / CAL_PER_MOL * T))
        g0, g1 = ct.Solution(self.mech), self.apply_rate(ct.Solution(self.mech), fA, dEa)
        for P in (0.01, 1.0, 100.0, 1e4):
            rel = abs(k_eff(g1, P * ct.one_atm) / k_eff(g0, P * ct.one_atm) / want - 1)
            worst = max(worst, rel)
            if verbose:
                print(f"      rate knob   P = {P:>8.2f} atm  k'/k = "
                      f"{k_eff(g1, P*ct.one_atm)/k_eff(g0, P*ct.one_atm):.8f}  "
                      f"expected {want:.8f}  dev {rel:.1e}")

        # (b) efficiency knob, pure bath gas: scaling eps by e is the same as scaling
        #     the pressure by e, so  k(T, P; e*eps) = k(T, e*P; eps)  exactly.
        e = 1.6
        ge = self.build_solution(eps_factor=e)
        gn = self.build_solution()
        for P in (0.1, 1.0, 10.0):
            a, b = k_eff(ge, P * ct.one_atm), k_eff(gn, e * P * ct.one_atm)
            rel = abs(a / b - 1); worst = max(worst, rel)
            if verbose:
                print(f"      eps knob    P = {P:>8.2f} atm  k(eps x{e}) / k(P x{e}) = "
                      f"{a/b:.8f}  expected 1  dev {rel:.1e}   "
                      f"(k ratio vs nominal {a/k_eff(gn, P*ct.one_atm):.4f})")

        # (c) the rebuild at nominal reproduces the original mechanism exactly
        g0.TPX = gn.TPX = T, ct.one_atm, "H2O2:0.002, H2O:0.001, O2:0.0005, AR:0.9965"
        rel = float(np.max(np.abs(gn.forward_rate_constants - g0.forward_rate_constants)
                           / np.maximum(np.abs(g0.forward_rate_constants), 1e-300)))
        worst = max(worst, rel)
        if verbose:
            print(f"      rebuild at nominal reproduces every k of the mechanism: dev {rel:.1e}")
        return worst
