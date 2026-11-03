#!/usr/bin/env python
"""summarize_analyses.py STAGE MODE NCOND -- markdown summary of the newest analysis
folder of every variant x weighting for one stage (run_stage_optimizations.sh).
Reads analysis_<NCOND>cond_<variant>_<MODE>[_indep]_<stamp>/result.json."""
import sys, glob, json, os
import numpy as np

stage, mode, ncond = sys.argv[1], sys.argv[2], int(sys.argv[3])
VARIANTS = ["bothlim", "bothlim_r3", "eps", "eps_r3"]


def newest(variant, weighting):
    suf = "_indep" if weighting == "indep" else ""
    pat = f"analysis_{ncond}cond_{variant}_{mode}{suf}_2*"
    ds = [d for d in glob.glob(pat) if os.path.exists(os.path.join(d, "result.json"))]
    return max(ds, key=os.path.getmtime) if ds else None


rows, details = [], []
for w in ("dataset", "indep"):
    for v in VARIANTS:
        d = newest(v, w)
        if d is None:
            rows.append(f"| {v} | {w} | (missing) | | | | | | |")
            continue
        r = json.load(open(os.path.join(d, "result.json")))
        loo = r.get("leave_one_out", {})
        n_pass = sum(1 for h in loo.values() if h["max_abs_F_held"] < 1.0)
        sig = [n for n, a in r.get("ablation", {}).items() if a["significant"]]
        names, x, sd = r["param_names"], np.array(r["x_opt"]), np.array(r["sd"])
        at_bound = [n for n, xi in zip(names, x) if abs(xi) > 0.99]
        rows.append(f"| {v} | {w} | {r['objective']:.2f} | {r['sum_sq_data_residuals']:.1f} / "
                    f"{r['n_data_points']} | {r.get('max_abs_F', float('nan')):.2f} | "
                    f"{r.get('surrogate_shift_in_sd', float('nan')):.2f} | {n_pass}/{len(loo)} | "
                    f"{', '.join(sig) or '-'} | {', '.join(at_bound) or '-'} |")
        piv = {k[:-6]: p for k, p in r.get("rate_constants", {}).items() if k.endswith("_pivot")}
        details.append(f"**{v}, {w}** (`{d}`): " + "; ".join(
            f"{n} x{p['ratio']:.2f} ±{p['sd_lnk']:.2f} at {p['T']:.0f} K" for n, p in piv.items())
            + (f"; HO2 cross section x{np.exp(x[names.index('lnS_HO2_215')] * 0.7):.2f}"
               if "lnS_HO2_215" in names else ""))

print(f"# Stage {stage}: {mode} surrogates, {ncond} conditions\n")
print("| variant | weighting | objective | data chi2 / points | max abs F (Cantera) | "
      "surrogate shift [sd] | LOO pass | significant | at bound |")
print("|---|---|---|---|---|---|---|---|---|")
print("\n".join(rows))
print("\nRate constants at their pivot temperature (k/k_nom, 1-sd of ln k):\n")
print("\n\n".join(details))
