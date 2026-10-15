"""
Turn a finished experiment's output folder into paper-ready tables, statistics and figures.

    python analyze.py --results "path/to/retrocycles_output"      # writes <results>/analysis/

Reads only the small files: final evaluations (metrics/final_*.csv), baselines,
per-iteration training logs (logs/*_train.csv), periodic evaluations
(metrics/*_eval.csv), heatmaps and PBT's generation log. The large per-round
training CSVs are not needed.

Statistics
- Win rates: Wilson 95% confidence intervals.
- Each condition vs main against the same bot: two-proportion test (chi-square, 2x2).
- Head-to-head vs main: exact binomial test against 50% (draws excluded).
- Coordination metrics vs main: Welch's t-test across rounds, Cohen's d effect size.
- All p-values in a family are Holm-corrected for multiple comparisons.
IMPORTANT: each condition was trained once (one seed). These tests measure how sure
we are about each trained team's behaviour across 1024 rounds; they do NOT measure
how much a different training seed would change the result. Say so in the paper.
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ---------------------------------------------------------------- conditions (fixed order)
CONDITIONS = [  # (run label, display name, what changed)
    ("main_s0", "Main", "full method"),
    ("no_credit_s0", "No credit after death", "a cycle's episode ends at its own death"),
    ("no_anneal_s0", "No shaping anneal", "conquest/kill shaping never fades"),
    ("partial_info_s0", "Partial info", "enemies beyond 15 cells (45 m) hidden"),
    ("no_pool_s0", "No opponent pool", "no frozen past snapshots"),
    ("no_humanlike_s0", "No reaction delay", "instant reactions"),
    ("no_agent_id_s0", "No slot ID", "cycles don't see their slot number"),
    ("pbt_s0", "PBT", "population-based training (best member)"),
]
NAME = {r: n for r, n, _ in CONDITIONS}
ORDER = [r for r, _, _ in CONDITIONS]
OPPONENTS = {"heuristic": "rush bot", "split": "split bot", "random": "random"}

COORD_METRICS = [  # column, label, "higher is more coordinated?" (None = no direction)
    ("multi_attack_rate", "Coordinated pushes\n(share of time 2+ teammates in enemy zone)", True),
    ("frac_attacking", "Time in enemy zone\n(share of living-cycle time)", None),
    ("frac_defending", "Time in own zone\n(share of living-cycle time)", None),
    ("role_specialization", "Role specialization\n(spread of individual roles)", True),
    ("undefended_rate", "Zone left undefended\nwhile under attack", False),
    ("friendly_fire_deaths", "Friendly-fire deaths\nper round", False),
]

# ---------------------------------------------------------------- style (validated reference palette)
SURFACE = "#fcfcfb"
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8984"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQ = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 9.5,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.family": "DejaVu Sans", "legend.frameon": False, "legend.fontsize": 9,
        "lines.linewidth": 2, "savefig.dpi": 200, "savefig.bbox": "tight",
    })


# ---------------------------------------------------------------- statistics helpers
def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan, np.nan)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, c - h, c + h


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    idx = np.argsort(p[ok])
    q = p[ok][idx]
    m = len(q)
    adj = np.maximum.accumulate([(m - i) * q[i] for i in range(m)])
    vals = np.empty(m)
    vals[idx] = np.minimum(adj, 1.0)
    out[ok] = vals
    return out


def cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    s = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / s if s > 0 else np.nan


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


# ---------------------------------------------------------------- loading
def load_final(res, run, opp):
    f = os.path.join(res, "metrics", f"final_{run}_vs_{opp}.csv")
    if not os.path.exists(f):
        return None
    d = pd.read_csv(f)
    return d[d.controller == run].copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="the experiment output folder (has metrics/, logs/ ...)")
    ap.add_argument("--out", default=None, help="default: <results>/analysis")
    args = ap.parse_args()
    res = args.results
    out = args.out or os.path.join(res, "analysis")
    os.makedirs(out, exist_ok=True)
    style()
    runs = [r for r in ORDER if os.path.exists(os.path.join(res, "metrics", f"final_{r}_vs_heuristic.csv"))]
    summary_lines = []

    # ============================================================ 1. final win rates
    rows = []
    finals = {}
    for r in runs:
        for opp in OPPONENTS:
            d = load_final(res, r, opp)
            if d is None:
                continue
            finals[(r, opp)] = d
            n = len(d)
            k = int((d.result == "win").sum())
            p, lo, hi = wilson(k, n)
            rows.append(dict(run=r, condition=NAME[r], opponent=opp, rounds=n, wins=k,
                             draws=int((d.result == "draw").sum()), win_rate=p, ci_low=lo, ci_high=hi))
    win = pd.DataFrame(rows)
    # vs main, same opponent
    pv = []
    for i, row in win.iterrows():
        if row.run == "main_s0":
            pv.append(np.nan)
            continue
        m = win[(win.run == "main_s0") & (win.opponent == row.opponent)].iloc[0]
        table = [[row.wins, row.rounds - row.wins], [m.wins, m.rounds - m.wins]]
        pv.append(stats.chi2_contingency(table)[1] if row.wins != m.wins else 1.0)
    win["p_vs_main"] = pv
    win["p_vs_main_holm"] = holm(win["p_vs_main"])
    win.to_csv(os.path.join(out, "table_win_rates.csv"), index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, opp, col in zip(axes, ["heuristic", "split"], SERIES[:2]):
        w = win[win.opponent == opp].set_index("run").reindex(runs)
        y = np.arange(len(runs))[::-1]
        ax.errorbar(w.win_rate, y, xerr=[w.win_rate - w.ci_low, w.ci_high - w.win_rate],
                    fmt="o", color=col, ecolor=col, elinewidth=1.5, capsize=0, markersize=7)
        for yi, (r, rr) in zip(y, w.iterrows()):
            ax.annotate(f"{rr.win_rate:.0%}", (rr.ci_high, yi), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=8.5, color=INK2)
        ax.axvline(0.5, color=INK3, lw=1, ls=(0, (4, 3)), zorder=0)
        ax.set_xlim(0, 1.12)
        ax.set_xticks([0, .25, .5, .75, 1])
        ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
        ax.set_title(f"vs {OPPONENTS[opp]}", loc="left")
        ax.set_xlabel("win rate (95% CI, 1024 rounds)")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(np.arange(len(runs))[::-1])
    axes[0].set_yticklabels([NAME[r] for r in runs])
    fig.suptitle("Final win rate of each trained team against the scripted bots", x=0.01, ha="left",
                 fontsize=12, fontweight="bold", color=INK)
    fig.savefig(os.path.join(out, "fig1_win_rates_vs_bots.png"))
    plt.close(fig)

    # ============================================================ 2. head-to-head vs main
    h2h = []
    for r in runs:
        if r == "main_s0":
            continue
        f = os.path.join(res, "metrics", f"final_main_s0_vs_{r}.csv")
        if not os.path.exists(f):
            continue
        d = pd.read_csv(f)
        a = d[d.controller == r]
        k, dr, n = int((a.result == "win").sum()), int((a.result == "draw").sum()), len(a)
        p, lo, hi = wilson(k, n)
        decisive = n - dr
        pb = stats.binomtest(k, decisive, 0.5).pvalue if decisive else np.nan
        h2h.append(dict(run=r, condition=NAME[r], rounds=n, wins_vs_main=k, draws=dr,
                        win_rate_vs_main=p, ci_low=lo, ci_high=hi, p_vs_50pct=pb))
    h2h = pd.DataFrame(h2h)
    if len(h2h):
        h2h["p_vs_50pct_holm"] = holm(h2h.p_vs_50pct)
        h2h.to_csv(os.path.join(out, "table_head_to_head_vs_main.csv"), index=False)
        hs = h2h.set_index("run").reindex([r for r in runs if r != "main_s0"])
        fig, ax = plt.subplots(figsize=(7.5, 3.8))
        y = np.arange(len(hs))[::-1]
        better = hs.win_rate_vs_main > 0.5
        for yi, (r, rr), b in zip(y, hs.iterrows(), better):
            c = SERIES[0] if b else INK3
            ax.plot([0.5, rr.win_rate_vs_main], [yi, yi], color=c, lw=2, solid_capstyle="round")
            ax.errorbar(rr.win_rate_vs_main, yi, xerr=[[rr.win_rate_vs_main - rr.ci_low], [rr.ci_high - rr.win_rate_vs_main]],
                        fmt="o", color=c, ecolor=c, elinewidth=1.2, markersize=7)
            ax.annotate(f"{rr.win_rate_vs_main:.0%} {stars(rr.p_vs_50pct_holm)}",
                        (max(rr.ci_high, 0.5), yi), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=8.5, color=INK2)
        ax.axvline(0.5, color=INK, lw=1)
        ax.set_yticks(y)
        ax.set_yticklabels([NAME[r] for r in hs.index])
        ax.set_xlim(0, 1.1)
        ax.set_xticks([0, .25, .5, .75, 1])
        ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
        ax.set_xlabel("win rate of the condition against Main (95% CI; Holm-corrected binomial test vs 50%)")
        ax.set_title("Head-to-head: each condition vs Main (right of the line = beats Main)", loc="left")
        ax.grid(axis="y", visible=False)
        fig.savefig(os.path.join(out, "fig2_head_to_head_vs_main.png"))
        plt.close(fig)

    # ============================================================ 3. learning curves (periodic eval)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    highlight = ["main_s0", "no_anneal_s0", "no_credit_s0", "pbt_s0"]
    curves = {}
    for r in runs:
        if r == "pbt_s0":
            gens = os.path.join(res, "logs", "pbt_s0_generations.jsonl")
            continue
        f = os.path.join(res, "metrics", f"{r}_eval.csv")
        if not os.path.exists(f):
            continue
        d = pd.read_csv(f)
        d = d[d.controller == "policy"]
        d["win"] = d.result == "win"
        curves[r] = d.groupby(["run_label", "samples"]).win.mean().reset_index()
    for ax, opp in zip(axes, ["heuristic", "split"]):
        for r, c in curves.items():
            cc = c[c.run_label.str.endswith(f"_vs_{opp}")]
            x = cc.samples / 1e9
            yv = cc.win.rolling(3, min_periods=1, center=True).mean()   # 3-point smoothing; 128 rounds per point
            if r in highlight:
                col = SERIES[highlight.index(r)]
                ax.plot(x, yv, color=col, lw=2, label=NAME[r], zorder=3)
                ax.annotate(NAME[r], (x.iloc[-1], yv.iloc[-1]), xytext=(4, 0), textcoords="offset points",
                            fontsize=8, color=INK2, va="center")
            else:
                ax.plot(x, yv, color=INK3, lw=1, alpha=0.7, label="other ablations" if r == "partial_info_s0" else None,
                        zorder=2)
        ax.set_title(f"vs {OPPONENTS[opp]}", loc="left")
        ax.set_xlabel("training samples (billions of agent-steps)")
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlim(0, 2.35)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    axes[0].set_ylabel("win rate (periodic eval, smoothed)")
    axes[0].legend(loc="upper left")
    fig.suptitle("Learning curves: win rate during training", x=0.01, ha="left", fontsize=12,
                 fontweight="bold", color=INK)
    fig.savefig(os.path.join(out, "fig3_learning_curves.png"))
    plt.close(fig)

    # ============================================================ 4. coordination metrics vs the rush bot
    base = {}
    for lab, ctrl in [("split_vs_heuristic", "split"), ("heuristic_mirror", "heuristic_A")]:
        f = os.path.join(res, "metrics", f"{lab}.csv")
        if os.path.exists(f):
            d = pd.read_csv(f)
            base[ctrl] = d[d.controller == ctrl]
    coord_rows = []
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2), sharey=True)
    for ax, (col, label, higher) in zip(axes.flat, COORD_METRICS):
        y = np.arange(len(runs))[::-1]
        main_vals = finals[("main_s0", "heuristic")][col]
        for yi, r in zip(y, runs):
            v = finals[(r, "heuristic")][col].dropna()
            m, se = v.mean(), v.std(ddof=1) / np.sqrt(len(v))
            p = d_eff = np.nan
            if r != "main_s0":
                p = stats.ttest_ind(v, main_vals.dropna(), equal_var=False).pvalue
                d_eff = cohens_d(v, main_vals)
            coord_rows.append(dict(metric=col, run=r, condition=NAME[r], mean=m, ci_low=m - 1.96 * se,
                                   ci_high=m + 1.96 * se, n=len(v), p_vs_main=p, cohens_d_vs_main=d_eff))
            c = SERIES[0] if r != "main_s0" else INK
            ax.errorbar(m, yi, xerr=1.96 * se, fmt="o", color=c, ecolor=c, elinewidth=1.3, markersize=6)
        for (bname, bd), ls in zip(base.items(), [(0, (4, 3)), (0, (1, 2))]):
            if col in bd:
                ax.axvline(bd[col].mean(), color=INK3, lw=1.2, ls=ls, zorder=0,
                           label="split bot (hand-coded roles)" if bname == "split" else "rush bot")
        ax.set_title(label, loc="left", fontsize=9.5)
        ax.grid(axis="y", visible=False)
    for ax in axes[:, 0]:
        ax.set_yticks(np.arange(len(runs))[::-1])
        ax.set_yticklabels([NAME[r] for r in runs])
    axes[0, 0].legend(loc="lower right", fontsize=8)
    fig.suptitle("Coordination behaviour in the final evaluation vs the rush bot (mean, 95% CI; 1024 rounds)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(out, "fig4_coordination_metrics.png"))
    plt.close(fig)
    coord = pd.DataFrame(coord_rows)
    coord["p_vs_main_holm"] = holm(coord.p_vs_main)
    coord.to_csv(os.path.join(out, "table_coordination_vs_rush.csv"), index=False)

    # coordination over training (per-iteration self-play averages)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, (col, lab) in zip(axes, [("frac_defending", "time in own zone"), ("role_specialization", "role specialization"),
                                     ("multi_attack_rate", "coordinated pushes")]):
        for r in runs:
            f = os.path.join(res, "logs", f"{r}_train.csv")
            if r == "pbt_s0" or not os.path.exists(f):
                continue
            t = pd.read_csv(f)
            if col not in t:
                continue
            yv = t[col].rolling(20, min_periods=1).mean()
            if r in highlight:
                ax.plot(t.samples / 1e9, yv, color=SERIES[highlight.index(r)], lw=2, label=NAME[r])
            else:
                ax.plot(t.samples / 1e9, yv, color=INK3, lw=1, alpha=0.6)
        ax.set_title(f"Self-play: {lab}", loc="left", fontsize=10)
        ax.set_xlabel("training samples (billions)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig5_coordination_during_training.png"))
    plt.close(fig)

    # ============================================================ 6. heatmaps (where cycles drive / die)
    hm_runs = [r for r in ["main_s0", "no_anneal_s0", "no_credit_s0"] if r in runs]
    if hm_runs:
        fig, axes = plt.subplots(2, len(hm_runs), figsize=(3.6 * len(hm_runs), 7.2))
        axes = np.atleast_2d(axes).reshape(2, len(hm_runs))
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq", SEQ)
        for j, r in enumerate(hm_runs):
            fs = sorted(glob.glob(os.path.join(res, "heatmaps", r, "*.npz")))
            if not fs:
                continue
            d = np.load(fs[-1])
            for i, key in enumerate(["visits", "deaths"]):
                a = d[key]
                ax = axes[i, j]
                ax.imshow(np.log1p(a / max(a.sum(), 1) * 1e5), cmap=cmap, interpolation="nearest")
                zc, zr = d["zone_center"], float(d["zone_radius"])
                for k, zcol in enumerate([SERIES[0], SERIES[1]]):
                    ax.add_patch(plt.Circle((zc[k, 1], zc[k, 0]), zr, fill=False, color=zcol, lw=1.5))
                ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
                if i == 0:
                    ax.set_title(NAME[r], fontsize=10)
                if j == 0:
                    ax.set_ylabel("where cycles drive" if i == 0 else "where cycles die", fontsize=9.5)
        fig.suptitle("Heatmaps from the last training evaluation vs the rush bot (128 rounds) (own zone = blue circle, bottom; enemy = orange, top)",
                     x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(os.path.join(out, "fig6_heatmaps.png"))
        plt.close(fig)

    # ============================================================ 7. PBT hyperparameters over generations
    gpath = os.path.join(res, "logs", "pbt_s0_generations.jsonl")
    if os.path.exists(gpath):
        gens = [json.loads(l) for l in open(gpath)]
        fig, axes = plt.subplots(1, 5, figsize=(16, 3.4))
        hps = ["lr", "ent_coef", "conquest", "kill"]
        members = sorted({m["id"] for g in gens for m in g["members"]})
        for k in members:
            col = SERIES[k % len(SERIES)]
            gx = [g["generation"] for g in gens]
            for ax, hp in zip(axes, hps):
                ax.plot(gx, [next(m for m in g["members"] if m["id"] == k)["hparams"][hp] for g in gens],
                        color=col, marker="o", markersize=4, lw=1.5, label=f"member {k}")
            axes[4].plot(gx, [next(m for m in g["members"] if m["id"] == k)["fitness"] for g in gens],
                         color=col, marker="o", markersize=4, lw=1.5)
        for ax, t in zip(axes, ["learning rate", "entropy bonus", "conquest shaping weight", "kill shaping weight",
                                "fitness (cross-play score)"]):
            ax.set_title(t, loc="left", fontsize=10)
            ax.set_xlabel("generation")
        axes[0].set_yscale("log")
        axes[0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "fig7_pbt_hyperparameters.png"))
        plt.close(fig)
        last = gens[-1]
        best = next(m for m in last["members"] if m["id"] == last["best"])
        summary_lines.append(f"PBT best member m{last['best']} final hyperparameters: " +
                             ", ".join(f"{k}={v:.3g}" for k, v in best["hparams"].items()))

    # ============================================================ summary
    with open(os.path.join(out, "summary.md"), "w", encoding="utf-8") as f:
        f.write("# Results summary\n\nEach condition trained once (one seed), 2e9 agent-steps; final "
                "evaluation 1024 rounds per matchup.\n\n## Win rate vs bots\n\n")
        f.write("| Condition | vs rush bot | vs split bot | vs random |\n|---|---|---|---|\n")
        for r in runs:
            cells = []
            for opp in ["heuristic", "split", "random"]:
                w = win[(win.run == r) & (win.opponent == opp)]
                if len(w):
                    w = w.iloc[0]
                    sig = "" if r == "main_s0" else f" {stars(w.p_vs_main_holm)}"
                    cells.append(f"{w.win_rate:.1%} [{w.ci_low:.1%}-{w.ci_high:.1%}]{sig}")
                else:
                    cells.append("-")
            f.write(f"| {NAME[r]} | " + " | ".join(cells) + " |\n")
        f.write("\nStars: difference from Main against the same bot, Holm-corrected "
                "(* p<.05, ** p<.01, *** p<.001).\n\n## Head-to-head vs Main\n\n")
        f.write("| Condition | its win rate vs Main | p (vs 50%, Holm) |\n|---|---|---|\n")
        for _, h in h2h.iterrows():
            f.write(f"| {h.condition} | {h.win_rate_vs_main:.1%} [{h.ci_low:.1%}-{h.ci_high:.1%}] | "
                    f"{h.p_vs_50pct_holm:.2g} {stars(h.p_vs_50pct_holm)} |\n")
        f.write("\n## Coordination metrics vs the rush bot (mean; Cohen's d vs Main)\n\n")
        piv = coord.pivot(index="run", columns="metric", values="mean").reindex(runs)
        dd = coord.pivot(index="run", columns="metric", values="cohens_d_vs_main").reindex(runs)
        cols = [c for c, _, _ in COORD_METRICS]
        f.write("| Condition | " + " | ".join(cols) + " |\n|" + "---|" * (len(cols) + 1) + "\n")
        for r in runs:
            cells = [f"{piv.loc[r, c]:.3f}" + ("" if r == "main_s0" else f" (d={dd.loc[r, c]:+.2f})") for c in cols]
            f.write(f"| {NAME[r]} | " + " | ".join(cells) + " |\n")
        for b, bd in base.items():
            f.write(f"| *{b} (scripted)* | " + " | ".join(f"{bd[c].mean():.3f}" for c in cols) + " |\n")
        f.write("\n" + "\n".join(summary_lines) + "\n")
    print(f"wrote tables, figures and summary.md to {out}")


if __name__ == "__main__":
    main()
