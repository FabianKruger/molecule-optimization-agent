"""Table, significance tests and appendix figure for the medicinal chemistry tasks.

    pixi run -e baselines python scripts/stats/medchem_stats.py

Writes into data/results/stats/: table_medchem.tex, significance.csv and
ablation-significance.pdf/.png.

Outcomes are the mean best-so-far score over the 50 call budget, by the
trapezoidal rule as in the PMO benchmark. IC50 uses pIC50, with molecules
failing QED 0.6 or novelty set to the pIC50 floor of 0.

Two comparison families, each corrected within itself: every pair of the four
SEISMO information levels, and SEISMO against each alternative. They are kept
apart because a blocked ANOVA estimates one error term from every variant in
the model, and adding the zero-shot agent, which makes a single batched call,
doubles the error term the ablation is judged against.
"""

from __future__ import annotations

import csv
import itertools
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
from scipy.stats import alexandergovern, studentized_range, ttest_ind, ttest_rel
from scipy.stats import f as f_dist

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "data/results"
OUT = RESULTS / "stats"
BUDGET = 50

QED_THRESHOLD, NOVELTY_THRESHOLD = 0.6, 0.5
INVALID_IC50_NM, PIC50_FLOOR = 1e9, 0.0
QUERCETIN_THRESHOLD = 0.8

VARIANTS = {
    "seismo_full": "SEISMO -- full explanation",
    "seismo_partial": "SEISMO -- partial explanation",
    "seismo_none": "SEISMO -- no explanation",
    "seismo_nodesc": "SEISMO -- no description",
    "zeroshot_desc": "Zero-shot LLM (description)",
    "zeroshot_nodesc": "Zero-shot LLM (no description)",
    "reinvent": "REINVENT",
    "graph_ga": "Graph-GA",
    "gpbo": "GP BO",
}
FAMILIES = {
    "ablation": ["seismo_full", "seismo_partial", "seismo_none", "seismo_nodesc"],
    "alternatives": ["seismo_full", "zeroshot_desc", "reinvent", "graph_ga", "gpbo"],
}
CONTROL = "seismo_full"

_XAI = {"seismo_full": "full_xai", "seismo_partial": "partial_xai",
        "seismo_none": "no_xai", "seismo_nodesc": "no_description"}
_ZERO = {"zeroshot_desc": "task", "zeroshot_nodesc": "generic"}
_BASE = {"reinvent": "reinvent", "graph_ga": "graph_ga", "gpbo": "gpbo"}


# ---------------------------------------------------------------------------
#  Outcomes
# ---------------------------------------------------------------------------
def _auc(points: list[tuple[int, float]], start: float) -> float:
    """Mean best-so-far value over the budget, trapezoidal as in PMO."""
    best, previous, area = start, 0, 0.0
    for call, value in sorted(points):
        if call > BUDGET:
            break
        updated = max(best, value)
        area += (call - previous) * (updated + best) / 2
        best = updated
        previous = call
    return (area + best * (BUDGET - previous)) / BUDGET


def _files(directory: Path) -> list[Path]:
    return sorted(directory.rglob("*.json")) if directory.is_dir() else []


def _score_points(path: Path) -> list[tuple[int, float]]:
    return [(int(e["iteration"]), float(e["score"]))
            for e in json.loads(path.read_text())["trace"]]


def _pic50_points(path: Path) -> list[tuple[int, float]]:
    """pIC50 per call. Agent runs nest the constraint scores, baseline grid runs
    store them flat and are already in pIC50."""
    points = []
    for entry in json.loads(path.read_text())["trace"]:
        scores = entry.get("scores")
        if scores and "QED" in scores:
            ok = (float(scores["QED"]) >= QED_THRESHOLD
                  and float(scores.get("Novelty", 0.0)) >= NOVELTY_THRESHOLD)
            value = 9.0 - math.log10(float(entry["score"]) if ok else INVALID_IC50_NM)
        else:
            ok = (float(entry["qed"]) >= QED_THRESHOLD
                  and float(entry["novelty"]) >= NOVELTY_THRESHOLD)
            value = float(entry["pic50"]) if ok else PIC50_FLOOR
        points.append((int(entry["iteration"]), value))
    return points


def _reference(path: Path) -> int:
    match = re.search(r"(?:mol|target)(\d+)", path.name)
    if match:
        return int(match.group(1))
    config = json.loads(path.read_text()).get("config", {})
    if "target_idx" in config:
        return int(config["target_idx"])
    raise SystemExit(f"cannot read a reference index from {path}")


def _directory(variant: str, task: str) -> Path:
    if task == "sim_qed":
        if variant in _XAI:
            return RESULTS / "similarity_qed_pubchem" / f"claude-opus-4.5_{_XAI[variant]}"
        if variant in _ZERO:
            return RESULTS / "zero_shot/similarity_qed_pubchem" / _ZERO[variant]
        return RESULTS / "baseline_grid/sim_qed_pubchem" / _BASE[variant] / "default"
    if task == "ic50":
        if variant in _XAI:
            return RESULTS / "ic50mpro_qed" / f"claude-opus-4.5_{_XAI[variant]}"
        if variant in _ZERO:
            return RESULTS / "zero_shot/ic50mpro_qed_novel" / _ZERO[variant]
        return RESULTS / "baseline_grid/ic50" / _BASE[variant] / "default"
    if variant in _XAI:
        return RESULTS / "similarity_qed_quercetin" / f"claude-opus-4.5_{_XAI[variant]}_0.8"
    if variant in _ZERO:
        return RESULTS / "zero_shot/similarity_qed_quercetin" / _ZERO[variant]
    return RESULTS / "__no_baseline_quercetin_runs__"


def sim_qed(variants=None) -> list[tuple[str, int, float]]:
    return [(v, _reference(f), _auc(_score_points(f), 0.0))
            for v in (variants or VARIANTS) for f in _files(_directory(v, "sim_qed"))]


def ic50(variants=None) -> list[tuple[str, None, float]]:
    return [(v, None, _auc(_pic50_points(f), PIC50_FLOOR))
            for v in (variants or VARIANTS) for f in _files(_directory(v, "ic50"))]


def quercetin(variants=None) -> list[tuple[str, None, int | None]]:
    rows = []
    for variant in variants or VARIANTS:
        for path in _files(_directory(variant, "quercetin")):
            best, hit = 0.0, None
            for call, score in sorted(_score_points(path)):
                if call > BUDGET:
                    break
                best = max(best, score)
                if hit is None and best >= QUERCETIN_THRESHOLD:
                    hit = call
            rows.append((variant, None, hit))
    return rows


def grouped(rows) -> dict[str, list]:
    out = defaultdict(list)
    for variant, _, value in rows:
        out[variant].append(value)
    return out


# ---------------------------------------------------------------------------
#  Tests
# ---------------------------------------------------------------------------
def fmt(p: float) -> str:
    """studentized_range.sf stops resolving near 1e-13 and returns 0.0 beyond,
    which is the integration giving up rather than a true underflow."""
    return "<1e-13 (below the routine's resolution)" if p <= 0.0 else f"{p:.3g}"


def holm(pairs: list[tuple[str, float]]) -> dict[str, float]:
    ordered = sorted(pairs, key=lambda kv: kv[1])
    adjusted, running = {}, 0.0
    for i, (label, p) in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - i) * p))
        adjusted[label] = running
    return adjusted


def blocked_tukey(rows, task: str) -> list[dict]:
    """Randomised-block ANOVA on cell means, Tukey HSD for the pairs.

    Reference is a block. Replicate counts differ between variants, so the
    analysis is on the cell means, which is balanced, and variant is tested
    against the variant-by-reference interaction, not against replicate noise.
    """
    cell = defaultdict(list)
    for variant, block, value in rows:
        cell[(variant, block)].append(value)
    keys = [k for k in VARIANTS if any(v == k for v, _ in cell)]
    blocks = sorted({b for _, b in cell})
    if len(cell) != len(keys) * len(blocks):
        raise SystemExit(f"{task} is not a complete grid: {len(cell)} cells")

    mean = {k: statistics.mean(v) for k, v in cell.items()}
    grand = statistics.mean(mean.values())
    by = {k: statistics.mean([mean[(k, b)] for b in blocks]) for k in keys}
    ss_variant = len(blocks) * sum((m - grand) ** 2 for m in by.values())
    ss_block = len(keys) * sum(
        (statistics.mean([mean[(k, b)] for k in keys]) - grand) ** 2 for b in blocks)
    ss_total = sum((v - grand) ** 2 for v in mean.values())
    df = (len(keys) - 1) * (len(blocks) - 1)
    ms_error = (ss_total - ss_variant - ss_block) / df
    f_stat = (ss_variant / (len(keys) - 1)) / ms_error
    print(f"  {len(keys)} variants x {len(blocks)} references, "
          f"F({len(keys) - 1},{df}) = {f_stat:.1f}, "
          f"p = {fmt(float(f_dist.sf(f_stat, len(keys) - 1, df)))}")

    out = []
    for a, b in itertools.combinations(keys, 2):
        diff = by[a] - by[b]
        q = abs(diff) / (ms_error / len(blocks)) ** 0.5
        out.append({"task": task, "test": "blocked ANOVA, Tukey HSD",
                    "comparison": f"{a} vs {b}", "effect": round(diff, 4),
                    "statistic": round(q, 3),
                    "p_adjusted": float(studentized_range.sf(q, len(keys), df))})
    return out


def games_howell(groups: dict[str, list[float]], task: str) -> list[dict]:
    """Studentized range with per-pair Welch errors.

    Not Tukey, because the variant standard deviations differ by more than a
    factor of 10 and Tukey pools variance. Not a rank test either: at 5 against
    5 any exact rank test is floored at 2/C(10,5) = 0.0079 before correction,
    so every pair reports the same number whatever its size.
    """
    keys = [k for k in VARIANTS if k in groups]
    # Alexander-Govern weights by 1/se^2, so a variant with zero variance makes
    # it undefined. Dropped from the omnibus only, kept in every pair.
    usable = [k for k in keys if statistics.variance(groups[k]) > 0]
    result = alexandergovern(*[groups[k] for k in usable])
    dropped = sorted(set(keys) - set(usable))
    print(f"  Alexander-Govern A = {result.statistic:.1f}, "
          f"p = {fmt(float(result.pvalue))}"
          + (f"  (excludes {', '.join(dropped)}, zero variance)" if dropped else ""))

    out = []
    for a, b in itertools.combinations(keys, 2):
        xa, xb = groups[a], groups[b]
        va, vb = statistics.variance(xa), statistics.variance(xb)
        na, nb = len(xa), len(xb)
        se = ((va / na + vb / nb) / 2) ** 0.5
        diff = statistics.mean(xa) - statistics.mean(xb)
        if se == 0:
            q, p = float("inf"), (1.0 if diff == 0 else 0.0)
        else:
            q = abs(diff) / se
            df = (va / na + vb / nb) ** 2 / (
                (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
            p = float(studentized_range.sf(q, len(keys), df))
        out.append({"task": task, "test": "Alexander-Govern, Games-Howell",
                    "comparison": f"{a} vs {b}", "effect": round(diff, 3),
                    "statistic": round(q, 3) if q != float("inf") else "inf",
                    "p_adjusted": p})
    return out


def many_to_one(rows, task: str, paired: bool) -> list[dict]:
    """Each variant against the control, Holm corrected.

    Sim-QED is paired across the references. IC50 has independent runs and
    unequal variances, so Welch.
    """
    by = defaultdict(lambda: defaultdict(list))
    for variant, block, value in rows:
        by[variant][block].append(value)
    keys = [k for k in VARIANTS if k in by and k != CONTROL]

    raw, effect = [], {}
    for variant in keys:
        if paired:
            blocks = sorted(by[CONTROL])
            a = [statistics.mean(by[CONTROL][b]) for b in blocks]
            b_ = [statistics.mean(by[variant][b]) for b in blocks]
            result = ttest_rel(a, b_)
        else:
            a = [v for vals in by[CONTROL].values() for v in vals]
            b_ = [v for vals in by[variant].values() for v in vals]
            result = ttest_ind(a, b_, equal_var=False)
        comparison = f"{CONTROL} vs {variant}"
        raw.append((comparison, float(result.pvalue)))
        effect[comparison] = (statistics.mean(a) - statistics.mean(b_),
                              float(result.statistic))

    label = ("paired t across references, Holm" if paired else "Welch t, Holm")
    print(f"  {CONTROL} against {len(keys)} alternatives, {label}")
    out = []
    for comparison, p in sorted(holm(raw).items(), key=lambda kv: kv[1]):
        diff, statistic = effect[comparison]
        print(f"    {comparison:40s} diff {diff:+8.4f}  p {fmt(p)}")
        out.append({"task": task, "test": label, "comparison": comparison,
                    "effect": round(diff, 4), "statistic": round(statistic, 3),
                    "p_adjusted": p})
    return out


def run_tests() -> list[dict]:
    rows = []
    keys = FAMILIES["ablation"]
    print("\n--- ablation: every pair of the four information levels ---")
    print(" Sim-QED")
    rows += blocked_tukey(sim_qed(keys), "sim_qed:ablation")
    print(" pIC50-QED-Novel")
    rows += games_howell(grouped(ic50(keys)), "ic50:ablation")

    keys = FAMILIES["alternatives"]
    print("\n--- alternatives: SEISMO full against each alternative ---")
    print(" Sim-QED")
    rows += many_to_one(sim_qed(keys), "sim_qed:alternatives", paired=True)
    print(" pIC50-QED-Novel")
    rows += many_to_one(ic50(keys), "ic50:alternatives", paired=False)
    return rows


# ---------------------------------------------------------------------------
#  Table
# ---------------------------------------------------------------------------
PLACEHOLDER = "\\todoval{?}"


def _mean_sd(values, digits) -> tuple[str, float]:
    m = statistics.mean(values)
    return f"${m:.{digits}f} \\pm {statistics.stdev(values):.{digits}f}$", m


def _quercetin_cell(values) -> tuple[str, float]:
    reached = [v for v in values if v is not None]
    if not reached:
        return "$>50$", float("inf")
    if len(reached) < len(values):
        # A mean over only the runs that reached it would flatter them.
        return f"${len(reached)}/{len(values)}$ reached", float("inf")
    return _mean_sd([float(v) for v in reached], 1)


def write_table() -> None:
    ic, sim, que = grouped(ic50()), grouped(sim_qed()), grouped(quercetin())
    missing = [v for v in VARIANTS if not que.get(v)]
    if missing:
        print(f"no Quercetin runs for {', '.join(missing)}, "
              f"emitting {PLACEHOLDER}", file=sys.stderr)

    caption = (
        "  \\caption{Results of the medicinal-chemistry tasks, comparing four "
        "information levels of SEISMO with the LLM zero-shot capabilities with "
        "and without task description and three classical generative models as "
        "additional baselines. Mean and standard deviation over replicates.}"
    )
    lines = [
        "% Generated by scripts/stats/medchem_stats.py, do not edit by hand.",
        "\\begin{table*}[t]", caption, "  \\label{tab:medchem-main}", "  \\centering",
        "  \\resizebox{\\textwidth}{!}{%", "  \\begin{tabular}{lccc}", "    \\toprule",
        "    Model variant", "    & IC$_{50}$--QED--Novel",
        "    & Sim--QED, 20 molecules", "    & Sim--QED, Quercetin\\\\",
        "    & SARS-CoV-2 M$^\\mathrm{pro}$", "    & top-1 AUC $\\uparrow$",
        "    & oracle calls to target $\\downarrow$\\\\",
        "    & top-1 AUC of pIC$_{50}$ $\\uparrow$ & &\\\\", "    \\midrule",
    ]
    best = {"ic": max((statistics.mean(v), k) for k, v in ic.items() if v)[1],
            "sim": max((statistics.mean(v), k) for k, v in sim.items() if v)[1]}
    finite = [(_quercetin_cell(v)[1], k) for k, v in que.items() if v]
    best["que"] = min(finite)[1] if finite and min(finite)[0] != float("inf") else None

    for index, variant in enumerate(VARIANTS):
        if index in (4, 6):
            lines.append("    \\addlinespace[0.3em]")
        cells = []
        for key, source, digits in (("ic", ic, 2), ("sim", sim, 2), ("que", que, 1)):
            values = source.get(variant)
            if not values:
                cells.append(PLACEHOLDER)
                continue
            text = (_quercetin_cell(values)[0] if key == "que"
                    else _mean_sd(values, digits)[0])
            if variant == best[key]:
                text = text.replace("$", "$\\mathbf{", 1).rstrip("$") + "}$"
            cells.append(text)
        lines.append(f"    {VARIANTS[variant]:<32s} & " + " & ".join(cells) + "\\\\")

    lines += ["    \\bottomrule", "  \\end{tabular}}", "\\end{table*}", ""]
    (OUT / "table_medchem.tex").write_text("\n".join(lines))

    print(f"\n{'variant':34s} {'pIC50':>18s} {'Sim-QED':>18s} {'Quercetin':>18s}")
    for variant, label in VARIANTS.items():
        def show(source, digits, variant=variant):
            v = source.get(variant)
            if not v:
                return "--"
            text = (_quercetin_cell(v)[0] if source is que else _mean_sd(v, digits)[0])
            return text.replace("$", "")
        print(f"  {label:32s} {show(ic, 2):>18s} "
              f"{show(sim, 2):>18s} {show(que, 1):>18s}")


# ---------------------------------------------------------------------------
#  Figure
# ---------------------------------------------------------------------------
INK, INK_2, MUTED, AXIS, MARK = "#0b0b0b", "#52514e", "#898781", "#c3c2b7", "#52514e"
SHORT = {
    "seismo_full": "Full\nexplanation", "seismo_partial": "Partial\nexplanation",
    "seismo_none": "No\nexplanation", "seismo_nodesc": "No task\ndescription",
    "zeroshot_desc": "Zero-shot\nLLM", "reinvent": "REINVENT",
    "graph_ga": "Graph-GA", "gpbo": "GP BO",
}
# Smallest p reported as a number: below this it would rest on a distribution
# tail these sample sizes cannot verify.
P_FLOOR = 1e-4


def _plabel(p: float) -> str:
    if p < P_FLOOR:
        return "$p<10^{-4}$"
    if p < 0.001:
        base, exponent = f"{p:.0e}".split("e")
        return f"$p={base}{{\\times}}10^{{{int(exponent)}}}$"
    return f"$p={p:.3f}$" if p < 0.1 else f"$p={p:.2f}$"


def _panel(ax, values, pvalues, task, family, order, ylabel, tag, digits,
           bottom, top, ticks):
    height = top - bottom
    ax.set_ylim(bottom, top)
    ax.set_yticks(ticks)
    for i, variant in enumerate(order):
        vals = values[variant]
        reach = min(0.3, 0.03 * len(vals))
        xs = [i + (j - (len(vals) - 1) / 2) * (2 * reach / max(len(vals) - 1, 1))
              for j in range(len(vals))]
        ax.scatter(xs, vals, s=8, color=MARK, zorder=3, alpha=0.6,
                   edgecolors="white", linewidths=0.4)
        m = statistics.mean(vals)
        ax.plot([i - 0.3, i + 0.3], [m, m], color=INK, lw=1.5, zorder=4)
        ax.text(i, bottom + 0.015 * height, f"{m:.{digits}f}", ha="center",
                fontsize=6.2, color=INK_2)

    # Brackets, shortest spans lowest, greedily packed so none overlap.
    span = blended_transform_factory(ax.transData, ax.transAxes)
    pairs = ([(CONTROL, v) for v in order if v != CONTROL] if family == "alternatives"
             else [(a, b) for i, a in enumerate(order) for b in order[i + 1:]])
    pairs.sort(key=lambda ab: (order.index(ab[1]) - order.index(ab[0]),
                               order.index(ab[0])))
    level_right: list[float] = []
    for a, b in pairs:
        key = f"{task}:{family}|"
        p = pvalues.get(f"{key}{a} vs {b}", pvalues.get(f"{key}{b} vs {a}"))
        if p is None:
            continue
        xa, xb = order.index(a) + 0.08, order.index(b) - 0.08
        for level, right in enumerate(level_right):
            if xa > right + 0.14:
                level_right[level] = xb
                break
        else:
            level = len(level_right)
            level_right.append(xb)
        y = 0.62 + level * 0.088
        ax.plot([xa, xa, xb, xb], [y, y + 0.015, y + 0.015, y], lw=0.7, color=MUTED,
                transform=span, clip_on=False, solid_capstyle="butt")
        ax.text((xa + xb) / 2, y + 0.024, "n.s." if p > 0.05 else _plabel(p),
                ha="center", va="bottom", fontsize=6.0, transform=span,
                color=INK if p <= 0.05 else MUTED)

    ax.set_xticks(range(len(order)), [SHORT[v] for v in order],
                  fontsize=7.5 if len(order) <= 4 else 6.6)
    ax.set_ylabel(ylabel)
    ax.set_xlim(-0.6, len(order) - 0.4)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, axis="y", linestyle="-", linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    ax.text(-0.17, 1.02, tag, transform=ax.transAxes, fontsize=9.5,
            fontweight="bold", va="bottom", ha="left")


def write_figure(rows: list[dict]) -> None:
    pvalues = {f"{r['task']}|{r['comparison']}": r["p_adjusted"] for r in rows}

    cells = defaultdict(list)
    for variant, reference, value in sim_qed():
        cells[(variant, reference)].append(value)
    sim_values = defaultdict(list)
    for (variant, _), vals in cells.items():
        sim_values[variant].append(statistics.mean(vals))
    ic_values = grouped(ic50())

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7.5,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.labelcolor": INK_2,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
        "grid.color": AXIS, "grid.linewidth": 0.5, "text.color": INK,
        "pdf.fonttype": 42,
    })
    fig = plt.figure(figsize=(7.3, 6.4))
    gs = fig.add_gridspec(2, 2, hspace=0.62, wspace=0.34, left=0.105, right=0.985,
                          top=0.93, bottom=0.085)
    sim_label = "AUC (Sim–QED)"
    ic_label = "AUC (pIC$_{50}$–QED–Novel)"
    sim_axis = (0.40, 1.30, [0.5, 0.6, 0.7, 0.8, 0.9])
    ic_axis = (-0.6, 13.0, [0, 2, 4, 6, 8])

    for column, family in enumerate(FAMILIES):
        _panel(fig.add_subplot(gs[0, column]), sim_values, pvalues, "sim_qed",
               family, FAMILIES[family], sim_label, "ab"[column], 3, *sim_axis)
        _panel(fig.add_subplot(gs[1, column]), ic_values, pvalues, "ic50",
               family, FAMILIES[family], ic_label, "cd"[column], 2, *ic_axis)

    for extension in ("pdf", "png"):
        fig.savefig(OUT / f"ablation-significance.{extension}", dpi=300,
                    facecolor="white", bbox_inches="tight")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = run_tests()
    with open(OUT / "significance.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "p_adjusted": f"{row['p_adjusted']:.4g}"})
    write_table()
    write_figure(rows)
    for name in ("significance.csv", "table_medchem.tex",
                 "ablation-significance.pdf", "ablation-significance.png"):
        print(f"written to {OUT / name}")


if __name__ == "__main__":
    main()
