"""Data figures of the Forests manuscript (vector PDF, sized for the MDPI text width of about 12.5 cm).

    DOSSIER_SET=s1 .venv/Scripts/python paper/forests_eudr/figures/make_figures.py

Reads the sample design, the human reference labels, the GADM provinces and the scoring report of the current round
(runs/dossier/score_round_report.json); writes fig_study_area.pdf, fig_change_alerts.pdf, fig_forest2020.pdf and
fig_deforestation.pdf next to this script. Diagrams (pipeline, legal rule, evidence timeline) are TikZ files.
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("DOSSIER_SET", "s1")
ROOT = Path(__file__).resolve().parents[1]  # research_code root
sys.path.insert(0, str(ROOT))

import geopandas as gpd  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

OUT = Path(__file__).resolve().parent
W = 4.92  # inches = 12.5 cm
plt.rcParams.update({"font.size": 7.5, "font.family": "DejaVu Sans", "axes.linewidth": 0.6, "pdf.fonttype": 42,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "legend.frameon": False})
ITEMS = ROOT / "data" / "processed" / "dossier" / "s1" / "items_s1.csv"
TRUTH = ROOT / "data" / "interim" / "dossier" / "legal_vlm" / "human_truth.csv"
REPORT = ROOT / "runs" / "dossier" / "score_round_report.json"
GADM = ROOT / "data" / "raw" / "gazetteer" / "gadm41_VNM.gpkg"
PROV = ["Kon Tum", "Gia Lai", "Dak Lak", "Dak Nong", "Lam Dong"]
C_OURS, C_MAP, C_OTHER = "#1b5e20", "#c62828", "#757575"
NAMES = {"S1_ensemble": "VLM ensemble (S1)", "vlm_v2vi": "VLM, single prompt", "vlm_v2vi_shuf": "VLM, shuffled options",
         "E1_M1": "Time series + VLM (E1)", "gfcV4_hansen": "GFC2020 V4 + loss", "gfcV1_hansen": "GFC2020 V1 + loss",
         "national_loss": "National map + loss", "whisp_fao": "Whisp (FAO)", "whisp_like": "Map convergence",
         "rules_legal": "Legal rules on the card", "lgbm_dev": "LightGBM (DEV)", "tmf_loss": "JRC TMF", "dw_conversion": "Dynamic World",
         "ndvi_rule": "NDVI trajectory rule", "gfw_drivers": "GFC2020 V4 + drivers", "all_alerts": "Every alert"}
OURS = {"S1_ensemble", "vlm_v2vi", "vlm_v2vi_shuf", "E1_M1"}
MAPS = {"gfcV4_hansen", "gfcV1_hansen", "national_loss", "whisp_fao", "whisp_like", "tmf_loss", "dw_conversion", "gfw_drivers", "all_alerts"}


def colour(m):
    return C_OURS if m in OURS else (C_MAP if m in MAPS else C_OTHER)


def study_area():
    items = pd.read_csv(ITEMS)
    g = gpd.read_file(GADM, layer="ADM_ADM_1")
    vn = g.dissolve()
    ch = g[g.VARNAME_1.isin(PROV)]
    fig, (ax0, ax) = plt.subplots(1, 2, figsize=(W, 3.3), gridspec_kw={"width_ratios": [0.42, 1]})
    vn.plot(ax=ax0, color="#eeeeee", edgecolor="#9e9e9e", lw=0.3)
    ch.plot(ax=ax0, color="#a5d6a7", edgecolor="#1b5e20", lw=0.3)
    ax0.set_xlim(102, 110), ax0.set_ylim(8.3, 23.5), ax0.set_aspect("equal")
    ax0.set_xticks([]), ax0.set_yticks([]), ax0.set_xlabel(""), ax0.set_ylabel("")
    ax0.set_title("(a) Vietnam", fontsize=7.5, loc="left")
    # DEV blocks of the 0.2 degree grid
    for b in items[items.split == "dev"].block.unique():
        i, j = map(int, b.split("_"))
        ax.add_patch(Rectangle((i * 0.2, j * 0.2), 0.2, 0.2, facecolor="#fff3e0", edgecolor="#ffb74d", lw=0.3, zorder=0))
    ch.plot(ax=ax, facecolor="none", edgecolor="#424242", lw=0.5, zorder=2)
    for name, r in ch.set_index("VARNAME_1").iterrows():
        p = r.geometry.representative_point()
        ax.text(p.x, p.y, name, fontsize=6.5, ha="center", va="center", color="#212121", zorder=4,
                bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.7})
    a, p = items[items.kind == "alert"], items[items.kind == "plot"]
    ax.scatter(a.lon, a.lat, s=2.5, c="#c62828", lw=0, label=f"Alert patches (n = {len(a)})", zorder=3)
    ax.scatter(p.lon, p.lat, s=3, marker="s", c="#1565c0", lw=0, label=f"1-ha coffee plots (n = {len(p)})", zorder=3)
    x0, y0, x1, y1 = ch.total_bounds
    ax.set_xlim(x0 - 0.1, x1 + 0.1), ax.set_ylim(y0 - 0.1, y1 + 0.1), ax.set_aspect("equal")
    ax.tick_params(labelsize=6.5)
    ax.set_xlabel("Longitude (°E)", fontsize=7), ax.set_ylabel("Latitude (°N)", fontsize=7)
    h, l = ax.get_legend_handles_labels()
    h.append(Patch(facecolor="#fff3e0", edgecolor="#ffb74d", lw=0.5)), l.append("DEV blocks (0.2°); others TEST")
    ax.legend(h, l, loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=6.3, markerscale=2.5, handlelength=1.2)
    ax.set_title("(b) Central Highlands: sample and DEV/TEST blocks", fontsize=7.5, loc="left")
    ax0.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="#212121", lw=0.6))
    fig.tight_layout(w_pad=0.6)
    fig.savefig(OUT / "fig_study_area.pdf", bbox_inches="tight")
    plt.close(fig)


def truth():
    meta = pd.read_csv(ITEMS).set_index("item_id")
    t = pd.read_csv(TRUTH)
    t = t[t.item_id.isin(meta.index)].copy()
    t["year"] = meta.loc[t.item_id, "stratum"].str[:4].values
    return t


def change_alerts():
    """Human-labelled change after 2020 on alert patches that were forest in 2020, by year of first alert."""
    t = truth()
    a = t[(t.kind == "alert") & (t.y1 == 1) & t.cause.notna()]
    order = [("conversion_to_agriculture", "Converted to agriculture\n(deforestation)", "#c62828"),
             ("cleared_use_unseen", "Cleared, no use seen yet\n(undetermined)", "#f9a825"),
             ("plantation_rotation", "Plantation harvested\nand replanted", "#8d6e63"),
             ("other_non_agricultural", "Other change, no\nconversion", "#90a4ae"),
             ("none", "No change", "#c8e6c9")]
    tab = pd.crosstab(a.year, a.cause).reindex(columns=[k for k, _, _ in order], fill_value=0)
    tot = tab.sum(axis=1)
    sh = tab.div(tot, axis=0)
    fig, ax = plt.subplots(figsize=(W, 2.2))
    left = np.zeros(len(sh))
    ys = np.arange(len(sh))[::-1]
    for k, lab, c in order:
        ax.barh(ys, sh[k].values, left=left, color=c, edgecolor="white", lw=0.5, height=0.7, label=lab)
        for y, l0, v, n in zip(ys, left, sh[k].values, tab[k].values):
            if v >= 0.07:
                ax.text(l0 + v / 2, y, str(n), ha="center", va="center", fontsize=6.5)
        left += sh[k].values
    ax.set_yticks(ys, [f"{y} (n = {n})" for y, n in zip(sh.index, tot.values)])
    ax.set_xlim(0, 1), ax.set_xlabel("Share of labelled alert patches that were forest in 2020")
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylabel("Year of first alert")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=5, fontsize=6, handlelength=1, columnspacing=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.savefig(OUT / "fig_change_alerts.pdf", bbox_inches="tight")
    plt.close(fig)
    return tab


def predictions():
    from src.dossier.legal_vlm import baselines, vlm_predictions
    vp, _ = vlm_predictions("Qwen3.5-9B")
    return pd.concat([vp, baselines()], ignore_index=True)


def forest2020(rep):
    sets = [("all", "All labels"), ("test", "TEST"), ("blind", "Blind"), ("blind_test", "Blind TEST")]
    ms = ["S1_ensemble", "vlm_v2vi", "rules_legal", "whisp_fao", "gfcV4_hansen", "national_loss", "ndvi_rule", "dw_conversion",
          "gfcV1_hansen", "tmf_loss"]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(W, 2.7), gridspec_kw={"width_ratios": [1.45, 1]})
    mk = {"all": "o", "test": "s", "blind": "^", "blind_test": "D"}
    for k, m in enumerate(ms[::-1]):
        vals = [rep["sets"][s]["y1"].get(m, {}).get("bal_acc") for s, _ in sets]
        ax.hlines(k, min(v for v in vals if v), max(v for v in vals if v), color=colour(m), lw=1.2, alpha=0.5)
        for (s, lab), v in zip(sets, vals):
            if v is not None:
                ax.scatter(v, k, marker=mk[s], s=12, color=colour(m), zorder=3, label=lab if k == 0 else None)
    ax.set_yticks(range(len(ms)), [NAMES[m].replace(" + loss", "") for m in ms[::-1]])
    for tl, m in zip(ax.get_yticklabels(), ms[::-1]):
        tl.set_color(colour(m))
    ax.axvline(0.5, color="#bdbdbd", lw=0.6, ls=":")
    ax.set_xlim(0.4, 1.0), ax.set_xlabel("Balanced accuracy, forest on 31 Dec 2020")
    ax.legend(loc="lower right", fontsize=6, handletextpad=0.2, borderaxespad=0.1)
    ax.set_title("(a) Forest-2020 call against reference labels", fontsize=7.5, loc="left")
    # (b) share of labelled coffee plots called forest in 2020
    t = truth()
    p = t[(t.kind == "plot") & t.y1.notna()].set_index("item_id")
    P = predictions().pivot_table(index="item_id", columns="method", values="forest_2020", aggfunc="first")
    P = P.reindex(p.index)
    rows = [("Reference labels", p.y1.mean(), "#424242")]
    for m in ("gfcV1_hansen", "gfcV4_hansen", "national_loss", "vlm_v2vi"):
        v = P[m].map({"yes": 1, "no": 0}).dropna()
        rows.append((NAMES[m].replace(" + loss", ""), v.mean(), colour(m)))
    ys = np.arange(len(rows))[::-1]
    bx.barh(ys, [r[1] for r in rows], color=[r[2] for r in rows], height=0.6)
    for y, r in zip(ys, rows):
        bx.text(r[1] + 0.02, y, f"{r[1]:.0%}", va="center", fontsize=6.5)
    bx.set_yticks(ys, [r[0].replace("VLM, single prompt", "VLM") for r in rows])
    bx.set_xlim(0, 1.05), bx.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    bx.set_xlabel("Share called forest in 2020")
    bx.set_title(f"(b) Labelled coffee plots (n = {len(p)})", fontsize=7.5, loc="left")
    for a_ in (ax, bx):
        for s in ("top", "right"):
            a_.spines[s].set_visible(False)
    fig.tight_layout(w_pad=0.8)
    fig.savefig(OUT / "fig_forest2020.pdf", bbox_inches="tight")
    plt.close(fig)
    return rows


def deforestation(rep):
    """Balanced accuracy of the deforestation call on the four label sets (same layout as fig_forest2020 a)."""
    sets = [("all", "All labels"), ("test", "TEST"), ("blind", "Blind"), ("blind_test", "Blind TEST")]
    ms = ["E1_M1", "S1_ensemble", "vlm_v2vi_shuf", "vlm_v2vi", "ndvi_rule", "gfw_drivers", "gfcV1_hansen", "gfcV4_hansen",
          "all_alerts", "national_loss", "rules_legal", "tmf_loss", "whisp_fao", "lgbm_dev", "dw_conversion"]
    mk = {"all": "o", "test": "s", "blind": "^", "blind_test": "D"}
    fig, ax = plt.subplots(figsize=(W, 3.1))
    for k, m in enumerate(ms[::-1]):
        vals = [rep["sets"][s]["y2"].get(m, {}).get("bal_acc") if (m != "lgbm_dev" or s in ("test", "blind_test")) else None
                for s, _ in sets]
        ok = [v for v in vals if v is not None]
        ax.hlines(k, min(ok), max(ok), color=colour(m), lw=1.2, alpha=0.45)
        for (s, lab), v in zip(sets, vals):
            if v is not None:
                r = rep["sets"][s]["y2"]
                ax.scatter(v, k, marker=mk[s], s=13, color=colour(m), zorder=3,
                           label=f"{lab} (n = {r['n']}, deforested = {r['n_pos']})" if k == 0 else None)
    lab = {"E1_M1": "Time series + VLM (E1, exploratory)", "lgbm_dev": "LightGBM (DEV; TEST sets only)"}
    ax.set_yticks(range(len(ms)), [lab.get(m, NAMES[m]) for m in ms[::-1]])
    for tl, m in zip(ax.get_yticklabels(), ms[::-1]):
        tl.set_color(colour(m))
    ax.axvline(0.5, color="#bdbdbd", lw=0.6, ls=":")
    ax.text(0.505, len(ms) - 0.6, "no skill", fontsize=6, color="#9e9e9e")
    ax.set_xlim(0.4, 1.0), ax.set_xlabel("Balanced accuracy, deforestation after 31 Dec 2020")
    ax.legend(loc="upper center", bbox_to_anchor=(0.35, -0.13), ncol=2, fontsize=6, handletextpad=0.2, columnspacing=1.0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.savefig(OUT / "fig_deforestation.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    rep = json.loads(REPORT.read_text(encoding="utf-8"))
    study_area()
    tab = change_alerts()
    print(tab, tab.sum().sum())
    print(forest2020(rep))
    deforestation(rep)
    print("figures ->", OUT)


if __name__ == "__main__":
    main()
