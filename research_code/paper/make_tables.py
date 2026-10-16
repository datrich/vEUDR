"""Result tables of the Forests manuscript, written as LaTeX from the scoring report so that no number is typed by hand.

    .venv/Scripts/python paper/forests_eudr/figures/make_tables.py

Reads runs/dossier/score_round_report.json; writes paper/forests_eudr/tables/tab_forest2020.tex, tab_deforestation.tex
and tab_cause.tex.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # research_code root
REP = json.loads((ROOT / "runs" / "dossier" / "score_round_report.json").read_text(encoding="utf-8"))
OUT = ROOT / "paper" / "forests_eudr" / "tables"
SETS = [("all", "All"), ("test", "TEST"), ("blind", "Blind"), ("blind_test", "Blind TEST")]
GROUPS = [
    ("This study: facts read from imagery, law in code", [
        ("S1_ensemble", "VLM, five-prompt ensemble (S1)"), ("vlm_v2vi", "VLM, single prompt (primary)"),
        ("vlm_v2vi_shuf", "VLM, shuffled answer options"), ("E1_M1", "Time series + VLM answers (E1)$^{a}$")]),
    ("Map overlays and map products", [
        ("gfcV4_hansen", "GFC2020 V4 + Hansen loss"), ("gfcV1_hansen", "GFC2020 V1 + Hansen loss"),
        ("national_loss", "National 2020 map + loss"), ("whisp_fao", "Whisp (FAO), official"),
        ("gfw_drivers", "GFC2020 V4 + loss + drivers map"), ("tmf_loss", "JRC TMF"), ("dw_conversion", "Dynamic World"),
        ("all_alerts", "Every alert (plots: Hansen loss)")]),
    ("Rules and learning on the same evidence", [
        ("ndvi_rule", "NDVI trajectory rule"), ("rules_legal", "Legal rules on the evidence card"),
        ("lgbm_dev", "LightGBM on card features (DEV)")]),
]


def f(x, nd=2):
    return "--" if x is None else f"{x:.{nd}f}"


def best(target, s, keys):
    v = [REP["sets"][s][target].get(k, {}).get("bal_acc") for k in keys]
    return max(x for x in v if x is not None)


def table(target, cols, caption, label, note):
    keys = [k for _, g in GROUPS for k, _ in g]
    head = " & ".join(cols_head for _, cols_head, _ in cols)
    lines = [r"\begin{table}[H]", rf"\caption{{{caption}\label{{{label}}}}}", r"\begin{adjustwidth}{-\extralength}{0cm}",
             r"\newcolumntype{C}{>{\centering\arraybackslash}X}", rf"\begin{{tabularx}}{{\fulllength}}{{l{'C' * len(cols)}}}",
             r"\toprule", rf"\textbf{{Method}} & {head}\\"]
    sub = " & ".join(sub for _, _, sub in cols)
    lines += [rf" & {sub}\\", r"\midrule"]
    keys = [k for k in keys if not (target == "y1" and k == "gfw_drivers")]
    mx = {s: best(target, s, [k for k in keys if k != "lgbm_dev"]) for s, _, _ in cols if s in dict(SETS)}
    for gname, g in GROUPS:
        lines.append(rf"\multicolumn{{{len(cols) + 1}}}{{l}}{{\textit{{{gname}}}}}\\")
        for k, name in g:
            if target == "y1" and k == "gfw_drivers":
                continue  # identical to GFC2020 V4 for the forest-2020 call
            if target == "y1":
                name = name.replace(" + Hansen loss", "").replace(" + loss", "")
            row = [name]
            present = False
            for s, _, metric in cols:
                s0 = s.split(":")[0]
                r = REP["sets"][s0][target].get(k)
                if not isinstance(r, dict):
                    row.append("--")
                    continue
                present = True
                m = metric_of(s)
                v = r.get(m)
                cell = f(v)
                if k == "lgbm_dev" and s0 not in ("test", "blind_test"):  # trained on DEV: scored on TEST only
                    row.append("--")
                    continue
                if m == "bal_acc" and v is not None and s0 in ("all", "test") and abs(v - mx.get(s0, -1)) < 1e-9:
                    cell = rf"\textbf{{{cell}}}"
                row.append(cell)
            if present:
                lines.append(r"\quad " + " & ".join(row) + r"\\")
    n = " & ".join(rf"{REP['sets'][s.split(':')[0]][target]['n']} ({REP['sets'][s.split(':')[0]][target]['n_pos']})" for s, _, _ in cols)
    lines += [r"\midrule", rf"Items (positives) & {n}\\", r"\bottomrule", r"\end{tabularx}", r"\end{adjustwidth}",
              rf"\noindent{{\footnotesize{{{note}}}}}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def metric_of(s):
    return s.split(":")[1] if ":" in s else "bal_acc"


def cause():
    rows = [("Alert patches with a labelled change", "n"), ("Agricultural vs.\\ other change: items", "n_agri_vs_not"),
            ("\\quad VLM, single prompt", "vlm_v2vi_agri_vs_not"), ("\\quad VLM, shuffled options", "vlm_v2vi_shuf_agri_vs_not"),
            ("\\quad Drivers map (1 km)", "gfw_drivers_agri_vs_not"), ("Five change classes: VLM, single prompt", "vlm_v2vi"),
            ("Five change classes: VLM, shuffled options", "vlm_v2vi_shuf")]
    lines = [r"\begin{table}[H]",
             r"\caption{Cause of change on alert patches: share of patches where the method's change class agrees with the reference label. "
             r"The drivers map~\cite{sims2025drivers} gives one dominant driver per 1-km cell and has no ``no change'' or ``cleared, no use yet'' class, "
             r"so it is compared on agricultural versus other change only. The VLM has no ``cleared, no use seen yet'' option (Section~\ref{sec:legal}); its eight change options are folded into the form's classes (rotation; agriculture; other, including fire, logging, infrastructure and renovation), which caps agreement on five classes.\label{tab:cause}}",
             r"\newcolumntype{C}{>{\centering\arraybackslash}X}", r"\begin{tabularx}{\textwidth}{lCCCC}", r"\toprule",
             r"& \textbf{All} & \textbf{TEST} & \textbf{Blind} & \textbf{Blind TEST}\\", r"\midrule"]
    for name, k in rows:
        vals = [REP["sets"][s]["cause_alerts"].get(k) for s, _ in SETS]
        cells = [str(v) if k.startswith("n") else f(v) for v in vals]
        lines.append(name + " & " + " & ".join(cells) + r"\\")
    ex = json.loads((ROOT / "runs" / "dossier" / "extra_analyses.json").read_text(encoding="utf-8")).get("cause_baselines")
    if ex:
        lines.append(r"\midrule")
        lines.append(r"Majority class, agricultural vs.\ other & " + f"{ex['all']['majority2']:.2f} & -- & {ex['blind']['majority2']:.2f} & --" + r"\\")
        lines.append(r"Kappa, VLM shuffled / drivers map & " + f"{ex['all']['kappa2_vlm_v2vi_shuf']:.2f} / {ex['all']['kappa2_drivers']:.2f} & -- & "
                     f"{ex['blind']['kappa2_vlm_v2vi_shuf']:.2f} / {ex['blind']['kappa2_drivers']:.2f} & --" + r"\\")
        lines.append(r"Majority class, five classes & " + f"{ex['all']['majority5']:.2f} & -- & {ex['blind']['majority5']:.2f} & --" + r"\\")
    lines += [r"\bottomrule", r"\end{tabularx}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def main():
    OUT.mkdir(exist_ok=True)
    note_common = (r"Balanced accuracy is the mean of sensitivity and specificity; items a method leaves uncertain are excluded for that method "
                   r"(coverage $\geq$0.89 for every method; scores with uncertain counted as wrong are given in the text). Best value in the All "
                   r"and TEST columns in bold; the blind columns rest on few positive items and are not ranked. Person reference labels "
                   r"(the 1000-item reference with model labels: Supplementary Material S7); blind items carry no model suggestion.")
    t1 = table("y1", [("all", r"\textbf{All}", "BA"), ("test", r"\textbf{TEST}", "BA"), ("blind", r"\textbf{Blind}", "BA"),
                      ("blind_test", r"\textbf{Blind TEST}", "BA"), ("all:sensitivity", r"\textbf{All}", "Sens."),
                      ("all:specificity", r"\textbf{All}", "Spec.")],
               r"Forest on 31 December 2020: balanced accuracy (BA) against the reference labels on four label sets, and sensitivity and "
               r"specificity on all labels.", "tab:forest", note_common)
    t2 = table("y2", [("all", r"\textbf{All}", "BA"), ("test", r"\textbf{TEST}", "BA"), ("blind", r"\textbf{Blind}", "BA"),
                      ("blind_test", r"\textbf{Blind TEST}", "BA"), ("all:sensitivity", r"\textbf{All}", "Sens."),
                      ("all:false_green", r"\textbf{All}", "False green")],
               r"Deforestation after 31 December 2020: balanced accuracy (BA) on four label sets; sensitivity and false-green rate "
               r"(share of ``no deforestation'' calls that are wrong) on all labels.", "tab:defor",
               note_common + r" $^{a}$ Out-of-fold predictions of 5-fold spatial-block cross-validation on all labels, mean of five "
               r"repetitions; exploratory (Section~\ref{sec:compared}). E1 is not defined for the forest-2020 call.")
    (OUT / "tab_forest2020.tex").write_text(t1.replace(r"\quad Time series + VLM answers (E1)$^{a}$ & -- & -- & -- & -- & -- & --\\" + "\n", ""), encoding="utf-8")
    (OUT / "tab_deforestation.tex").write_text(t2, encoding="utf-8")
    (OUT / "tab_cause.tex").write_text(cause(), encoding="utf-8")
    print("tables ->", OUT)


if __name__ == "__main__":
    main()
