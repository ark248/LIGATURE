"""Replay the exploratory CD8a--epithelial case from frozen contact-count caches.

No fitting, threshold selection or clinical endpoint search is performed here.
The input caches come from eval_contact.py and the contact-trained direct model.
Only aggregate results are written. See --help for portable input paths.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


FEATURES = ("conditional", "abundance", "engaged_burden")
METHODS = ("measured", "stain", "stain_q", "ligature", "ligature_q", "direct")
# Marker order in the ORION 16-marker cache (eval_contact.py).
SOURCE, TARGET = 5, 14  # CD8a, Pan-CK


def interval(values):
    finite = np.asarray(values)[np.isfinite(values)]
    return np.quantile(finite, [.025, .975]).tolist() if len(finite) else None


def features(num, den, cells):
    """Pool region counts first; form each draw's ratio before averaging draws."""
    assert cells > 0
    contact = np.divide(num, den, out=np.zeros_like(num, dtype=float), where=den > 0)
    return [float(contact.mean()), float(np.asarray(den, float).mean() / cells),
            float(np.asarray(num, float).mean() / cells)]


def read_forecasts(generator_runs, reference_run, direct_dir, direct_tags):
    runs, meta = [], None
    for run in generator_runs:
        profile = {name: {} for name in METHODS[:-1]}
        metadata = {}
        for fold in range(5):
            with np.load(run / f"fold{fold}/contact_stats.npz") as z, np.load(
                    reference_run / f"fold{fold}/test_stats.npz") as n:
                assert np.array_equal(z["crop"], n["crop"])
                cases = z["case"]
                for method, prefix in zip(METHODS[:-1],
                        ("truth", "plugin", "plugin_q", "samples", "samples_q")):
                    num = z[prefix + "_num"][..., SOURCE, TARGET]
                    den = z[prefix + "_den"][..., SOURCE]
                    for case in sorted(set(cases)):
                        mask = cases == case
                        cells = int(n["ncell"][mask].sum())
                        pooled_num, pooled_den = num[mask].sum(0), den[mask].sum(0)
                        if method == "measured":
                            assert pooled_den >= 20, "Contact case requires 20 measured source cells"
                        assert np.all(pooled_num >= 0) and np.all(pooled_num <= pooled_den + 1e-3)
                        assert np.all(pooled_den <= cells + 1e-3)
                        profile[method][str(case)] = features(pooled_num, pooled_den, cells)
                        if str(case) in metadata:
                            assert metadata[str(case)] == (fold, cells), "Patient repeated across test folds"
                        metadata[str(case)] = (fold, cells)
                if meta is not None:
                    for case in set(cases):
                        for method in METHODS[:3]:
                            assert profile[method][str(case)] == runs[0][method][str(case)]
        if meta is None:
            meta = metadata
        assert meta == metadata
        runs.append(profile)
    cases = sorted(meta)
    predictions = {
        name: np.asarray([[run[name][c] for c in cases]
                          for run in (runs if name.startswith("ligature") else runs[:1])])
        for name in METHODS[:-1]}
    direct = []
    for tag in direct_tags:
        pred = {}
        for fold in range(5):
            with np.load(direct_dir / f"fold{fold}/direct_contact{tag}.npz") as z, np.load(
                    reference_run / f"fold{fold}/contact_stats.npz") as ref:
                assert np.array_equal(z["crop"], ref["crop"])
                for case in set(ref["case"]):
                    mask = ref["case"] == case
                    pred[str(case)] = features(z["num"][mask, SOURCE, TARGET].sum(),
                        np.maximum(z["den"][mask, SOURCE].sum(), 1e-6), meta[str(case)][1])
        assert set(pred) == set(cases)
        direct.append([pred[c] for c in cases])
    predictions["direct"] = np.asarray(direct)
    return cases, meta, predictions


def read_clinical(path):
    """Read final published ORION Supplementary Table 3, cohort 1 only."""
    ns = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as z:
        strings = ["".join(v.itertext()) for v in
                   ET.fromstring(z.read("xl/sharedStrings.xml")).findall("s:si", ns)]
        rows = []
        for row in ET.fromstring(z.read("xl/worksheets/sheet3.xml")).findall("s:sheetData/s:row", ns):
            cells = {}
            for cell in row.findall("s:c", ns):
                value = cell.find("s:v", ns)
                text = (value.text or "") if value is not None else ""
                if cell.get("t") == "s" and text:
                    text = strings[int(text)]
                cells[re.sub(r"\d", "", cell.get("r"))] = text
            rows.append(cells)
    header = rows[2]
    table = pd.DataFrame([{header.get(k, k): v for k, v in row.items()} for row in rows[3:]])
    table = table[(table.Cohort == "1") & table["Specimen ID"].str.fullmatch(r"C\d{2}")].copy()
    assert table["Specimen ID"].nunique() == len(table) == 40
    for key in ("PFSDays", "Recurrence", "PFSCensor", "IFM1"):
        table[key] = pd.to_numeric(table[key], errors="coerce")
    table = table.dropna(subset=["PFSDays", "Recurrence", "PFSCensor", "IFM1"])
    assert len(table) == 39
    assert (table.PFSDays > 0).all() and set(table.Recurrence) == {0., 1.}
    np.testing.assert_array_equal(table.Recurrence, 1 - table.PFSCensor)
    stages = []
    for text in table["Stage At Diagnosis"]:
        match = re.fullmatch(r"(IV|III|II|I)[ABC]?", text.replace(" ", ""))
        assert match, text
        stages.append({"I": 1, "II": 2, "III": 3, "IV": 4}[match[1]])
    table["stage"] = stages
    table.index = "CRC" + table["Specimen ID"].str.slice(1)
    return table.sort_index()


def concordance(risk, time, event, mask, weights):
    # Positive risk predicts an earlier event; equal survival times are excluded.
    comparable = (time[:, None] < time[None, :]) & event[:, None].astype(bool) & mask
    delta = risk[:, None] - risk[None, :]
    credit = ((delta > 1e-8) + .5 * (np.abs(delta) <= 1e-8)) * comparable
    den = np.einsum("bi,ij,bj->b", weights, comparable.astype(float), weights, optimize=True)
    num = np.einsum("bi,ij,bj->b", weights, credit, weights, optimize=True)
    return np.divide(num, den, out=np.full(len(den), np.nan), where=den > 0), int(comparable.sum())


def analyze(cases, meta, predictions, labels):
    result = dict(scope="Exploratory fixed-model case; no new fitting or endpoint search. "
                  "Patient bootstrap omits full training/development uncertainty. "
                  "Marker gates are proxies, not definitive cell-type or tumor labels.",
                  assay_patients=len(cases), generator_runs=len(predictions["ligature"]),
                  direct_runs=len(predictions["direct"]), radius_um=20, quantile=.85,
                  marker_pair=["CD8a", "Pan-CK"], assay=[], contact_comparisons=[])
    picks = np.random.default_rng(20261105).integers(0, len(cases), (2000, len(cases)))
    errors = {}
    for j, feature in enumerate(FEATURES):
        truth = predictions["measured"][0, :, j]
        base = np.mean((predictions["stain"][0, :, j] - truth) ** 2)
        for name, values in predictions.items():
            err = ((values[:, :, j] - truth) ** 2).mean(0)
            errors[feature, name] = err
            result["assay"].append(dict(feature=feature, method=name, runs=len(values),
                mse_relative_to_stain=float(err.mean() / base),
                spearman=float(np.mean([spearmanr(v[:, j], truth).statistic for v in values]))))
    for name in ("ligature", "ligature_q"):
        for baseline in ("stain_q", "direct"):
            a, b = errors["conditional", name], errors["conditional", baseline]
            result["contact_comparisons"].append(dict(method=name, baseline=baseline,
                mse_ratio=float(a.mean() / b.mean()),
                ci95=interval(a[picks].mean(1) / b[picks].mean(1))))
    if labels is None:
        return result
    clinical_cases = sorted(set(cases) & set(labels.index))
    labels = labels.loc[clinical_cases]
    ix = [cases.index(c) for c in clinical_cases]
    time, event = labels.PFSDays.to_numpy(), labels.Recurrence.to_numpy()
    fold = np.asarray([meta[c][0] for c in clinical_cases])
    stage = labels.stage.to_numpy()
    n = len(labels)
    rng = np.random.default_rng(2026110502)
    weights = np.zeros((2000, n))
    for f in sorted(set(fold)):
        ids = np.flatnonzero(fold == f)
        weights += np.eye(n)[rng.choice(ids, (2000, len(ids)), replace=True)].sum(1)
    masks = dict(pooled=np.ones((n, n), bool), within_fold=fold[:, None] == fold[None, :],
                 within_fold_and_stage=(fold[:, None] == fold[None, :]) & (stage[:, None] == stage[None, :]))
    result.update(clinical_patients=n, events=int(event.sum()), clinical=[], controls=[], clinical_comparisons=[])
    points, boots = {}, {}
    for j, feature in enumerate(FEATURES):
        for name, values in predictions.items():
            for mode, mask in masks.items():
                p, b = [], []
                for risk in -values[:, ix, j]:
                    v, pairs = concordance(risk, time, event, mask, np.ones((1, n)))
                    p.append(v[0]); b.append(concordance(risk, time, event, mask, weights)[0])
                point, boot = float(np.mean(p)), np.nanmean(b, axis=0)
                points[feature, name, mode], boots[feature, name, mode] = point, boot
                result["clinical"].append(dict(feature=feature, method=name, comparison=mode,
                    concordance=point, ci95=interval(boot), comparable_pairs=pairs))
    for name, risk in (("published_IFM1", -labels.IFM1.to_numpy()),
                       ("clinical_stage", stage.astype(float)), ("constant_risk", np.zeros(n))):
        for mode, mask in masks.items():
            p, pairs = concordance(risk, time, event, mask, np.ones((1, n)))
            b, _ = concordance(risk, time, event, mask, weights)
            result["controls"].append(dict(method=name, comparison=mode, concordance=float(p[0]),
                                          ci95=interval(b), comparable_pairs=pairs))
    for mode in masks:
        for feature in FEATURES:
            for base in ("stain_q", "direct"):
                a, b = (feature, "ligature_q", mode), (feature, base, mode)
                result["clinical_comparisons"].append(dict(feature=feature, comparison=mode,
                    contrast="ligature_q minus " + base, difference=points[a] - points[b],
                    ci95=interval(boots[a] - boots[b])))
        for name in METHODS:
            a, b = ("engaged_burden", name, mode), ("abundance", name, mode)
            result["clinical_comparisons"].append(dict(feature="burden_vs_abundance", comparison=mode,
                contrast=name + " burden minus abundance", difference=points[a] - points[b],
                ci95=interval(boots[a] - boots[b])))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generator-runs", nargs="+", type=Path, required=True)
    parser.add_argument("--reference-run", type=Path, help="Run containing test_stats.npz; default: first generator")
    parser.add_argument("--direct-dir", type=Path, required=True)
    parser.add_argument("--direct-tags", nargs="+", default=["_b0", "_b1", "_b2"])
    parser.add_argument("--clinical-table", type=Path, help="Original published Supplementary Tables 1--5 XLSX")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases, meta, predictions = read_forecasts(args.generator_runs,
        args.reference_run or args.generator_runs[0], args.direct_dir, args.direct_tags)
    labels = read_clinical(args.clinical_table) if args.clinical_table else None
    result = analyze(cases, meta, predictions, labels)
    if args.clinical_table:
        result["clinical_source_sha256"] = hashlib.sha256(args.clinical_table.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Wrote aggregate case results for {len(cases)} patients to {args.output}")


if __name__ == "__main__":
    main()
