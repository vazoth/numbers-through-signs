"""Compute result tables from the supplied experimental records."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent
PRIMARY = sorted((("digit", "word"), ("word", "digit"), ("digit", "run"),
                  ("run", "digit"), ("word", "run"), ("run", "word")))
KEYS = ["model", "task", "template_id", "source_format", "target_format", "site", "window", "donor"]


def outcomes(frame):
    f = frame.copy()
    source_sign = np.where(f.source_expected_true, 1., -1.)
    target_sign = np.where(f.target_expected_true, 1., -1.)
    f["eligible"] = (f.source_raw_truth_margin * source_sign > 0) & (f.target_raw_truth_margin * target_sign > 0)
    if "patched_raw_truth_margin" in f:
        f["exact_noop_margin_delta"] = pd.to_numeric(f.get("exact_noop_margin_delta", np.nan))
        f["iia"] = np.where(f.eligible & f.donor.ne("exact-target"), f.patched_raw_truth_margin * source_sign > 0, np.nan)
        f["preserved"] = f.patched_raw_truth_margin * target_sign > 0
        f["gain"] = (f.patched_raw_truth_margin - f.target_raw_truth_margin) * source_sign
        # Exact-target controls use a same-batch reference rather than endpoint baselines.
        noop = f.donor.eq("exact-target")
        if noop.any():
            f.loc[noop, "gain"] = f.loc[noop, "exact_noop_margin_delta"] * source_sign[noop]
    return f


def site_selection(f):
    f = f.assign(outcome=f.patched_raw_truth_margin * np.where(f.source_expected_true, 1., -1.) > 0)
    denominator = f.source_source_margin - f.target_source_margin
    f["restoration"] = ((f.patched_source_margin - f.target_source_margin) / denominator).where(denominator.abs() > 1e-8)
    scores = f.groupby(["model", "site"], as_index=False).agg(
        trials=("outcome", "size"), iia=("outcome", "mean"), restoration=("restoration", "mean"))
    scores["selected"] = False
    for _, g in scores.groupby("model"):
        winner = g.sort_values(["iia", "restoration"], ascending=False, kind="stable").index[0]
        scores.loc[winner, "selected"] = True
    return scores


def admission(behavior, screening, criteria):
    f = behavior.assign(correct=behavior.raw_truth_margin * np.where(behavior.expected_true, 1., -1.) > 0)
    rows = []
    for (model, task, template), g in f.groupby(["model", "task", "template_id"], sort=False):
        accuracy = g.groupby(["format", "expected_true"]).correct.mean()
        ba = {fmt: float(accuracy.loc[fmt].mean()) for fmt in ("digit", "word", "run")}
        screen = screening[(screening.model == model) & (screening.template_id == template)]
        eligible = screen.groupby(["source_format", "target_format"]).eligible.sum()
        rows.append({"model": model, "task": task, "template": template,
                     "digit_ba": ba["digit"], "word_ba": ba["word"], "count_ba": ba["run"],
                     "minimum_eligible": int(eligible.min()),
                     "admitted": (min(ba["digit"], ba["word"]) >= criteria["symbol_balanced_accuracy"]
                                  and ba["run"] >= criteria["count_balanced_accuracy"]
                                  and accuracy.loc["run"].min() > criteria["each_count_class_exclusive"]
                                  and eligible.min() >= criteria["minimum_eligible"])})
    return pd.DataFrame(rows)


def readouts(f):
    rows = []
    for (model, sf, tf), g in f.groupby(["model", "source_format", "target_format"]):
        means = g.groupby(["held_out_value", "template_family"], as_index=False).prediction.mean()
        rows.append({"model": model, "source": sf, "target": tf,
                     "rho": float(spearmanr(means.held_out_value, means.prediction).statistic)})
    return pd.DataFrame(rows)


def interventions(f):
    return f.groupby(KEYS + ["control"], as_index=False).agg(
        requested=("eligible", "size"), eligible=("eligible", "sum"),
        eligibility=("eligible", "mean"), iia=("iia", "mean"),
        preservation=("preserved", "mean"), gain=("gain", "mean"),
        exact_noop_max_abs_delta=("exact_noop_margin_delta", lambda x: x.abs().max()))


def one_shot(frame, protocol):
    f = frame[frame.window.eq("1") & frame.donor.eq("source")]
    summary = interventions(f)
    cf = summary[summary.control.eq("counterfactual")].copy()
    identity = summary[summary.control.eq("same-number")]
    cf = cf.merge(identity[KEYS + ["preservation", "requested"]].rename(
        columns={"preservation": "identity", "requested": "identity_trials"}), on=KEYS, validate="one_to_one")
    cf["gain_lower"] = np.nan
    for idx, row in cf.iterrows():
        pair = (row.source_format, row.target_format)
        # Task offsets reserve five window groups for each principal direction.
        direction = PRIMARY.index(pair) if pair in PRIMARY else 6
        offset = protocol["bootstrap_task_offsets"][row.model][row.template_id]
        seed = protocol["bootstrap_seed"] + 1000 * offset + 85 * direction + 1
        g = f[f.control.eq("counterfactual") & f.model.eq(row.model) & f.template_id.eq(row.template_id)
              & f.source_format.eq(pair[0]) & f.target_format.eq(pair[1])]
        clusters = g.groupby("source_value", sort=True).gain.agg(["sum", "count"])
        draws = np.random.default_rng(seed).integers(0, len(clusters), size=(protocol["bootstrap_samples"], len(clusters)))
        means = clusters["sum"].to_numpy()[draws].sum(axis=1) / clusters["count"].to_numpy()[draws].sum(axis=1)
        cf.loc[idx, "gain_lower"] = np.quantile(means, .025)
    gate = protocol["direct_gate"]
    cf["meets_direct_criterion"] = ((cf.eligibility >= gate["eligibility"]) & (cf.identity >= gate["preservation"])
                             & (cf.iia >= gate["iia"]) & (cf.gain_lower > gate["gain_lower_exclusive"]))
    return cf


def analyze(root=ROOT):
    root = Path(root)
    protocol = json.loads((root / "data/protocol.json").read_text())
    data = {name: pd.read_parquet(root / f"data/{name}.parquet") for name in
            ("development", "behavior", "screening", "readouts", "interchange", "depth", "subspace")}
    admitted = admission(data["behavior"], outcomes(data["screening"]), protocol["admission"])
    causal = outcomes(data["interchange"])
    results = {"sites": site_selection(data["development"]), "admission": admitted,
               "readouts": readouts(data["readouts"]), "one_shot": one_shot(causal, protocol),
               "interchange": interventions(causal)}
    results.update({name: interventions(outcomes(data[name])) for name in ("depth", "subspace")})
    results["one_shot"] = results["one_shot"].merge(
        admitted[["model", "template", "admitted"]].rename(columns={"template": "template_id"}),
        on=["model", "template_id"], validate="many_to_one")
    return results


if __name__ == "__main__":
    output = ROOT / "results"
    output.mkdir(parents=True, exist_ok=True)
    for name, table in analyze().items():
        destination = output / f"{name}.csv"
        table.to_csv(destination, index=False)
        print(f"{destination}: {len(table)} rows")
