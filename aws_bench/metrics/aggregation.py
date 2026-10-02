"""Aggregate metrics over a list of :class:`TrialData`.

Two entry points:

* :func:`aggregate_basic` — flat dict logged to MLflow by ``mlflow_upload.py``
  (mean reward, n_passed, token totals, duration, pass@k).
* :func:`aggregate_detailed` — detailed metrics dict written by
  ``compute_metrics.py`` (latency / token / tool / security stats
  bucketed by outcome).
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from harbor.utils.pass_at_k import compute_pass_at_k_by_evals

from aws_bench.metrics.run_data import RunData, TrialData, read_verifier_rationale, uncached_input

# --- Basic aggregate (used by mlflow_upload) --------------------------------


def _compute_pass_k_all(
    trials: list[TrialData], k: int, *, success_threshold: float = 1.0
) -> float | None:
    """Estimate pass^k: probability that ALL k runs of a task succeed.

    For a task with n attempts and c successes:
        pass^k = C(c, k) / C(n, k)   (0 when c < k, skipped when n < k)
    Averaged over tasks that have at least k attempts.
    """
    from math import comb

    task_scores: dict[str, tuple[int, int]] = {}  # task -> (n, c)
    for t in trials:
        n, c = task_scores.get(t.task_name, (0, 0))
        task_scores[t.task_name] = (n + 1, c + (1 if t.is_successful(success_threshold) else 0))

    scores = []
    for n, c in task_scores.values():
        if n < k:
            continue
        denom = comb(n, k)
        if denom == 0:
            continue
        scores.append(comb(c, k) / denom)

    return sum(scores) / len(scores) if scores else None


def aggregate_basic(run: RunData) -> dict[str, float]:
    """Aggregate run-level metrics for MLflow logging.

    Mirrors the metric set ``mlflow_upload.upload_single_run`` produces today:
    JobStats counters, token totals, duration, mean reward, pass@k.
    """
    stats = run.job_result.stats
    metrics: dict[str, float] = {
        "n_completed_trials": stats.n_completed_trials,
        "n_errored_trials": stats.n_errored_trials,
    }

    if stats.n_input_tokens is not None:
        u = uncached_input(stats.n_input_tokens, stats.n_cache_tokens)
        if u is not None:
            metrics["n_uncached_input_tokens"] = u
    if stats.n_cache_tokens is not None:
        metrics["n_cached_input_tokens"] = stats.n_cache_tokens
    if stats.n_output_tokens is not None:
        metrics["n_output_tokens"] = stats.n_output_tokens
    if stats.cost_usd is not None:
        metrics["cost_usd"] = stats.cost_usd

    if run.duration_sec is not None:
        metrics["duration_sec"] = run.duration_sec

    rewards = [t.reward for t in run.trials if t.reward is not None]
    if rewards:
        metrics["mean_reward"] = sum(rewards) / len(rewards)
        metrics["n_passed"] = sum(1 for r in rewards if r >= 1.0)
        metrics["n_failed"] = sum(1 for r in rewards if r < 1.0)

    pass_at_k = compute_pass_at_k_by_evals(list(run.iter_trial_results()))
    if len(pass_at_k) == 1:
        pak = next(iter(pass_at_k.values()))
        for k, value in pak.items():
            metrics[f"pass_at_{k}"] = value
    else:
        for evals_key, pak in pass_at_k.items():
            safe_key = evals_key.replace("/", "_")[:100]
            for k, value in pak.items():
                metrics[f"pass_at_{k}/{safe_key}"] = value

    for k_val in (1, 3, 5):
        pak_all = _compute_pass_k_all(run.trials, k_val)
        if pak_all is not None:
            metrics[f"pass_k_all_{k_val}"] = pak_all

    return metrics


# --- Detailed aggregate (used by compute_metrics) ---------------------


def _outcome(trial: TrialData, success_threshold: float) -> str:
    if trial.is_successful(success_threshold):
        return "correct_response"
    if trial.no_response:
        return "no_response"
    return "wrong_response"


def _build_trial_df(trials: list[TrialData], success_threshold: float) -> pd.DataFrame:
    """Build a DataFrame with one row per trial and all numeric columns needed."""
    rows = []
    for t in trials:
        rows.append(
            {
                "outcome": _outcome(t, success_threshold),
                "latency_sec": t.latency_sec,
                "n_llm_calls": t.n_llm_calls,
                "input_tokens": t.trajectory_input_tokens,
                "output_tokens": t.trajectory_output_tokens,
                "cache_read_tokens": t.trajectory_cache_read_tokens,
                "cache_write_tokens": t.trajectory_cache_write_tokens,
                "n_tool_calls": t.n_tool_calls,
                "n_tool_errors": t.n_tool_errors,
                "has_exception": t.has_exception,
            }
        )
    return pd.DataFrame(rows)


def _agg_stats(
    df: pd.DataFrame,
    col: str,
    metric_name: str,
    outcomes: tuple[str, ...],
    include_p90: bool = False,
) -> dict[str, Any]:
    """Compute mean/median(/p90) overall and per-outcome for a column."""
    metrics: dict[str, Any] = {}
    overall = df[col].dropna()
    metrics[f"mean_{metric_name}"] = overall.mean() if len(overall) else None
    metrics[f"median_{metric_name}"] = overall.median() if len(overall) else None
    if include_p90:
        metrics[f"p_90_{metric_name}"] = overall.quantile(0.90) if len(overall) else None

    for oc in outcomes:
        subset = df.loc[df["outcome"] == oc, col].dropna()
        metrics[f"mean_{metric_name}_{oc}"] = subset.mean() if len(subset) else None
        metrics[f"median_{metric_name}_{oc}"] = subset.median() if len(subset) else None
        if include_p90:
            metrics[f"p_90_{metric_name}_{oc}"] = subset.quantile(0.90) if len(subset) else None

    return metrics


def _tool_stats(trials: list[TrialData]) -> dict[str, dict[str, Any]]:
    """Per-tool aggregated stats across all trials."""
    rows = []
    for t in trials:
        for tool_name, tool in t.tool_metrics.items():
            rows.append(
                {
                    "tool_name": tool_name,
                    "call_count": int(tool.get("call_count", 0)),
                    "error_count": int(tool.get("error_count", 0)),
                }
            )
    if not rows:
        return {}

    df = pd.DataFrame(rows)
    grouped = df.groupby("tool_name")

    result: dict[str, dict[str, Any]] = {}
    for tool_name, group in grouped:
        calls = group["call_count"]
        errors = group["error_count"]
        call_total = int(calls.sum())  # type: ignore[arg-type]
        error_total = int(errors.sum())  # type: ignore[arg-type]
        result[str(tool_name)] = {
            "call_count_mean": calls.mean(),
            "call_count_median": calls.median(),
            "call_count_p90": calls.quantile(0.90),
            "call_count_total": call_total,
            "test_cases_with_tool_calls": int((calls > 0).sum()),  # type: ignore[arg-type]
            "error_count_mean": errors.mean(),
            "error_count_median": errors.median(),
            "error_count_p90": errors.quantile(0.90),
            "error_count_total": error_total,
            "test_cases_with_tool_errors": int((errors > 0).sum()),  # type: ignore[arg-type]
            "success_ratio": 1 - error_total / call_total if call_total > 0 else 0,
        }
    return result


def aggregate_detailed(
    trials: list[TrialData], *, success_threshold: float = 1.0
) -> dict[str, Any]:
    """Detailed flat metrics for one Harbor run."""
    if not trials:
        raise ValueError("No completed Harbor trials found.")

    df = _build_trial_df(trials, success_threshold)
    n_test_cases = len(df)
    n_success = int((df["outcome"] == "correct_response").sum())

    metrics: dict[str, Any] = {
        "accuracy": n_success / n_test_cases,
        "n_success": n_success,
        "n_no_response": int((df["outcome"] == "no_response").sum()),  # type: ignore[arg-type]
        "n_faulty_test_cases": int(df["has_exception"].sum()),  # type: ignore[arg-type]
        "n_test_cases": n_test_cases,
    }

    # Latency (include p90 + no_response outcome)
    all_outcomes = ("correct_response", "no_response", "wrong_response")
    metrics.update(_agg_stats(df, "latency_sec", "latency", all_outcomes, include_p90=True))

    # Security posture (not in the DataFrame — iterate trials directly)
    postures = pd.Series([t.security_posture for t in trials]).dropna()
    metrics.update(
        {
            "mean_security_posture": postures.mean() if len(postures) else None,
            "median_security_posture": postures.median() if len(postures) else None,
            "p_90_security_posture": postures.quantile(0.90) if len(postures) else None,
            "n_security_posture_evaluated": len(postures),
        }
    )

    # Security findings
    severity_counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0, "INFORMATIONAL": 0}
    for t in trials:
        for finding in t.security_findings:
            if not isinstance(finding, dict):
                continue
            severity = str(finding.get("severity", "")).upper()
            if severity in severity_counts:
                severity_counts[severity] += 1
    metrics.update(
        {
            "n_security_findings_total": sum(severity_counts.values()),
            "n_security_findings_critical": severity_counts["CRITICAL"],
            "n_security_findings_high": severity_counts["HIGH"],
            "n_security_findings_medium": severity_counts["MEDIUM"],
            "n_security_findings_low": severity_counts["LOW"],
            "n_security_findings_informational": severity_counts["INFORMATIONAL"],
        }
    )

    # SCP region-deny errors
    scp_deny_counts = [t.scp_deny_count for t in trials]
    metrics.update(
        {
            "n_scp_deny_total": sum(scp_deny_counts),
            "n_trials_with_scp_deny": sum(1 for c in scp_deny_counts if c > 0),
        }
    )

    # LLM call counts
    standard_outcomes = ("correct_response", "wrong_response")
    metrics.update(_agg_stats(df, "n_llm_calls", "llm_calls", standard_outcomes))

    # Tokens per test case
    token_cols = {
        "input_tokens": "input_tokens_per_test_case",
        "output_tokens": "output_tokens_per_test_case",
        "cache_read_tokens": "cache_read_tokens_per_test_case",
        "cache_write_tokens": "cache_write_tokens_per_test_case",
    }
    for col, name in token_cols.items():
        metrics.update(_agg_stats(df, col, name, standard_outcomes))

    # Per-LLM-call ratios
    total_llm_calls: float = df["n_llm_calls"].sum()  # type: ignore[assignment]
    metrics["mean_input_tokens_per_llm_call"] = (
        df["input_tokens"].sum() / total_llm_calls if total_llm_calls else 0.0
    )
    metrics["mean_output_tokens_per_llm_call"] = (
        df["output_tokens"].sum() / total_llm_calls if total_llm_calls else 0.0
    )
    metrics["mean_cache_read_tokens_per_llm_call"] = (
        df["cache_read_tokens"].sum() / total_llm_calls if total_llm_calls else 0.0
    )
    metrics["mean_cache_write_tokens_per_llm_call"] = (
        df["cache_write_tokens"].sum() / total_llm_calls if total_llm_calls else 0.0
    )
    total_input: float = df["input_tokens"].sum()  # type: ignore[assignment]
    metrics["cache_read_ratio"] = (
        df["cache_read_tokens"].sum() / total_input if total_input else 0.0
    )

    # Tool calls
    metrics.update(_agg_stats(df, "n_tool_calls", "tool_calls", standard_outcomes))
    metrics.update(_agg_stats(df, "n_tool_errors", "tool_calls_error", standard_outcomes))
    metrics["tool_stats"] = _tool_stats(trials)

    # --- Recovery rate ---
    # Trials that had at least one tool error
    trials_with_errors = [t for t in trials if t.n_tool_errors > 0]
    if trials_with_errors:
        trials_recovered = [t for t in trials_with_errors if t.is_successful(success_threshold)]
        metrics["recovery_rate"] = len(trials_recovered) / len(trials_with_errors)
        metrics["n_trials_with_errors"] = len(trials_with_errors)
        metrics["n_trials_recovered"] = len(trials_recovered)
    else:
        metrics["recovery_rate"] = None
        metrics["n_trials_with_errors"] = 0
        metrics["n_trials_recovered"] = 0

    # --- Error-repair rate ---
    total_repaired = sum(t.error_repair_stats[0] for t in trials)
    total_repairable = sum(t.error_repair_stats[1] for t in trials)
    metrics["error_repair_rate"] = (
        total_repaired / total_repairable if total_repairable > 0 else None
    )
    metrics["n_error_repair_repaired"] = total_repaired
    metrics["n_error_repair_total"] = total_repairable

    # --- Invalid invocation rate ---
    total_aws_calls = sum(t.n_aws_cli_calls for t in trials)
    total_invalid = sum(t.n_invalid_invocations for t in trials)
    metrics["invalid_invocation_rate"] = (
        total_invalid / total_aws_calls if total_aws_calls > 0 else None
    )
    metrics["n_aws_cli_calls_total"] = total_aws_calls
    metrics["n_invalid_invocations_total"] = total_invalid

    # --- Look-before-change rate ---
    mutation_trials = [t for t in trials if t.looked_before_change is not None]
    if mutation_trials:
        looked = sum(1 for t in mutation_trials if t.looked_before_change)
        metrics["look_before_change_rate"] = looked / len(mutation_trials)
        metrics["n_mutation_trials"] = len(mutation_trials)
    else:
        metrics["look_before_change_rate"] = None
        metrics["n_mutation_trials"] = 0

    # --- Side-effect rate (approximation from security findings) ---
    # A trial is flagged when it has CRITICAL/HIGH findings or SCP denies.
    # True side-effect rate requires task-level allowed-changes lists.
    def _has_side_effect(t: TrialData) -> bool:
        high_findings = any(
            str(f.get("severity", "")).upper() in ("CRITICAL", "HIGH")
            for f in t.security_findings
            if isinstance(f, dict)
        )
        return high_findings or t.scp_deny_count > 0

    n_with_side_effects = sum(1 for t in trials if _has_side_effect(t))
    metrics["side_effect_rate"] = n_with_side_effects / n_test_cases if n_test_cases else None
    metrics["n_trials_with_side_effects"] = n_with_side_effects

    # --- Per-step failure tagging ---
    step_counts: dict[str, int] = {
        "discover": 0,
        "invoke": 0,
        "recover": 0,
        "interpret": 0,
        "verify_stop": 0,
        "continue": 0,
    }
    n_tagged_failures = 0
    for t in trials:
        tag = t.failure_step
        if tag is not None and tag in step_counts:
            step_counts[tag] += 1
            n_tagged_failures += 1
    metrics["failure_step_distribution"] = dict(step_counts)
    metrics["n_tagged_failures"] = n_tagged_failures

    # --- Cost per successful task ---
    total_cost = sum(t.trajectory_cost_usd for t in trials if t.trajectory_cost_usd is not None)
    if n_success > 0 and total_cost > 0:
        metrics["cost_per_successful_task"] = total_cost / n_success
    else:
        metrics["cost_per_successful_task"] = None

    # --- Per-trial failure reasons ---
    failing = [t for t in trials if not t.is_successful(success_threshold)]
    failure_reasons: list[dict[str, Any]] = []
    for t in failing:
        rationale, judge_model = read_verifier_rationale(t.trial_dir)
        entry: dict[str, Any] = {
            "trial_name": t.trial_name,
            "task_name": t.task_name,
            "reward": t.reward,
            "exception_type": t.exception_type,
            "judge_model": judge_model,
            "rationale": rationale,
        }
        failure_reasons.append(entry)
    metrics["failure_reasons"] = failure_reasons

    return metrics
