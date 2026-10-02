"""Shared helpers for post-processing Harbor run directories.

Provides two layers:

* Low-level loaders (``load_model``, ``load_json``, ``load_trajectory``,
  ``get_trial_dirs``, ``find_run_dirs``) and small utilities
  (``parse_datetime``, ``compute_duration_sec``, ``uncached_input``).
* :class:`TrialData` and :class:`RunData` — dataclasses that wrap one
  trial / one run and expose the primitives both ``mlflow_upload.py`` and
  ``compute_metrics.py`` extract today (rewards, tokens, tool
  metrics, security findings, latency, last agent message, etc.).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional, TypeVar

from harbor.models.job.config import JobConfig
from harbor.models.job.lock import JobLock
from harbor.models.job.result import JobResult
from harbor.models.trajectories import Step, Trajectory
from harbor.models.trial.result import TrialResult

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Surfaced as the rationale when verifier output exists but can't be parsed
# into a judge rationale (unknown/unsupported format, or a parse error).
UNSUPPORTED_RATIONALE = (
    "Rationale extraction is not supported for this verifier yet. Check verifier output."
)


# --- Low-level loaders -------------------------------------------------------


def load_model(model_cls: type[T], path: Path) -> T | None:
    """Load a Pydantic model from a JSON file, returning None on failure."""
    if not path.exists():
        return None
    try:
        return model_cls.model_validate_json(path.read_text())  # type: ignore[attr-defined]
    except Exception as exc:
        logger.warning("Failed to parse %s: %s", path, exc)
        return None


def load_json(path: Path) -> dict[str, Any] | None:
    """Load a JSON object from disk, returning None on failure."""
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to read %s: %s", path, exc)
        return None
    if not isinstance(data, dict):
        logger.warning("Expected JSON object in %s, got %s", path, type(data).__name__)
        return None
    return data


def load_trajectory(path: Path) -> Trajectory | None:
    """Load an ATIF trajectory using Harbor's native Pydantic model."""
    return load_model(Trajectory, path)


def read_verifier_rationale(trial_dir: Path) -> tuple[Optional[str], Optional[str]]:
    """Return the LLM judge rationale and model for a trial's verifier output.

    Prefers rewardkit's structured ``verifier/reward-details.json`` (the judge's
    per-criterion ``reasoning`` plus ``judge.model``). Falls back to the legacy
    ``verifier/test-stdout.txt`` format, where the rationale rode along on stdout
    behind an ``Agent output:`` marker.

    Returns ``(None, None)`` when the trial has no verifier output at all. When
    verifier output exists but can't be parsed (unsupported format or a parse
    error), returns :data:`UNSUPPORTED_RATIONALE` so the assessment points the
    operator at the raw verifier output.
    """
    verifier_dir = trial_dir / "verifier"
    reward_details_path = verifier_dir / "reward-details.json"
    stdout_path = verifier_dir / "test-stdout.txt"

    if not reward_details_path.exists() and not stdout_path.exists():
        return None, None

    try:
        details = load_json(reward_details_path)
        if details:
            reward = details.get("reward", {})
            parts = []
            for crit in reward.get("criteria") or []:
                reasoning = (crit.get("reasoning") or "").strip()
                if not reasoning:
                    continue
                name = crit.get("name")
                parts.append(f"{name}: {reasoning}" if name else reasoning)
            if parts:
                rationale = "\n\n".join(parts)[:10000]
                judge_model = (reward.get("judge") or {}).get("model")
                return rationale, judge_model

        if stdout_path.exists():
            content = stdout_path.read_text()
            marker = "Agent output:"
            idx = content.find(marker)
            rationale = (content[idx + len(marker) :] if idx != -1 else content).strip()[:10000]
            judge_model = None
            for line in content.splitlines():
                if line.startswith("Judge model:"):
                    judge_model = line.split(":", 1)[1].strip()
                    break
            if rationale:
                return rationale, judge_model
    except Exception as exc:
        logger.info("Failed to parse verifier rationale for %s: %s", trial_dir.name, exc)

    # Verifier output exists but yielded no usable rationale.
    return UNSUPPORTED_RATIONALE, None


def get_trial_dirs(run_dir: Path) -> list[Path]:
    """Find all trial subdirectories in a Harbor run directory."""
    return [
        child
        for child in sorted(run_dir.iterdir())
        if child.is_dir() and (child / "result.json").exists()
    ]


def find_run_dirs(logs_dir: Path) -> list[Path]:
    """Find all Harbor run directories directly under a logs directory."""
    return [
        child
        for child in sorted(logs_dir.iterdir())
        if child.is_dir() and (child / "result.json").exists()
    ]


def uncached_input(n_input: Optional[int], n_cache: Optional[int]) -> Optional[int]:
    """Derive uncached input token count (mirrors Harbor viewer logic)."""
    if n_input is None:
        return None
    if n_cache is None:
        return n_input
    return max(0, n_input - n_cache)


def parse_datetime(value: Any) -> datetime | None:
    """Parse a datetime or ISO-8601 string into a datetime object."""
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def parse_datetime_ns(value: Any) -> int | None:
    """Parse a datetime or ISO-8601 string into nanoseconds since epoch."""
    dt = parse_datetime(value)
    if dt is None:
        return None
    return int(dt.timestamp() * 1_000_000_000)


def compute_duration_sec(started_at: Any, finished_at: Any) -> float | None:
    """Compute duration in seconds between datetime objects or ISO strings."""
    start = parse_datetime(started_at)
    finish = parse_datetime(finished_at)
    if not start or not finish:
        return None
    return (finish - start).total_seconds()


# --- Helpers used by TrialData / RunData ------------------------------------


def _as_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        try:
            f = float(value)
        except (TypeError, ValueError):
            return None
        if f != f:  # NaN
            return None
        return f
    return None


def _as_int(value: Any) -> int:
    number = _as_number(value)
    return int(number) if number is not None else 0


def _get_nested(data: Any, *keys: str) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


# Primary: match the SCP ARN in enhanced error messages (2026+).
_SCP_DENY_ARN_RE = re.compile(
    r"explicit deny in a service control policy: "
    r"arn:aws:organizations::\d+:policy/o-[a-z0-9]+/service_control_policy/p-[a-z0-9]+"
)
# Fallback: older error format without the ARN suffix.
_SCP_DENY_PATTERN = "explicit deny in a service control policy"

# AWS CLI invalid invocation error patterns
_AWS_INVALID_PATTERNS = [
    re.compile(r"Invalid choice: '"),
    re.compile(r"Unknown options?: "),
    re.compile(r"No such (?:command|option): "),
    re.compile(r"Error: (?:No such|Unknown|Invalid) "),
    re.compile(r"Unknown parameter in input"),
    re.compile(r"unrecognized arguments"),
    re.compile(r"aws: error: argument "),
    re.compile(r"usage: aws.*\nerror:", re.DOTALL | re.IGNORECASE),
]

# AWS CLI verb classification
_AWS_READ_VERBS = frozenset(
    {
        "describe",
        "list",
        "get",
        "show",
        "search",
        "query",
        "check",
        "head",
        "test",
        "verify",
        "validate",
        "preview",
        "scan",
        "find",
        "lookup",
        "fetch",
        "retrieve",
        "view",
    }
)
_AWS_WRITE_VERBS = frozenset(
    {
        "create",
        "update",
        "delete",
        "put",
        "add",
        "attach",
        "detach",
        "remove",
        "terminate",
        "run",
        "start",
        "stop",
        "reboot",
        "modify",
        "associate",
        "disassociate",
        "enable",
        "disable",
        "set",
        "apply",
        "deploy",
        "register",
        "deregister",
        "allocate",
        "release",
        "import",
        "export",
        "restore",
        "copy",
        "move",
        "grant",
        "revoke",
        "publish",
        "subscribe",
        "unsubscribe",
        "reset",
        "rotate",
        "flush",
        "purge",
    }
)


def _contains_scp_deny(text: str) -> bool:
    """Return True if text contains an SCP deny error (ARN match preferred, string fallback)."""
    return _SCP_DENY_ARN_RE.search(text) is not None or _SCP_DENY_PATTERN in text


def _scp_deny_count_from_trajectory(trajectory: Trajectory | None) -> int:
    """Count observation results containing an SCP access-denied error."""
    if not trajectory:
        return 0
    count = 0
    for step in trajectory.steps:
        if step.observation is None:
            continue
        for result in step.observation.results:
            content = result.content
            if content is None:
                continue
            if isinstance(content, str):
                if _contains_scp_deny(content):
                    count += 1
            else:
                # list[ContentPart]
                for part in content:
                    if part.type == "text" and part.text and _contains_scp_deny(part.text):
                        count += 1
                        break
    return count


def _last_agent_message(trajectory: Trajectory | None) -> str:
    if not trajectory:
        return ""
    for step in reversed(trajectory.steps):
        if step.source != "agent":
            continue
        message = step.message
        if isinstance(message, str):
            return message
        parts = [part.text or "" for part in message if part.type == "text"]
        return "\n".join(parts)
    return ""


def _trajectory_latency(trajectory: Trajectory | None) -> float | None:
    if not trajectory:
        return None
    timestamps = [parse_datetime(step.timestamp) for step in trajectory.steps if step.timestamp]
    timestamps = [ts for ts in timestamps if ts is not None]
    if len(timestamps) < 2:
        return None
    return (timestamps[-1] - timestamps[0]).total_seconds()


def _is_tool_error(step: Step, tool_call_id: str) -> bool:
    extra = step.extra or {}
    if extra.get("tool_result_is_error") is True:
        return True
    metadata = extra.get("metadata")
    if isinstance(metadata, dict) and metadata.get("is_error") is True:
        return True
    result_metadata = extra.get("tool_result_metadata")
    if isinstance(result_metadata, dict) and result_metadata.get("is_error") is True:
        return True
    if step.observation is None:
        return False
    for result in step.observation.results:
        if result.source_call_id != tool_call_id:
            continue
        result_extra = result.extra or {}
        content = result.content
        return result_extra.get("is_error") is True or (
            isinstance(content, str) and "[error] tool reported failure" in content
        )
    return False


def _tool_metrics_from_trajectory(
    trajectory: Trajectory | None,
) -> dict[str, dict[str, float | int]]:
    stats: dict[str, dict[str, float | int]] = {}
    if not trajectory:
        return stats
    for step in trajectory.steps:
        for tool_call in step.tool_calls or []:
            tool_name = str(tool_call.function_name or "unknown_tool")
            tool_call_id = str(tool_call.tool_call_id or "")
            tool_stats = stats.setdefault(
                tool_name,
                {"call_count": 0, "success_count": 0, "error_count": 0, "total_time": 0.0},
            )
            is_error = _is_tool_error(step, tool_call_id)
            tool_stats["call_count"] = int(tool_stats["call_count"]) + 1
            if is_error:
                tool_stats["error_count"] = int(tool_stats["error_count"]) + 1
            else:
                tool_stats["success_count"] = int(tool_stats["success_count"]) + 1
    return stats


def _llm_usage_from_trajectory(trajectory: Trajectory | None) -> dict[str, int | float | None]:
    """Aggregate per-step LLM usage from the trajectory (token + cost totals).

    Falls back to ``trajectory.final_metrics`` when present.
    """
    total_input = 0
    total_output = 0
    cache_read = 0
    cache_write = 0
    cost = 0.0
    saw_cost = False
    n_llm_calls = 0

    if not trajectory:
        return {
            "n_llm_calls": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_cache_read_tokens": 0,
            "total_cache_write_tokens": 0,
            "cost_usd": None,
        }

    for step in trajectory.steps:
        if step.source != "agent":
            continue
        metrics = step.metrics
        llm_call_count = step.llm_call_count
        if isinstance(llm_call_count, int):
            n_llm_calls += llm_call_count
        elif metrics is not None:
            n_llm_calls += 1

        if metrics is None:
            continue
        total_input += metrics.prompt_tokens or 0
        total_output += metrics.completion_tokens or 0
        extra = metrics.extra or {}
        if metrics.cached_tokens is not None:
            cache_read += metrics.cached_tokens
        else:
            cache_read += _as_int(extra.get("cache_read_input_tokens"))
        cache_write += _as_int(extra.get("cache_creation_input_tokens"))
        step_cost = metrics.cost_usd
        if step_cost is not None:
            saw_cost = True
            cost += step_cost

    final_metrics = trajectory.final_metrics
    if final_metrics is not None:
        if final_metrics.total_prompt_tokens is not None:
            total_input = final_metrics.total_prompt_tokens
        if final_metrics.total_completion_tokens is not None:
            total_output = final_metrics.total_completion_tokens
        if final_metrics.total_cached_tokens is not None:
            cache_read = final_metrics.total_cached_tokens
        final_cost = final_metrics.total_cost_usd
        if final_cost is not None:
            saw_cost = True
            cost = final_cost
        extra = final_metrics.extra or {}
        if _as_number(extra.get("total_cache_read_input_tokens")) is not None:
            cache_read = _as_int(extra.get("total_cache_read_input_tokens"))
        if _as_number(extra.get("total_cache_creation_input_tokens")) is not None:
            cache_write = _as_int(extra.get("total_cache_creation_input_tokens"))

    return {
        "n_llm_calls": n_llm_calls,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_cache_read_tokens": cache_read,
        "total_cache_write_tokens": cache_write,
        "cost_usd": cost if saw_cost else None,
    }


def _security_metrics(
    trajectory: Trajectory | None, raw_result: dict[str, Any] | None
) -> tuple[float | None, list[dict[str, Any]]]:
    raw_result = raw_result or {}
    trajectory_extra = trajectory.extra if trajectory else None
    final_metrics_extra = (
        trajectory.final_metrics.extra
        if trajectory and trajectory.final_metrics and trajectory.final_metrics.extra
        else None
    )
    candidates = [
        raw_result.get("non_functional_validation_metrics"),
        _get_nested(trajectory_extra, "non_functional_validation_metrics"),
        _get_nested(final_metrics_extra, "non_functional_validation_metrics"),
        _get_nested(final_metrics_extra, "security"),
    ]
    for candidate in candidates:
        if isinstance(candidate, dict):
            posture = _as_number(candidate.get("security_posture"))
            findings = candidate.get("security_findings") or []
            return posture, findings if isinstance(findings, list) else []
    return None, []


def _aws_operation_key(command: str) -> str | None:
    """Extract 'service subcommand' key from an aws CLI command string."""
    m = re.search(r"aws\s+(?:--\S+\s+\S*\s*)*(\S+)(?:\s+([\w-]+))?", command)
    if not m:
        return None
    service = m.group(1)
    if service.startswith("-"):
        return None
    subcmd = m.group(2) or ""
    if subcmd.startswith("-"):
        subcmd = ""
    return f"{service} {subcmd}".strip() if subcmd else service


def _is_invalid_aws_invocation(text: str) -> bool:
    """Return True if text contains an AWS CLI invalid-invocation error."""
    return any(p.search(text) is not None for p in _AWS_INVALID_PATTERNS)


def _get_observation_text(step: Step) -> str:
    """Extract all text from a step's observation results as one string."""
    if step.observation is None:
        return ""
    parts: list[str] = []
    for result in step.observation.results:
        content = result.content
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for part in content:
                if part.type == "text" and part.text:
                    parts.append(part.text)
    return "\n".join(parts)


def _aws_cli_metrics_from_trajectory(
    trajectory: Trajectory | None,
) -> dict[str, Any]:
    """Compute AWS CLI-specific metrics from a trajectory.

    Returns a dict with:
      n_aws_cli_calls: total aws CLI invocations (bash calls containing 'aws ')
      n_invalid_invocations: calls that produced invalid-invocation errors
      error_repair_repaired: errors where agent retried same op successfully
      error_repair_total: total tool errors on aws CLI calls
      looked_before_change: True/False/None (None = no write calls observed)
    """
    if not trajectory:
        return {
            "n_aws_cli_calls": 0,
            "n_invalid_invocations": 0,
            "error_repair_repaired": 0,
            "error_repair_total": 0,
            "looked_before_change": None,
        }

    # Build sequence: (op_key | None, is_aws, is_write, is_read, is_tool_error, obs_text)
    records: list[dict[str, object]] = []
    for step in trajectory.steps:
        for tool_call in step.tool_calls or []:
            fn = str(tool_call.function_name or "")
            if fn not in ("bash", "execute_bash"):
                continue
            args = tool_call.arguments or {}
            command = str(args.get("command", args.get("cmd", "")))
            is_aws = bool(re.search(r"aws\s+", command))
            if not is_aws:
                continue
            op_key = _aws_operation_key(command)
            subcmd = (op_key.split(" ", 1)[1] if op_key and " " in op_key else "").lower()
            is_read = subcmd in _AWS_READ_VERBS
            is_write = subcmd in _AWS_WRITE_VERBS
            obs_text = _get_observation_text(step)
            is_error = _is_tool_error(step, str(tool_call.tool_call_id or ""))
            records.append(
                {
                    "op_key": op_key,
                    "is_read": is_read,
                    "is_write": is_write,
                    "is_error": is_error,
                    "obs_text": obs_text,
                }
            )

    n_aws = len(records)
    n_invalid = sum(1 for r in records if _is_invalid_aws_invocation(str(r["obs_text"])))

    # Error-repair: for each errored call, check if a later call to same op succeeded
    repaired = 0
    total_errors = sum(1 for r in records if r["is_error"])
    for i, r in enumerate(records):
        if not r["is_error"] or r["op_key"] is None:
            continue
        for j in range(i + 1, len(records)):
            if records[j]["op_key"] == r["op_key"] and not records[j]["is_error"]:
                repaired += 1
                break

    # Look-before-change
    first_write_idx = next((i for i, r in enumerate(records) if r["is_write"]), None)
    if first_write_idx is None:
        looked_before_change: bool | None = None
    else:
        looked_before_change = any(records[j]["is_read"] for j in range(first_write_idx))

    return {
        "n_aws_cli_calls": n_aws,
        "n_invalid_invocations": n_invalid,
        "error_repair_repaired": repaired,
        "error_repair_total": total_errors,
        "looked_before_change": looked_before_change,
    }


def _failure_step_tag(
    aws_metrics: dict[str, Any],
    n_tool_errors: int,
    n_llm_calls: int,
    reward: float | None,
) -> str | None:
    """Heuristic step-of-failure tag for a failed trial.

    Returns None for passing trials. For failing trials returns one of:
    discover | invoke | recover | interpret | verify_stop | continue
    """
    if reward is not None and reward >= 1.0:
        return None
    n_aws = int(aws_metrics.get("n_aws_cli_calls", 0) or 0)
    n_invalid = int(aws_metrics.get("n_invalid_invocations", 0) or 0)
    repaired = int(aws_metrics.get("error_repair_repaired", 0) or 0)
    total_errors = int(aws_metrics.get("error_repair_total", 0) or 0)

    if n_aws == 0:
        return "discover"
    if n_invalid > 0:
        return "invoke"
    if total_errors > 0 and repaired < total_errors:
        return "recover"
    if n_llm_calls > 15:
        return "verify_stop"
    if n_aws > 0 and n_tool_errors == 0:
        return "interpret"
    return "continue"


# --- TrialData / RunData -----------------------------------------------------


@dataclass
class TrialData:
    """All per-trial primitives needed by metrics uploaders."""

    trial_dir: Path
    result: TrialResult
    trajectory: Trajectory | None
    raw_result: dict[str, Any] | None

    # Cached derived values
    _usage: dict[str, int | float | None] = field(init=False, repr=False)
    _tool_metrics: dict[str, dict[str, float | int]] = field(init=False, repr=False)
    _security: tuple[float | None, list[dict[str, Any]]] = field(init=False, repr=False)
    _scp_deny_count: int = field(init=False, repr=False)
    _aws_cli_metrics: dict[str, Any] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Compute cached derived values from trajectory."""
        self._usage = _llm_usage_from_trajectory(self.trajectory)
        self._tool_metrics = _tool_metrics_from_trajectory(self.trajectory)
        self._security = _security_metrics(self.trajectory, self.raw_result)
        self._scp_deny_count = _scp_deny_count_from_trajectory(self.trajectory)
        self._aws_cli_metrics = _aws_cli_metrics_from_trajectory(self.trajectory)

    # --- Identity ---
    @property
    def task_name(self) -> str:
        """Return the task name from the trial result."""
        return self.result.task_name

    @property
    def trial_name(self) -> str:
        """Return the trial name from the trial result."""
        return self.result.trial_name

    # --- Reward / status ---
    @property
    def reward(self) -> float | None:
        """Return the verifier reward score, or None if unavailable."""
        if self.result.verifier_result and self.result.verifier_result.rewards:
            r = self.result.verifier_result.rewards.get("reward")
            if r is not None:
                return float(r)
            for value in self.result.verifier_result.rewards.values():
                n = _as_number(value)
                if n is not None:
                    return n
        return None

    def is_successful(self, threshold: float = 1.0) -> bool:
        """Return whether the trial reward meets the success threshold."""
        r = self.reward
        return r is not None and r >= threshold

    @property
    def exception_type(self) -> str | None:
        """Return the exception type if the trial raised one."""
        return self.result.exception_info.exception_type if self.result.exception_info else None

    @property
    def has_exception(self) -> bool:
        """Return whether the trial raised an exception."""
        return self.result.exception_info is not None

    @property
    def last_agent_message(self) -> str:
        """Return the last agent message from the trajectory."""
        return _last_agent_message(self.trajectory)

    @property
    def has_response(self) -> bool:
        """Return whether the agent produced a non-empty response."""
        return bool(self.last_agent_message.strip())

    @property
    def no_response(self) -> bool:
        """Return whether the agent produced no response."""
        return not self.has_response

    # --- Tokens / cost (from TrialResult Pydantic aggregation) ---
    @property
    def token_cost_totals(self) -> tuple[int | None, int | None, int | None, float | None]:
        """Return (input, cache, output, cost) token totals from result."""
        return self.result.compute_token_cost_totals()

    @property
    def n_input_tokens(self) -> int | None:
        """Return total input tokens."""
        return self.token_cost_totals[0]

    @property
    def n_cache_tokens(self) -> int | None:
        """Return total cache-read tokens."""
        return self.token_cost_totals[1]

    @property
    def n_output_tokens(self) -> int | None:
        """Return total output tokens."""
        return self.token_cost_totals[2]

    @property
    def cost_usd(self) -> float | None:
        """Return total cost in USD."""
        return self.token_cost_totals[3]

    @property
    def uncached_input_tokens(self) -> int | None:
        """Return input tokens minus cache-read tokens."""
        return uncached_input(self.n_input_tokens, self.n_cache_tokens)

    # --- Trajectory-derived usage ---
    @property
    def n_llm_calls(self) -> int:
        """Return number of LLM calls from trajectory."""
        return int(self._usage["n_llm_calls"] or 0)

    @property
    def trajectory_input_tokens(self) -> int:
        """Return total input tokens from trajectory steps."""
        return int(self._usage["total_input_tokens"] or 0)

    @property
    def trajectory_output_tokens(self) -> int:
        """Return total output tokens from trajectory steps."""
        return int(self._usage["total_output_tokens"] or 0)

    @property
    def trajectory_cache_read_tokens(self) -> int:
        """Return total cache-read tokens from trajectory steps."""
        return int(self._usage["total_cache_read_tokens"] or 0)

    @property
    def trajectory_cache_write_tokens(self) -> int:
        """Return total cache-write tokens from trajectory steps."""
        return int(self._usage["total_cache_write_tokens"] or 0)

    @property
    def trajectory_cost_usd(self) -> float | None:
        """Return total cost in USD from trajectory steps."""
        v = self._usage["cost_usd"]
        return float(v) if isinstance(v, (int, float)) else None

    # --- Tools ---
    @property
    def tool_metrics(self) -> dict[str, dict[str, float | int]]:
        """Return per-tool call/success/error counts."""
        return self._tool_metrics

    @property
    def n_tool_calls(self) -> int:
        """Return total number of tool calls across all tools."""
        return sum(int(t.get("call_count", 0)) for t in self._tool_metrics.values())

    @property
    def n_tool_errors(self) -> int:
        """Return total number of tool errors across all tools."""
        return sum(int(t.get("error_count", 0)) for t in self._tool_metrics.values())

    # --- Security (non-functional validation) ---
    @property
    def security_posture(self) -> float | None:
        """Return the security posture score, or None if unavailable."""
        return self._security[0]

    @property
    def security_findings(self) -> list[dict[str, Any]]:
        """Return the list of security findings."""
        return self._security[1]

    # --- SCP region-deny ---
    @property
    def scp_deny_count(self) -> int:
        """Return count of SCP access-denied errors in the trajectory."""
        return self._scp_deny_count

    # --- AWS CLI metrics ---
    @property
    def n_aws_cli_calls(self) -> int:
        """Return total number of AWS CLI invocations in the trajectory."""
        return int(self._aws_cli_metrics.get("n_aws_cli_calls", 0) or 0)

    @property
    def n_invalid_invocations(self) -> int:
        """Return number of AWS CLI calls that produced invalid-invocation errors."""
        return int(self._aws_cli_metrics.get("n_invalid_invocations", 0) or 0)

    @property
    def error_repair_stats(self) -> tuple[int, int]:
        """Return (repaired, total_errors) for AWS CLI error-repair tracking."""
        return (
            int(self._aws_cli_metrics.get("error_repair_repaired", 0) or 0),
            int(self._aws_cli_metrics.get("error_repair_total", 0) or 0),
        )

    @property
    def looked_before_change(self) -> bool | None:
        """Return whether agent read before first write call; None if no writes."""
        v = self._aws_cli_metrics.get("looked_before_change")
        return bool(v) if v is not None else None

    @property
    def failure_step(self) -> str | None:
        """Return heuristic step-of-failure tag, or None for passing trials."""
        return _failure_step_tag(
            self._aws_cli_metrics,
            self.n_tool_errors,
            self.n_llm_calls,
            self.reward,
        )

    # --- Latency ---
    @property
    def agent_execution_sec(self) -> float | None:
        """Return agent execution duration in seconds."""
        if not self.result.agent_execution:
            return None
        return compute_duration_sec(
            self.result.agent_execution.started_at,
            self.result.agent_execution.finished_at,
        )

    @property
    def trial_duration_sec(self) -> float | None:
        """Return total trial duration in seconds."""
        return compute_duration_sec(self.result.started_at, self.result.finished_at)

    @property
    def latency_sec(self) -> float | None:
        """Return best-available latency (agent exec > trial > trajectory)."""
        for value in (
            self.agent_execution_sec,
            self.trial_duration_sec,
            _trajectory_latency(self.trajectory),
        ):
            if value is not None:
                return value
        return None

    # --- Convenience ---
    @property
    def trajectory_path(self) -> Path:
        """Return the expected path to the trajectory file."""
        return self.trial_dir / "agent" / "trajectory.json"

    @property
    def has_trajectory(self) -> bool:
        """Return whether the trajectory file exists on disk."""
        return self.trajectory_path.exists()


@dataclass
class RunData:
    """All metadata + trials for a single Harbor run directory."""

    run_dir: Path
    job_result: JobResult
    config: JobConfig | None
    lock: JobLock | None
    trials: list[TrialData]

    @classmethod
    def load(cls, run_dir: Path) -> "RunData | None":
        """Load a run directory. Returns None if result.json is missing/invalid."""
        run_dir = run_dir.expanduser().resolve()
        job_result = load_model(JobResult, run_dir / "result.json")
        if not job_result:
            return None
        config = load_model(JobConfig, run_dir / "config.json")
        lock = load_model(JobLock, run_dir / "lock.json")

        trials: list[TrialData] = []
        for trial_dir in get_trial_dirs(run_dir):
            tr = load_model(TrialResult, trial_dir / "result.json")
            if not tr:
                continue
            traj = load_trajectory(trial_dir / "agent" / "trajectory.json")
            raw = load_json(trial_dir / "result.json")
            trials.append(
                TrialData(
                    trial_dir=trial_dir,
                    result=tr,
                    trajectory=traj,
                    raw_result=raw,
                )
            )
        return cls(run_dir=run_dir, job_result=job_result, config=config, lock=lock, trials=trials)

    # --- Identity ---
    @property
    def job_id(self) -> str:
        """Return the job ID."""
        return str(self.job_result.id)

    # --- Config-derived metadata ---
    @property
    def agent_name(self) -> str | None:
        """Return the first agent's name from config."""
        if self.config and self.config.agents:
            return self.config.agents[0].name
        return None

    @property
    def model_name(self) -> str | None:
        """Return the model name without provider prefix."""
        if self.config and self.config.agents and self.config.agents[0].model_name:
            parts = self.config.agents[0].model_name.split("/", 1)
            return parts[1] if len(parts) == 2 else parts[0]
        return None

    @property
    def model_provider(self) -> str | None:
        """Return the model provider prefix."""
        if self.config and self.config.agents and self.config.agents[0].model_name:
            parts = self.config.agents[0].model_name.split("/", 1)
            return parts[0] if len(parts) == 2 else None
        return None

    @property
    def dataset_source(self) -> str | None:
        """Return the dataset source name or path stem."""
        if not (self.config and self.config.datasets):
            return None
        ds = self.config.datasets[0]
        if ds.path:
            return ds.path.name
        return ds.name

    @property
    def environment_type(self) -> str | None:
        """Return the environment type value."""
        if self.config and self.config.environment.type:
            return self.config.environment.type.value
        return None

    @property
    def harbor_version(self) -> str | None:
        """Return the Harbor version from the lock file."""
        return self.lock.harbor.version if self.lock else None

    @property
    def n_attempts(self) -> int | None:
        """Return the configured number of attempts per task."""
        return self.config.n_attempts if self.config else None

    @property
    def n_concurrent_trials(self) -> int | None:
        """Return the configured concurrency limit."""
        return self.config.n_concurrent_trials if self.config else None

    # --- Run-level timing ---
    @property
    def duration_sec(self) -> float | None:
        """Return the total run duration in seconds."""
        return compute_duration_sec(self.job_result.started_at, self.job_result.finished_at)

    # --- Convenience iterators ---
    def iter_trial_results(self) -> Iterable[TrialResult]:
        """Iterate over raw TrialResult objects for all trials."""
        return (t.result for t in self.trials)

    def task_attempt_numbers(self) -> list[int]:
        """Per-trial attempt index (1-based, grouped by task_name in load order)."""
        counters: dict[str, int] = {}
        attempts: list[int] = []
        for t in self.trials:
            counters[t.task_name] = counters.get(t.task_name, 0) + 1
            attempts.append(counters[t.task_name])
        return attempts
