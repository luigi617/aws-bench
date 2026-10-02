"""Tests for AWS CLI trajectory metrics (pass^k, recovery rate, error-repair rate, etc.)."""

from __future__ import annotations

from unittest.mock import MagicMock

from harbor.models.trajectories import Trajectory

from aws_bench.metrics.aggregation import _compute_pass_k_all, aggregate_detailed
from aws_bench.metrics.run_data import (
    TrialData,
    _aws_cli_metrics_from_trajectory,
    _aws_operation_key,
    _failure_step_tag,
    _is_invalid_aws_invocation,
)

_AGENT = {"name": "test-agent", "version": "1.0"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _aws_step(
    command: str,
    obs: str = "ok",
    is_error: bool = False,
    step_id: int = 1,
) -> dict:
    """Build a single trajectory step with one bash tool call to an aws command."""
    tc_id = f"tc{step_id}"
    content = "[error] tool reported failure" if is_error else obs
    return {
        "step_id": step_id,
        "source": "agent",
        "message": "running aws",
        "tool_calls": [
            {
                "tool_call_id": tc_id,
                "function_name": "bash",
                "arguments": {"command": command},
            }
        ],
        "observation": {
            "results": [{"source_call_id": tc_id, "content": content}],
        },
    }


def _build_trajectory(steps: list[dict]) -> Trajectory:
    """Build a Trajectory from a list of raw step dicts."""
    return Trajectory.model_validate(
        {"schema_version": "ATIF-v1.7", "agent": _AGENT, "steps": steps}
    )


def _make_trial_data(
    trajectory: Trajectory | None = None,
    reward: float | None = None,
    n_tool_errors: int = 0,  # informational; actual value derived from trajectory
    task_name: str = "test-task",
) -> TrialData:
    """Create a TrialData with a mocked result, following the test_scp_deny pattern."""
    result = MagicMock()
    result.verifier_result = None
    result.exception_info = None
    result.agent_execution = None
    result.started_at = None
    result.finished_at = None
    result.task_name = task_name
    result.trial_name = "trial-1"
    result.compute_token_cost_totals.return_value = (None, None, None, None)

    if reward is not None:
        verifier_result = MagicMock()
        verifier_result.rewards = {"reward": reward}
        result.verifier_result = verifier_result

    return TrialData(
        trial_dir=MagicMock(),
        result=result,
        trajectory=trajectory,
        raw_result=None,
    )


# ---------------------------------------------------------------------------
# Tests: _aws_operation_key
# ---------------------------------------------------------------------------


class TestAwsOperationKey:
    def test_ec2_describe(self) -> None:
        assert _aws_operation_key("aws ec2 describe-instances") == "ec2 describe-instances"

    def test_s3_ls(self) -> None:
        assert _aws_operation_key("aws s3 ls") == "s3 ls"

    def test_iam_create_role_strips_flags(self) -> None:
        assert _aws_operation_key("aws iam create-role --role-name test") == "iam create-role"

    def test_region_flag_skipped(self) -> None:
        key = _aws_operation_key("aws --region us-east-1 ec2 describe-instances")
        assert key == "ec2 describe-instances"

    def test_non_aws_command_returns_none(self) -> None:
        assert _aws_operation_key("ls -la /tmp") is None

    def test_service_only_no_subcommand(self) -> None:
        assert _aws_operation_key("aws s3") == "s3"


# ---------------------------------------------------------------------------
# Tests: _is_invalid_aws_invocation
# ---------------------------------------------------------------------------


class TestIsInvalidAwsInvocation:
    def test_valid_output_returns_false(self) -> None:
        assert _is_invalid_aws_invocation("i-1234567890abcdef0") is False

    def test_invalid_choice_returns_true(self) -> None:
        assert _is_invalid_aws_invocation("Invalid choice: 'bogus-command'") is True

    def test_unknown_options_returns_true(self) -> None:
        assert _is_invalid_aws_invocation("Unknown options: --foo") is True

    def test_no_such_command_returns_true(self) -> None:
        assert _is_invalid_aws_invocation("No such command: 'foo'") is True

    def test_empty_string_returns_false(self) -> None:
        assert _is_invalid_aws_invocation("") is False

    def test_access_denied_error_returns_false(self) -> None:
        assert _is_invalid_aws_invocation("An error occurred (AccessDenied)") is False

    def test_unrecognized_arguments_returns_true(self) -> None:
        assert _is_invalid_aws_invocation("unrecognized arguments: --unknown") is True


# ---------------------------------------------------------------------------
# Tests: _aws_cli_metrics_from_trajectory
# ---------------------------------------------------------------------------


class TestAwsCliMetricsFromTrajectory:
    def test_none_trajectory_returns_defaults(self) -> None:
        m = _aws_cli_metrics_from_trajectory(None)
        assert m["n_aws_cli_calls"] == 0
        assert m["n_invalid_invocations"] == 0
        assert m["error_repair_repaired"] == 0
        assert m["error_repair_total"] == 0
        assert m["looked_before_change"] is None

    def test_no_aws_calls_in_trajectory(self) -> None:
        traj = _build_trajectory(
            [
                {
                    "step_id": 1,
                    "source": "agent",
                    "message": "hi",
                    "tool_calls": [
                        {
                            "tool_call_id": "tc1",
                            "function_name": "bash",
                            "arguments": {"command": "ls -la"},
                        }
                    ],
                    "observation": {"results": [{"source_call_id": "tc1", "content": "file.txt"}]},
                }
            ]
        )
        m = _aws_cli_metrics_from_trajectory(traj)
        assert m["n_aws_cli_calls"] == 0

    def test_aws_read_call_counted(self) -> None:
        traj = _build_trajectory([_aws_step("aws ec2 describe-instances", obs="[]", step_id=1)])
        m = _aws_cli_metrics_from_trajectory(traj)
        assert m["n_aws_cli_calls"] == 1
        assert m["n_invalid_invocations"] == 0
        assert m["looked_before_change"] is None  # no write calls

    def test_aws_write_call_counted(self) -> None:
        traj = _build_trajectory(
            [_aws_step("aws s3api create-bucket --bucket foo", obs="ok", step_id=1)]
        )
        m = _aws_cli_metrics_from_trajectory(traj)
        assert m["n_aws_cli_calls"] == 1

    def test_error_call_tracked(self) -> None:
        traj = _build_trajectory(
            [_aws_step("aws ec2 describe-instances", is_error=True, step_id=1)]
        )
        m = _aws_cli_metrics_from_trajectory(traj)
        assert m["n_aws_cli_calls"] == 1
        assert m["error_repair_total"] == 1
        assert m["error_repair_repaired"] == 0

    def test_error_then_success_counts_repair(self) -> None:
        traj = _build_trajectory(
            [
                _aws_step("aws ec2 describe-instances", is_error=True, step_id=1),
                _aws_step("aws ec2 describe-instances", obs="ok", is_error=False, step_id=2),
            ]
        )
        m = _aws_cli_metrics_from_trajectory(traj)
        assert m["error_repair_total"] == 1
        assert m["error_repair_repaired"] == 1

    def test_looked_before_change_true(self) -> None:
        # Use bare verbs so they match _AWS_READ_VERBS / _AWS_WRITE_VERBS exactly
        traj = _build_trajectory(
            [
                _aws_step("aws ec2 describe", obs="[]", step_id=1),
                _aws_step("aws s3 create", obs="ok", step_id=2),
            ]
        )
        m = _aws_cli_metrics_from_trajectory(traj)
        assert m["looked_before_change"] is True

    def test_looked_before_change_false(self) -> None:
        traj = _build_trajectory(
            [
                _aws_step("aws s3 create", obs="ok", step_id=1),
                _aws_step("aws ec2 describe", obs="[]", step_id=2),
            ]
        )
        m = _aws_cli_metrics_from_trajectory(traj)
        assert m["looked_before_change"] is False

    def test_invalid_invocation_counted(self) -> None:
        traj = _build_trajectory(
            [_aws_step("aws ec2 bogus", obs="Invalid choice: 'bogus'", step_id=1)]
        )
        m = _aws_cli_metrics_from_trajectory(traj)
        assert m["n_invalid_invocations"] == 1


# ---------------------------------------------------------------------------
# Tests: TrialData AWS CLI properties
# ---------------------------------------------------------------------------


class TestTrialDataAwsCliProperties:
    def test_n_aws_cli_calls(self) -> None:
        traj = _build_trajectory(
            [
                _aws_step("aws ec2 describe-instances", step_id=1),
                _aws_step("aws s3 ls", step_id=2),
            ]
        )
        td = _make_trial_data(trajectory=traj)
        assert td.n_aws_cli_calls == 2

    def test_n_invalid_invocations(self) -> None:
        traj = _build_trajectory(
            [_aws_step("aws ec2 bogus", obs="Invalid choice: 'bogus'", step_id=1)]
        )
        td = _make_trial_data(trajectory=traj)
        assert td.n_invalid_invocations == 1

    def test_error_repair_stats_repaired(self) -> None:
        traj = _build_trajectory(
            [
                _aws_step("aws iam create", is_error=True, step_id=1),
                _aws_step("aws iam create", obs="ok", step_id=2),
            ]
        )
        td = _make_trial_data(trajectory=traj)
        repaired, total = td.error_repair_stats
        assert total == 1
        assert repaired == 1

    def test_looked_before_change_none_when_no_writes(self) -> None:
        traj = _build_trajectory([_aws_step("aws ec2 describe-instances", step_id=1)])
        td = _make_trial_data(trajectory=traj)
        assert td.looked_before_change is None

    def test_looked_before_change_true(self) -> None:
        # Bare-verb subcommands are required to match _AWS_READ_VERBS / _AWS_WRITE_VERBS
        traj = _build_trajectory(
            [
                _aws_step("aws ec2 describe", step_id=1),
                _aws_step("aws iam create", step_id=2),
            ]
        )
        td = _make_trial_data(trajectory=traj)
        assert td.looked_before_change is True

    def test_looked_before_change_false(self) -> None:
        traj = _build_trajectory(
            [
                _aws_step("aws iam create", step_id=1),
                _aws_step("aws ec2 describe", step_id=2),
            ]
        )
        td = _make_trial_data(trajectory=traj)
        assert td.looked_before_change is False

    def test_failure_step_discover_via_property(self) -> None:
        """No aws calls, failing trial -> 'discover'."""
        td = _make_trial_data(trajectory=None, reward=0.0)
        assert td.failure_step == "discover"

    def test_failure_step_none_for_passing_trial(self) -> None:
        td = _make_trial_data(trajectory=None, reward=1.0)
        assert td.failure_step is None


# ---------------------------------------------------------------------------
# Tests: _failure_step_tag (direct)
# ---------------------------------------------------------------------------


class TestFailureStepTag:
    def _aws_metrics(
        self,
        n_aws: int = 0,
        n_invalid: int = 0,
        repaired: int = 0,
        total_errors: int = 0,
    ) -> dict:
        return {
            "n_aws_cli_calls": n_aws,
            "n_invalid_invocations": n_invalid,
            "error_repair_repaired": repaired,
            "error_repair_total": total_errors,
            "looked_before_change": None,
        }

    def test_passing_returns_none(self) -> None:
        m = self._aws_metrics(n_aws=3)
        assert _failure_step_tag(m, n_tool_errors=0, n_llm_calls=5, reward=1.0) is None

    def test_discover_no_aws_calls(self) -> None:
        m = self._aws_metrics(n_aws=0)
        assert _failure_step_tag(m, n_tool_errors=0, n_llm_calls=5, reward=0.0) == "discover"

    def test_discover_reward_none_treated_as_failing(self) -> None:
        m = self._aws_metrics(n_aws=0)
        assert _failure_step_tag(m, n_tool_errors=0, n_llm_calls=0, reward=None) == "discover"

    def test_invoke_invalid_invocations(self) -> None:
        m = self._aws_metrics(n_aws=2, n_invalid=1)
        assert _failure_step_tag(m, n_tool_errors=0, n_llm_calls=5, reward=0.0) == "invoke"

    def test_recover_errors_not_fully_repaired(self) -> None:
        m = self._aws_metrics(n_aws=3, n_invalid=0, repaired=0, total_errors=2)
        assert _failure_step_tag(m, n_tool_errors=2, n_llm_calls=5, reward=0.0) == "recover"

    def test_verify_stop_too_many_llm_calls(self) -> None:
        m = self._aws_metrics(n_aws=3, n_invalid=0, repaired=0, total_errors=0)
        assert _failure_step_tag(m, n_tool_errors=0, n_llm_calls=16, reward=0.0) == "verify_stop"

    def test_interpret_aws_calls_no_tool_errors(self) -> None:
        m = self._aws_metrics(n_aws=3, n_invalid=0, repaired=0, total_errors=0)
        assert _failure_step_tag(m, n_tool_errors=0, n_llm_calls=5, reward=0.0) == "interpret"

    def test_continue_non_aws_tool_errors(self) -> None:
        # n_aws > 0, total_errors = 0, n_llm_calls <= 15, but n_tool_errors > 0
        m = self._aws_metrics(n_aws=3, n_invalid=0, repaired=0, total_errors=0)
        assert _failure_step_tag(m, n_tool_errors=2, n_llm_calls=5, reward=0.0) == "continue"


# ---------------------------------------------------------------------------
# Tests: _compute_pass_k_all
# ---------------------------------------------------------------------------


class TestComputePassKAll:
    def _make_trials(self, rewards_by_task: dict[str, list[float | None]]) -> list[TrialData]:
        trials = []
        for task_name, rewards in rewards_by_task.items():
            for reward in rewards:
                trials.append(_make_trial_data(reward=reward, task_name=task_name))
        return trials

    def test_k1_equals_accuracy(self) -> None:
        # task A: 1/2, task B: 2/2 -> avg = (0.5 + 1.0) / 2 = 0.75
        trials = self._make_trials({"taskA": [1.0, 0.0], "taskB": [1.0, 1.0]})
        result = _compute_pass_k_all(trials, k=1)
        assert result is not None
        assert abs(result - 0.75) < 1e-9

    def test_k3_all_pass(self) -> None:
        # 1 task, 3 trials, all succeed -> comb(3,3)/comb(3,3) = 1.0
        trials = self._make_trials({"taskA": [1.0, 1.0, 1.0]})
        result = _compute_pass_k_all(trials, k=3)
        assert result == 1.0

    def test_k3_none_pass(self) -> None:
        # 1 task, 3 trials, none succeed -> comb(0,3)/comb(3,3) = 0.0
        trials = self._make_trials({"taskA": [0.0, 0.0, 0.0]})
        result = _compute_pass_k_all(trials, k=3)
        assert result == 0.0

    def test_k3_fewer_than_k_attempts_returns_none(self) -> None:
        # 1 task, 2 trials, k=3 -> n < k -> task skipped -> returns None
        trials = self._make_trials({"taskA": [1.0, 1.0]})
        result = _compute_pass_k_all(trials, k=3)
        assert result is None

    def test_empty_trials_returns_none(self) -> None:
        assert _compute_pass_k_all([], k=1) is None

    def test_k3_one_of_three_passes(self) -> None:
        # comb(1,3) = 0, so pass^3 = 0
        trials = self._make_trials({"taskA": [1.0, 0.0, 0.0]})
        result = _compute_pass_k_all(trials, k=3)
        assert result == 0.0

    def test_k1_single_passing_trial(self) -> None:
        trials = self._make_trials({"taskA": [1.0]})
        result = _compute_pass_k_all(trials, k=1)
        assert result == 1.0

    def test_k1_single_failing_trial(self) -> None:
        trials = self._make_trials({"taskA": [0.0]})
        result = _compute_pass_k_all(trials, k=1)
        assert result == 0.0


# ---------------------------------------------------------------------------
# Tests: aggregate_detailed new metrics
# ---------------------------------------------------------------------------


class TestAggregateDetailedNewMetrics:
    def test_recovery_rate_with_errors(self) -> None:
        # 2 trials both with tool errors: 1 recovered, 1 did not
        traj_with_error = _build_trajectory(
            [_aws_step("aws ec2 describe-instances", is_error=True, step_id=1)]
        )
        td1 = _make_trial_data(trajectory=traj_with_error, reward=1.0)  # recovered
        td2 = _make_trial_data(trajectory=traj_with_error, reward=0.0)  # not recovered
        td3 = _make_trial_data(trajectory=None, reward=0.0)  # no errors
        metrics = aggregate_detailed([td1, td2, td3])
        assert metrics["n_trials_with_errors"] == 2
        assert metrics["n_trials_recovered"] == 1
        assert abs(metrics["recovery_rate"] - 0.5) < 1e-9

    def test_recovery_rate_none_when_no_tool_errors(self) -> None:
        trials = [_make_trial_data(reward=1.0), _make_trial_data(reward=0.0)]
        metrics = aggregate_detailed(trials)
        assert metrics["recovery_rate"] is None
        assert metrics["n_trials_with_errors"] == 0
        assert metrics["n_trials_recovered"] == 0

    def test_error_repair_rate_full_repair(self) -> None:
        traj = _build_trajectory(
            [
                _aws_step("aws iam create", is_error=True, step_id=1),
                _aws_step("aws iam create", obs="ok", step_id=2),
            ]
        )
        td = _make_trial_data(trajectory=traj, reward=0.0)
        metrics = aggregate_detailed([td])
        assert metrics["n_error_repair_total"] == 1
        assert metrics["n_error_repair_repaired"] == 1
        assert metrics["error_repair_rate"] == 1.0

    def test_error_repair_rate_no_errors(self) -> None:
        trials = [_make_trial_data(reward=1.0)]
        metrics = aggregate_detailed(trials)
        assert metrics["error_repair_rate"] is None
        assert metrics["n_error_repair_total"] == 0
        assert metrics["n_error_repair_repaired"] == 0

    def test_invalid_invocation_rate(self) -> None:
        traj = _build_trajectory(
            [
                _aws_step("aws ec2 bogus", obs="Invalid choice: 'bogus'", step_id=1),
                _aws_step("aws ec2 describe-instances", obs="ok", step_id=2),
            ]
        )
        td = _make_trial_data(trajectory=traj, reward=0.0)
        metrics = aggregate_detailed([td])
        assert metrics["n_aws_cli_calls_total"] == 2
        assert metrics["n_invalid_invocations_total"] == 1
        assert abs(metrics["invalid_invocation_rate"] - 0.5) < 1e-9

    def test_invalid_invocation_rate_none_when_no_aws_calls(self) -> None:
        trials = [_make_trial_data(reward=1.0)]
        metrics = aggregate_detailed(trials)
        assert metrics["invalid_invocation_rate"] is None
        assert metrics["n_aws_cli_calls_total"] == 0
        assert metrics["n_invalid_invocations_total"] == 0

    def test_look_before_change_rate_none_when_no_write_calls(self) -> None:
        # All trials have only read ops -> looked_before_change is None for all
        traj = _build_trajectory([_aws_step("aws ec2 describe-instances", step_id=1)])
        td = _make_trial_data(trajectory=traj)
        metrics = aggregate_detailed([td])
        assert metrics["look_before_change_rate"] is None
        assert metrics["n_mutation_trials"] == 0

    def test_look_before_change_rate(self) -> None:
        # trial A: read then write -> True; trial B: write only -> False
        # Use bare-verb subcommands to match _AWS_READ_VERBS / _AWS_WRITE_VERBS
        traj_a = _build_trajectory(
            [
                _aws_step("aws ec2 describe", step_id=1),
                _aws_step("aws iam create", step_id=2),
            ]
        )
        traj_b = _build_trajectory([_aws_step("aws iam create", step_id=1)])
        td_a = _make_trial_data(trajectory=traj_a)
        td_b = _make_trial_data(trajectory=traj_b)
        metrics = aggregate_detailed([td_a, td_b])
        assert metrics["n_mutation_trials"] == 2
        assert abs(metrics["look_before_change_rate"] - 0.5) < 1e-9

    def test_failure_step_distribution_discover(self) -> None:
        # Failing trial, no aws calls -> tagged as "discover"
        td = _make_trial_data(trajectory=None, reward=0.0)
        metrics = aggregate_detailed([td])
        dist = metrics["failure_step_distribution"]
        assert dist["discover"] == 1
        assert metrics["n_tagged_failures"] == 1

    def test_failure_step_distribution_passing_not_tagged(self) -> None:
        td = _make_trial_data(trajectory=None, reward=1.0)
        metrics = aggregate_detailed([td])
        assert metrics["n_tagged_failures"] == 0
        dist = metrics["failure_step_distribution"]
        assert all(v == 0 for v in dist.values())

    def test_failure_step_distribution_contains_all_keys(self) -> None:
        td = _make_trial_data(reward=0.0)
        metrics = aggregate_detailed([td])
        expected_keys = {"discover", "invoke", "recover", "interpret", "verify_stop", "continue"}
        assert set(metrics["failure_step_distribution"].keys()) == expected_keys

    def test_cost_per_successful_task_none_when_no_cost(self) -> None:
        # trajectory_cost_usd is None (no cost metadata in trajectory)
        td = _make_trial_data(reward=1.0)
        metrics = aggregate_detailed([td])
        assert metrics["cost_per_successful_task"] is None

    def test_cost_per_successful_task_none_when_no_success(self) -> None:
        td = _make_trial_data(reward=0.0)
        metrics = aggregate_detailed([td])
        assert metrics["cost_per_successful_task"] is None
