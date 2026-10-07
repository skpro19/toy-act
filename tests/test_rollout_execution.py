"""Execution horizons are explicit and independent of prediction length."""

import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from scripts import rollout, train_v2


class RolloutExecutionTest(unittest.TestCase):
    def test_execution_lengths_are_required_and_validated(self) -> None:
        for values in (None, [], [True], [0], [101], [1.5], [10, 10], 10):
            with self.subTest(values=values), self.assertRaises(ValueError):
                rollout.validate_n_action_steps(values=values, action_chunk_size=100)
        self.assertEqual(
            rollout.validate_n_action_steps(values=[10, 100], action_chunk_size=100), [10, 100])
        with self.assertRaises(KeyError):
            train_v2.validate_rollout_flags(
                rollout={"episodes": 30, "horizon": 200, "seed": 42}, action_chunk_size=100)

    def test_replanning_uses_fresh_observations_and_only_chunk_prefix(self) -> None:
        for execution_length, expected_calls in ((10, 20), (100, 2)):
            with self.subTest(execution_length=execution_length):
                env = Mock()
                env.reset.return_value = {"step": 0}
                env.step.side_effect = [
                    ({"step": step + 1}, 0, False, {"is_success": {"task": False}})
                    for step in range(200)
                ]
                chunk = np.arange(100 * 8).reshape(100, 8)
                with patch.object(rollout, "predict_action_chunk", return_value=chunk) as predict:
                    with patch.object(rollout, "model_action_to_sim", return_value=np.zeros(8)) as convert:
                        result = rollout.run_rollout(
                            model=Mock(), env=env, device=torch.device("cpu"),
                            normalization=Mock(), image_keys=("camera",), horizon=200,
                            terminate_on_success=False, render=False, video_writer=None,
                            video_skip=1, action_chunk_size=100, n_action_steps=execution_length)
                self.assertEqual(result["horizon"], 200)
                self.assertEqual(predict.call_count, expected_calls)
                self.assertEqual(
                    [call.kwargs["obs"]["step"] for call in predict.call_args_list],
                    list(range(0, 200, execution_length)))
                for index, call in enumerate(convert.call_args_list):
                    np.testing.assert_array_equal(
                        call.kwargs["model_action"], chunk[index % execution_length])

    def test_no_evaluation_still_accumulates_training_time(self) -> None:
        writer = Mock()
        totals = {"train": 7.0, "eval": 3.0}
        flags = {namespace: True for namespace in train_v2.TENSORBOARD_NAMESPACES}
        flags["eval"] = False
        for global_step in (20, 40):
            with patch.object(train_v2, "run_rollout") as run:
                train_v2.evaluate_and_log_checkpoint(
                    model=Mock(), env=None, dataset=Mock(), image_keys=("camera",),
                    normalization=Mock(), device=torch.device("cpu"), writer=writer,
                    global_step=global_step, run_started=train_v2.time.monotonic() - 60,
                    train_seconds=10, train_steps=20, train_samples=160,
                    cumulative_seconds=totals, n_action_steps=[10], tensorboard=flags,
                    episodes=30, horizon=200, seed=42, action_chunk_size=100)
                run.assert_not_called()
        self.assertEqual(totals, {"train": 27.0, "eval": 3.0})
        metrics = {call.args[0]: call.args[1] for call in writer.add_scalar.call_args_list}
        self.assertEqual(metrics["timing/interval_train_fraction"], 1)
        self.assertEqual(metrics["timing/interval_eval_fraction"], 0)
        self.assertAlmostEqual(metrics["timing/cumulative_eval_fraction"], 0.1)
        self.assertNotIn("timing/checkpoint_seconds", metrics)


if __name__ == "__main__":
    unittest.main()
