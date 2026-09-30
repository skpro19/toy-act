# toy-act

A toy ACT policy implemented from scratch to understand CVAEs, action chunking, temporal ensembling, and attention.

<p align="center">
  <img src="assets/rollout/bs64_two_camera_rollout.gif" alt="Two-camera ACT policy rollout on the Can task">
</p>

## Training curve

| lr | bs | num-steps | beta | action-loss |
| --- | --- | --- | --- | --- |
| 1e-4 | 64 | 100000 | 0.01 | l1 |

<img src="assets/training-run/eval_uniform_panels.png" alt="Evaluation episode horizon and success rate over training steps">

<img src="assets/training-run/batch_metrics_log_scale_uniform_panels.png" alt="Smoothed action, weighted KL, and total losses over training steps (log scale)">

<img src="assets/training-run/denorm_l1_log_scale_uniform_panels.png" alt="Smoothed denormalized joint and gripper L1 errors over training steps (log scale)">

<img src="assets/training-run/latent_log_scale_uniform_panels.png" alt="Latent mu norm and sigma mean over training steps (log scale)">

## Ablations

### Batch size

<p align="center">
  <img src="assets/rollout-two-camera/two_camera_rollout_grid.gif" alt="Batch-size ablation rollout comparison">
</p>

<img src="assets/eval-curves/batch_size_eval_curves_legend_right.png" alt="Batch-size ablation eval curves">

### Number of cameras

<p align="center">
  <img src="assets/rollout-camera-ablation/num_cameras_grid.gif" alt="Num-cameras ablation rollout comparison">
</p>

<img src="assets/eval-curves/num_cameras_eval_curves_legend_right.png" alt="Num-cameras ablation eval curves">
