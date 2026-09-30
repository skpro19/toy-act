# toy-act

A bare-bones ACT policy implemented from scratch to understand CVAEs, action chunking, temporal ensembling, and attention.

## Rollout

Two-camera policy (batch size 64) on the Can task. The rollout uses the checkpoint at step 56,000, which reached 90% success in training-time evaluation.

<img src="assets/rollout/bs64_two_camera_rollout.gif" alt="Two-camera ACT policy rollout on the Can task">

### Training run

<img src="assets/training-run/eval_uniform_panels.png" alt="Evaluation episode horizon and success rate over training steps">

<img src="assets/training-run/batch_metrics_log_scale_uniform_panels.png" alt="Smoothed action, weighted KL, and total losses over training steps (log scale)">

<img src="assets/training-run/denorm_l1_log_scale_uniform_panels.png" alt="Smoothed denormalized joint and gripper L1 errors over training steps (log scale)">

<img src="assets/training-run/latent_log_scale_uniform_panels.png" alt="Latent mu norm and sigma mean over training steps (log scale)">

<details>
<summary>Run details</summary>

`20260929-195938_bs64_lr1e-04_beta0.01_wu0_st100000_imgagentview_image-robot0_eye_in_hand_image_use_z1_l1`

</details>

## Ablations

### Batch size

<img src="assets/rollout-two-camera/two_camera_rollout_grid.gif" alt="Batch-size ablation rollout comparison">

<img src="assets/eval-curves/batch_size_eval_curves.png" alt="Batch-size ablation eval curves">

### Number of cameras

<img src="assets/rollout-camera-ablation/num_cameras_grid.gif" alt="Num-cameras ablation rollout comparison">

<img src="assets/eval-curves/num_cameras_eval_curves.png" alt="Num-cameras ablation eval curves">
