# toy-act

<p align="center">
  <img src="assets/rollout/bs64_two_camera_rollout.gif" alt="Two-camera ACT policy rollout on the Can task">
</p>

## Training curve

<table align="center">
  <thead>
    <tr><th>lr</th><th>bs</th><th>num-steps</th><th>beta</th><th>action-loss</th><th>architecture</th></tr>
  </thead>
  <tbody>
    <tr><td>1e-4</td><td>64</td><td>100000</td><td>0.01</td><td>l1</td><td>cvae encoder-decoder</td></tr>
  </tbody>
</table>

<img src="assets/training-run/eval_uniform_panels.png" alt="Evaluation episode horizon and success rate over training steps">

<img src="assets/training-run/batch_metrics_log_scale_uniform_panels.png" alt="Smoothed action, weighted KL, and total losses over training steps (log scale)">

<img src="assets/training-run/denorm_l1_log_scale_uniform_panels.png" alt="Smoothed denormalized joint and gripper L1 errors over training steps (log scale)">

<img src="assets/training-run/latent_log_scale_uniform_panels.png" alt="Latent mu norm and sigma mean over training steps (log scale)">

## Ablations

<details>
<summary>Batch size</summary>

<img src="assets/eval-curves/batch_size_eval_curves_legend_right.png" alt="Batch-size ablation eval curves">

<p align="center">
  <img src="assets/rollout-two-camera/two_camera_rollout_grid.gif" alt="Batch-size ablation rollout comparison">
</p>

</details>

<details>
<summary>Number of cameras</summary>

<img src="assets/eval-curves/num_cameras_eval_curves_legend_right.png" alt="Num-cameras ablation eval curves">

<p align="center">
  <img src="assets/rollout-camera-ablation/num_cameras_grid.gif" alt="Num-cameras ablation rollout comparison">
</p>

</details>
