#!/usr/bin/env python3
import yaml, sys

with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f)

print("=" * 72)
print("EXPERIMENT BASELINE \u2014 Immutable Config")
print("=" * 72)
print()

print("Branch:        dev")
print("Commit:        f906572c561fec7e7f1206776c7e67b04d872923")
print("Config path:   configs/flywheel/mlp_vision/mlp_vision_instance_BASEv2_seed0.yaml")
print()

sections = [
    ("Run", ["run_name", "arch"]),
    ("Flywheel loop", ["dagger_rounds"]),
    ("Expert data collection", ["num_expert_episodes", "max_steps", "train_capture_hz"]),
    ("Training", ["epochs", "batch_size", "early_stop_patience", "dataloader_workers", "persistent_workers"]),
    ("Dataset mixing", ["expert_ratio", "dagger_intervention_ratio"]),
    ("DAgger rollout", ["intervention_threshold", "intervention_steps", "dagger_episodes", "rollout_max_steps"]),
    ("In-loop evaluation", ["eval_interval", "eval_episodes", "eval_max_steps", "eval_capture_hz", "mode"]),
    ("Held-out evaluation", ["final_eval_episodes", "final_eval_seed", "final_eval_workers", "final_eval_capture_hz"]),
    ("Parallelism", ["workers"]),
    ("Seeds", ["global_seed"]),
]

for section_name, keys in sections:
    print(f"--- {section_name} ---")
    print(f"{'Key':<30} {'Value':<20}")
    print("-" * 50)
    for k in keys:
        v = cfg.get(k, "")
        print(f"{k:<30} {str(v):<20}")
    print()
