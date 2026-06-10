
import argparse
import os
import re
import glob

import numpy as np
import gymnasium as gym
import panda_gym  
from stable_baselines3 import SAC, PPO

STEP_RE = re.compile(r"_(\d+)_steps\.zip$")


def find_checkpoints(checkpoints_dir: str, prefix: str):
    """Return [(step, path), ...] sorted by step for one prefix.

    Searches checkpoints_dir recursively, so udr/adr checkpoints can live in
    separate sub-folders (e.g. udr_steps/, adr_steps/) under one parent.
    """
    pattern = os.path.join(checkpoints_dir, "**", f"{prefix}_*_steps.zip")
    found = []
    for path in glob.glob(pattern, recursive=True):
        m = STEP_RE.search(os.path.basename(path))
        if m:
            found.append((int(m.group(1)), path))
    found.sort(key=lambda x: x[0])
    return found


def load_model(path: str):
    name = os.path.basename(path).lower()
    if "sac" in name:
        return SAC.load(path)
    if "ppo" in name:
        return PPO.load(path)
    raise ValueError(f"Cannot determine algo (sac/ppo) from filename: {path}")


def eval_checkpoint(path: str, env, n_episodes: int, deterministic: bool):
    model = load_model(path)
    returns, successes = [], []
    for episode in range(1, n_episodes + 1):
        obs, info = env.reset(seed=episode)  
        terminated = truncated = False
        ep_return = 0.0
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += float(reward)
        returns.append(ep_return)
        if isinstance(info, dict) and "is_success" in info:
            successes.append(float(info["is_success"]))
    returns = np.array(returns, dtype=np.float32)
    succ = float(np.mean(successes)) if successes else float("nan")
    return float(returns.mean()), float(returns.std()), succ


def build_curve(prefix: str, args):
    ckpts = find_checkpoints(args.checkpoints_dir, prefix)
    if not ckpts:
        print(f" no checkpoints found for prefix '{prefix}' in {args.checkpoints_dir}")
        return None

    env = gym.make("PandaPush-v3", render_mode="rgb_array",
                   reward_type="dense", type=args.env_type)

    steps, means, stds, succ_rates = [], [], [], []
    for step, path in ckpts:
        mean_r, std_r, succ = eval_checkpoint(path, env, args.episodes,
                                              deterministic=not args.stochastic)
        steps.append(step)
        means.append(mean_r)
        stds.append(std_r)
        succ_rates.append(succ)
        print(f"  {prefix} | step={step:>8} | mean_return={mean_r:7.3f} "
              f"+/- {std_r:5.3f} | success={succ:.2%}")
    env.close()
    return {
        "prefix": prefix,
        "steps": np.array(steps),
        "means": np.array(means),
        "stds": np.array(stds),
        "succ": np.array(succ_rates),
    }


def main():
    p = argparse.ArgumentParser(description="Reconstruct learning/transfer curves from checkpoints")
    p.add_argument("--checkpoints-dir", default=os.path.join("results", "checkpoints"))
    p.add_argument("--prefix", required=True,
                   help="checkpoint name prefix, or comma-separated list to overlay")
    p.add_argument("--env-type", default="target", choices=["source", "target"],
                   help="env to evaluate on (target = transfer curve)")
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--stochastic", action="store_true", help="use stochastic policy")
    p.add_argument("--out", default="learning_curve", help="output basename (.csv/.png)")
    p.add_argument("--baseline-model", default=None,
                   help="comma-separated final-model path(s) to draw as flat reference "
                        "lines (e.g. results/sac_push_none_source_1000k) — for strategies "
                        "with no checkpoint series, like the 'none' lower bound")
    args = p.parse_args()

    prefixes = [s.strip() for s in args.prefix.split(",") if s.strip()]
    curves = []
    for prefix in prefixes:
        print(f"\nEvaluating checkpoints for: {prefix}  (env-type={args.env_type})")
        c = build_curve(prefix, args)
        if c is not None:
            curves.append(c)

    baselines = []
    if args.baseline_model:
        bpaths = [s.strip() for s in args.baseline_model.split(",") if s.strip()]
        benv = gym.make("PandaPush-v3", render_mode="rgb_array",
                        reward_type="dense", type=args.env_type)
        for bpath in bpaths:
            load_path = bpath if bpath.endswith(".zip") else bpath + ".zip"
            if not os.path.exists(load_path):
                print(f"[WARN] baseline model not found: {load_path}")
                continue
            mean_r, std_r, succ = eval_checkpoint(load_path, benv, args.episodes,
                                                  deterministic=not args.stochastic)
            label = os.path.basename(bpath).replace(".zip", "")
            baselines.append({"label": label, "mean": mean_r, "std": std_r, "succ": succ})
            print(f"  baseline {label} | mean_return={mean_r:7.3f} +/- {std_r:5.3f} "
                  f"| success={succ:.2%}")
        benv.close()

    if not curves and not baselines:
        print("No curves or baselines produced — nothing to save.")
        return

    csv_path = args.out + ".csv"
    with open(csv_path, "w") as f:
        f.write("prefix,step,mean_return,std_return,success_rate\n")
        for c in curves:
            for i in range(len(c["steps"])):
                f.write(f"{c['prefix']},{c['steps'][i]},{c['means'][i]:.5f},"
                        f"{c['stds'][i]:.5f},{c['succ'][i]:.5f}\n")
        for b in baselines:
            f.write(f"{b['label']} (baseline),final,{b['mean']:.5f},"
                    f"{b['std']:.5f},{b['succ']:.5f}\n")
    print(f"\nWrote {csv_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(7, 5))
        for c in curves:
            plt.plot(c["steps"], c["means"], marker="o", label=c["prefix"])
            plt.fill_between(c["steps"], c["means"] - c["stds"],
                             c["means"] + c["stds"], alpha=0.15)
        for b in baselines:
            plt.axhline(b["mean"], linestyle="--", alpha=0.8,
                        label=f"{b['label']} (final)")
        plt.xlabel("Training timesteps")
        plt.ylabel(f"Mean return over {args.episodes} episodes ({args.env_type})")
        plt.title(f"Evaluation curve on {args.env_type} environment")
        plt.legend()
        plt.grid(True, alpha=0.3)
        png_path = args.out + ".png"
        plt.tight_layout()
        plt.savefig(png_path, dpi=150)
        print(f"Wrote {png_path}")
    except ImportError:
        print("matplotlib not installed — wrote CSV only (plot it yourself).")


if __name__ == "__main__":
    main()
