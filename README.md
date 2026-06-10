# RL Project — Group 24 (FAIML 01VSDWS)

Course project for *Foundations of Artificial Intelligence and Machine Learning* (FAIML 01VSDWS), Politecnico di Torino. The overall theme is **sim-to-real transfer via domain randomization**, studied in a sim-to-sim setting.

This repository is split in two parts:

- **Part 1 — Hopper (this part).** Implement policy-gradient control on the MuJoCo `Hopper-v4` environment from scratch: REINFORCE (with and without a baseline) and Actor-Critic.
- **Part 2 — PandaGym Push (work in progress).** Train a robotic-arm pushing policy with stable-baselines3 (PPO/SAC) and apply Uniform/Automatic Domain Randomization. *Left for further implementation — see [Part 2](#part-2--pandagym-push-work-in-progress).*

Official assignment at [Google Doc](https://docs.google.com/document/d/1AXgLXux3l69vDAPLL-UYD3luFOw3JbyR-pLCS2yuNZk/edit?usp=sharing).

## Part 1 — Hopper

We train a one-legged `Hopper-v4` agent (11-D observation, 3-D continuous torque action in `[-1, 1]`) to hop forward as far as possible. Everything is implemented from scratch in PyTorch — no RL library is used for the agent itself.

**What we built**

- **`agent.py`** — the policy and the learning algorithm:
  - `Policy`: separate actor and critic MLPs (two `tanh` hidden layers each, orthogonal init). The actor outputs the mean of a Gaussian over actions with a learnable, state-independent log-std; actions are `tanh`-squashed into `[-1, 1]` with the corresponding log-prob (change-of-variables) correction.
  - `Agent`: supports `algorithm='reinforce'` and `algorithm='actor_critic'`, reward-to-go returns, baselines (`none` / `constant` / `ema`), optional entropy bonus, advantage normalization, gradient clipping, and separate actor/critic learning rates.
- **`train.py`** — the training loop. `main(...)` exposes the hyperparameters (algorithm, baseline, seed, learning rates, hidden size, γ, entropy weight, …) and saves the trained model plus the return/timing history into `results/`.
- **`part1.ipynb`** — the experiment driver: runs the REINFORCE baseline study, the Actor-Critic hyperparameter search, the refined multi-seed Actor-Critic runs, and produces all the plots in the report.
- **`visualize.py`** — renders deterministic rollouts of saved checkpoints to `.mp4`/`.gif`.
- **`test_random_policy.py`** — sanity-check script: rolls out a random policy to get familiar with the environment.

**Tasks**

1. **Environment inspection** — explore the Hopper observation/action spaces and episode termination (`test_random_policy.py`).
2. **REINFORCE** — Monte Carlo policy gradient from scratch, comparing no baseline, a constant baseline, and an exponential-moving-average (EMA) baseline.
3. **Actor-Critic** — a one-step TD Actor-Critic, then a refined configuration tuned via a small hyperparameter search and evaluated across seeds.

## How to run

Everything for Part 1 lives in **`part1/part1.ipynb`** — open it and run the cells top to bottom. The notebook is self-contained:

- Its **first cell installs the dependencies** (`%pip install -r ../requirements.txt`), so there is no separate setup step.
- It then runs the whole study end to end: Task 1 (inspecting the Hopper body masses), the vanilla REINFORCE run, the constant- and EMA-baseline experiments, the Actor-Critic hyperparameter sweep and its refinement, the multi-seed comparisons, and the time-limit (truncation) ablation — producing every plot used in the report along the way.

Trained checkpoints and reward/timing histories are written to `part1/results/`. Running on a Linux machine lets you also render the MuJoCo Hopper to watch a policy in action.

The notebook drives two helper modules directly: `agent.py` (the policy and learning algorithm) and `train.py` (whose `main(...)` runs one training job for a given algorithm/baseline/seed/hyperparameters). `test_random_policy.py` is a small standalone sanity check that rolls out a random policy.

**Visualizing a trained policy**

To re-record the demo clips, run from the `part1/` directory:

```bash
python visualize.py
```

This rolls out 3 deterministic evaluation episodes (the policy mean, no exploration noise) for each checkpoint in `MODEL_PATHS` and writes `.mp4` + `.gif` clips into `part1/videos/`. On a headless machine, prefix the command with `MUJOCO_GL=egl`.

## Trained policies in action

A few of the Hopper policies we trained in Part 1, rolled out in the MuJoCo `Hopper-v4` environment:

<table>
  <tr>
    <th align="center">REINFORCE (vanilla)</th>
    <th align="center">REINFORCE + EMA baseline (&alpha;=0.05)</th>
    <th align="center">Refined Actor-Critic</th>
  </tr>
  <tr>
    <td align="center"><img src="part1/videos/model_none.gif" width="240" alt="REINFORCE vanilla Hopper"></td>
    <td align="center"><img src="part1/videos/model_ema_a0.0500.gif" width="240" alt="REINFORCE + EMA baseline Hopper"></td>
    <td align="center"><img src="part1/videos/model_actor_critic_refined_seed42.gif" width="240" alt="Refined Actor-Critic Hopper"></td>
  </tr>
  <tr>
    <td align="center"><b>114 &plusmn; 104</b></td>
    <td align="center"><b>354 &plusmn; 173</b></td>
    <td align="center"><b>589 &plusmn; 297</b></td>
  </tr>
  <tr>
    <td align="center">Collapses within a few steps and barely moves.</td>
    <td align="center">Hops forward, but with an unstable gait.</td>
    <td align="center">Longest, smoothest, most stable hopping &mdash; our best policy.</td>
  </tr>
</table>

Returns are the final-50-episode mean &plusmn; std over 9 seeds (`[0, 1, 2, 3, 4, 5, 6, 7, 42]`).

Each clip maps to the following checkpoint:

| Clip | Checkpoint | Hidden size |
| --- | --- | --- |
| REINFORCE (vanilla) | `part1/results/model_none.pt` | 64 |
| REINFORCE + EMA baseline (&alpha;=0.05) | `part1/results/model_ema_a0.0500.pt` | 64 |
| Refined Actor-Critic | `part1/results/model_actor_critic_refined_seed42.pt` | 256 |

The clips were recorded over 3 deterministic evaluation episodes (the policy mean, with no exploration noise). They can be regenerated from the `part1/` directory with:

```bash
python visualize.py
```

The clips are also saved as full-quality `.mp4` files alongside the GIFs in `part1/videos/` for reference.

## Part 2 — PandaGym Push (work in progress)

Part 2 tackles the PandaGym `Push` task with stable-baselines3 (PPO/SAC) and domain randomization (UDR/ADR) on the cube mass. It is **not yet implemented in this branch** and is left for further work — the files currently in `part2/` are the professor's starting template (e.g. `test_random_policy.py` is an example random-policy rollout on `PandaPush-v3`). To set up the panda-gym package once we start:

```bash
cd part2/panda-gym
pip install -e .
```

## Project structure

```
RL-Project-Group24/
├── README.md
├── requirements.txt
├── part1/                      <-- Hopper (REINFORCE / Actor-Critic)
│   ├── part1.ipynb             <-- run this: all experiments + plots
│   ├── agent.py                <-- Policy + Agent (the algorithm)
│   ├── train.py                <-- training loop / main()
│   ├── visualize.py            <-- render checkpoints to mp4/gif
│   ├── test_random_policy.py   <-- random-policy sanity check
│   ├── results/                <-- saved checkpoints & histories
│   └── videos/                 <-- demo clips (Git LFS)
└── part2/                      <-- PushTask (work in progress; professor's template)
    ├── eval_sb3.py
    ├── rand_wrapper.py         <-- randomization wrapper for UDR/ADR
    ├── test_random_policy.py   <-- professor's example (random policy on PandaPush-v3)
    ├── train_sb3.py
    └── panda-gym/
        └── panda_gym/ (main package)
            └── envs/
                ├── core.py
                ├── panda_tasks.py
                ├── robots/
                │   └── panda.py
                └── tasks/
                    ├── flip.py
                    ├── pick_and_place.py
                    ├── push.py     <-- the environment used in Part 2
                    ├── reach.py
                    ├── slide.py
                    └── stack.py
```
