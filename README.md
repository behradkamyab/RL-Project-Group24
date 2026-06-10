# RL Project — Group 24 (FAIML 01VSDWS)

Course project for *Foundations of Artificial Intelligence and Machine Learning* (FAIML 01VSDWS), Politecnico di Torino. The overall theme is **sim-to-real transfer via domain randomization**, studied in a sim-to-sim setting.

This repository is split in two parts:

- **Part 1 — Hopper.** Implement policy-gradient control on the MuJoCo `Hopper-v4` environment from scratch: REINFORCE (with and without a baseline) and Actor-Critic. See [Part 1](#part-1--hopper).
- **Part 2 — PandaGym Push.** Train a robotic-arm pushing policy with stable-baselines3 (PPO/SAC) and study sim-to-real transfer with Uniform and Automatic Domain Randomization (UDR/ADR) over the cube mass. See [Part 2](#part-2--pandagym-push).

Official assignment at [Google Doc](https://docs.google.com/document/d/1AXgLXux3l69vDAPLL-UYD3luFOw3JbyR-pLCS2yuNZk/edit?usp=sharing).

## Part 1 — Hopper

We train a one-legged `Hopper-v4` agent (11-D observation, 3-D continuous torque action in `[-1, 1]`) to hop forward as far as possible. Everything is implemented from scratch in PyTorch — no RL library is used for the agent itself.

**What we built**

- **`agent.py`** — the policy and the learning algorithm:
  - `Policy`: separate actor and critic MLPs (two `tanh` hidden layers each, orthogonal init). The actor outputs the mean of a Gaussian over actions with a learnable, state-independent log-std; actions are `tanh`-squashed into `[-1, 1]` with the corresponding log-prob (change-of-variables) correction.
  - `Agent`: supports `algorithm='reinforce'` and `algorithm='actor_critic'`, reward-to-go returns, baselines (`none` / `constant` / `ema`), optional entropy bonus, advantage normalization, gradient clipping, and separate actor/critic learning rates.
- **`train.py`** — the training loop. `main(...)` exposes the hyperparameters (algorithm, baseline, seed, learning rates, hidden size, γ, entropy weight, …) and saves the trained model plus the return/timing history.
- **`part1.ipynb`** — the experiment driver: runs the REINFORCE baseline study, the Actor-Critic hyperparameter search, the refined multi-seed Actor-Critic runs, and produces all the plots in the report.
- **`visualize.py`** — renders deterministic rollouts of saved checkpoints to `.mp4`/`.gif`.
- **`test_random_policy.py`** — sanity-check script: rolls out a random policy to get familiar with the environment.

**Tasks**

1. **Environment inspection** — explore the Hopper observation/action spaces and episode termination (`test_random_policy.py`).
2. **REINFORCE** — Monte Carlo policy gradient from scratch, comparing no baseline, a constant baseline, and an exponential-moving-average (EMA) baseline.
3. **Actor-Critic** — a one-step TD Actor-Critic. We call the best configuration from the 12-trial hyperparameter search the **tuned Actor-Critic**, and the version obtained by applying three stability-oriented changes to it — a lower actor learning rate (3.9→3.5×10⁻⁵), tighter gradient clipping (0.5→0.3), and advantage normalization — the **refined Actor-Critic**. All methods are evaluated across 9 seeds.

## How to run

Everything for Part 1 lives in **`part1/part1.ipynb`** — open it and run the cells top to bottom. The notebook is self-contained:

- Its **first cell installs the dependencies** (`%pip install -r ../requirements.txt`), so there is no separate setup step.
- It then runs the whole study end to end: Task 1 (inspecting the Hopper body masses), the vanilla REINFORCE run, the constant- and EMA-baseline experiments, the Actor-Critic hyperparameter sweep and its refinement, the multi-seed comparisons, and the time-limit (truncation) ablation — producing every plot used in the report along the way.

Running on a Linux machine lets you also render the MuJoCo Hopper to watch a policy in action.

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
    <th align="center" width="240">REINFORCE (vanilla)</th>
    <th align="center" width="240">REINFORCE + EMA baseline (&alpha;=0.0005)</th>
    <th align="center" width="240">Refined Actor-Critic</th>
  </tr>
  <tr>
    <td align="center"><img src="part1/videos/model_none.gif" width="240" alt="REINFORCE vanilla Hopper"></td>
    <td align="center"><img src="part1/videos/model_ema_a0.0005.gif" width="240" alt="REINFORCE + EMA baseline Hopper"></td>
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
    <td align="center">Longest, smoothest gait; best mean, but on par with EMA.</td>
  </tr>
</table>

Returns are the final-50-episode mean &plusmn; std over 9 seeds (`[0, 1, 2, 3, 4, 5, 6, 7, 42]`).

Network size for each clip:

| Clip | Hidden size |
| --- | --- |
| REINFORCE (vanilla) | 64 |
| REINFORCE + EMA baseline (&alpha;=0.0005) | 64 |
| Refined Actor-Critic | 256 |

The clips were recorded over 3 deterministic evaluation episodes (the policy mean, with no exploration noise). They can be regenerated from the `part1/` directory with:

```bash
python visualize.py
```

The clips are also saved as full-quality `.mp4` files alongside the GIFs in `part1/videos/` for reference.

## Part 2 — PandaGym Push

Part 2 studies sim-to-real transfer on the `PandaPush-v3` task using Stable-Baselines3. The robot is a Franka-Emika Panda arm that must push a cube to a target position using a dense reward.

The project uses two versions of the task:

- `source`: the cube mass is `1.0 kg`.
- `target`: the cube mass is `5.0 kg`.

The source environment represents the simulator used for training, while the target environment represents the heavier "real" environment used for transfer testing. A policy trained only on the light source cube can still succeed on the heavier target cube, but the transfer is usually less efficient and gives worse returns. To reduce this source-to-target gap, we apply domain randomization over the cube mass during training.

### What we built

**`train_sb3.py`** — training script for PPO and SAC on `PandaPush-v3`.

Main options:

```bash
--algo sac|ppo
--sampling-strategy none|udr|adr
--env-type source|target
--timesteps
--seed
--net-arch
--learning-rate
--mass-range MIN MAX
--resume
```

In the final experiments, we used SAC with 1M training steps. For UDR and ADR, the mass range was `--mass-range 0.5 6.0`. SAC uses gSDE exploration and a replay buffer. The default network architecture used in the final experiments is `512,512`.

**`rand_wrapper.py`** — implements `RandomizationWrapper`, a `gym.Wrapper` that changes the cube mass at the beginning of each episode.

Supported modes:

- `none`: no mass randomization.
- `udr`: Uniform Domain Randomization. The cube mass is sampled uniformly from the full mass range at every episode.
- `adr`: Automatic Domain Randomization-style curriculum. The mass range starts near the source mass and is adapted during training according to recent success. When the policy performs well, the range is expanded; when performance drops, the range can be reduced.

In the final Task 6 runs, both UDR and ADR used the global mass range `[0.5, 6.0] kg`. This range includes masses below and above the source mass and also covers the heavier target mass. The goal is to train a policy that is robust to a broad set of cube masses, rather than a policy specialized only to the source mass.

**`eval_sb3.py`** — evaluation script for trained models. It loads a saved model and evaluates it on either the source or target environment.

Main options:

```bash
--model-path
--episodes
--env-type source|target
--seed-offset
```

It reports mean return, return standard deviation, return variance, minimum return, maximum return, and success rate. For the final report, each model was evaluated for 50 episodes and repeated over three seed offsets: `0`, `1000`, and `2000`.

**`run_multi_seed.sh`** — runs `eval_sb3.py` three times using seed offsets `0`, `1000`, and `2000`, and writes the output to `part2/evals_report/`.

```bash
bash run_multi_seed.sh results/sac_push_udr_source_1000k target
```

**`learning_curve.py`** — builds learning curves from saved checkpoints. It evaluates all checkpoint files matching a prefix and produces a `.png` plot. It can also overlay final baseline models as horizontal reference lines using `--baseline-model`. This was used to create the Task 6 target learning curve comparing UDR and ADR checkpoints over training.

**`test_random_policy.py`** — runs a random policy on `PandaPush-v3`. Used only to inspect the environment and understand the source/target task setup.

### Setup

The `part2/panda-gym/` package is **provided by the course professor** — it is a modified panda-gym in which `PandaPush-v3` accepts a `type="source"|"target"` argument that sets the cube mass (`1.0 kg` vs `5.0 kg`). We use it as given; our work is the training, randomization, and evaluation code in `part2/`. Install it in editable mode, then the project requirements:

```bash
cd part2/panda-gym
pip install -e .
cd ..
pip install -r ../requirements.txt
```

### How to run

**Task 4 — PPO vs SAC.** Train PPO and SAC on the source environment for 500k steps, then evaluate both on the source environment:

```bash
python train_sb3.py --algo ppo --sampling-strategy none --env-type source --timesteps 500000
python train_sb3.py --algo sac --sampling-strategy none --env-type source --timesteps 500000

bash run_multi_seed.sh results/ppo_push_none_source_500k source
bash run_multi_seed.sh results/sac_push_none_source_500k source
```

**Task 5 — source/target transfer bounds.** Train SAC on the source and target environments for 1M steps, then evaluate the three transfer configurations (`source→source`, `source→target`, `target→target`):

```bash
python train_sb3.py --algo sac --sampling-strategy none --env-type source --timesteps 1000000
python train_sb3.py --algo sac --sampling-strategy none --env-type target --timesteps 1000000

bash run_multi_seed.sh results/sac_push_none_source_1000k source
bash run_multi_seed.sh results/sac_push_none_source_1000k target
bash run_multi_seed.sh results/sac_push_none_target_1000k target
```

**Task 6 — domain randomization.** Train UDR and ADR agents on the source environment for 1M steps, then evaluate both on the source and target environments:

```bash
python train_sb3.py --algo sac --sampling-strategy udr --env-type source --timesteps 1000000 --mass-range 0.5 6.0
python train_sb3.py --algo sac --sampling-strategy adr --env-type source --timesteps 1000000 --mass-range 0.5 6.0

bash run_multi_seed.sh results/sac_push_udr_source_1000k source
bash run_multi_seed.sh results/sac_push_udr_source_1000k target
bash run_multi_seed.sh results/sac_push_adr_source_1000k source
bash run_multi_seed.sh results/sac_push_adr_source_1000k target
```

### Learning curve

To create the Task 6 target learning curve from UDR and ADR checkpoints:

```bash
python learning_curve.py \
  --checkpoints-dir . \
  --prefix sac_push_udr_source,sac_push_adr_source \
  --env-type target \
  --episodes 50 \
  --baseline-model results/sac_push_none_source_1000k,results/sac_push_none_target_1000k \
  --out task6_target_learning_curve
```

The plot compares UDR and ADR checkpoint performance on the target environment and overlays the non-randomized source baseline and target-trained upper bound.

### Hardware

Training was performed on a CUDA GPU, specifically a T4 GPU on Lightning AI. The training script requires CUDA and exits if no GPU is available. Evaluation can be run on CPU.

### Results

Each row reports the mean over three evaluation seed offsets (`0`, `1000`, `2000`), 50 episodes each. The `source → target` setting is the sim-to-real transfer test: the model is trained on the `1.0 kg` source cube and evaluated on the `5.0 kg` target cube.

| Algo | Strategy | Steps | Eval environment | Success rate | Mean return |
|---|---|---|---|---:|---:|
| SAC | none | 1M | source | 100.0% | -0.36 |
| SAC | none | 1M | **target** (transfer) | **93.3%** | -0.72 |
| SAC | UDR | 1M | **target** (transfer) | **100.0%** | -0.56 |
| SAC | ADR | 1M | **target** (transfer) | **100.0%** | -0.43 |
| SAC | none | 1M | target (oracle, trained on target) | 100.0% | -0.42 |
| SAC | none | 500k | source | 54.7% | -2.35 |
| PPO | none | 500k | source | 19.3% | -3.54 |

### Takeaways

- **Domain randomization closes the transfer gap.** Trained only on the source cube, SAC reaches `93.3%` target success with a mean return of `-0.72`. Both UDR and ADR recover `100%` target success, and **ADR (`-0.43`)** essentially matches the oracle trained directly on the target (`-0.42`).
- **SAC works better than PPO in this setup.** At 500k steps on the source environment, SAC reaches `54.7%` success (`-2.35` return) versus PPO's `19.3%` (`-3.54`), so SAC was selected for the final transfer and domain randomization experiments.
- **More training helps.** SAC on the source environment improves from `54.7%` success at 500k steps to `100%` at 1M steps — the Push task needs enough training budget before the policy becomes reliable.

Eval logs for every configuration above are kept under [`part2/evals_report/`](part2/evals_report/).

## Project structure

```text
RL-Project-Group24/
├── README.md
├── requirements.txt
├── part1/
│   ├── part1.ipynb
│   ├── agent.py
│   ├── train.py
│   ├── visualize.py
│   ├── test_random_policy.py
│   └── videos/
└── part2/
    ├── train_sb3.py
    ├── eval_sb3.py
    ├── rand_wrapper.py
    ├── learning_curve.py
    ├── run_multi_seed.sh
    ├── test_random_policy.py
    ├── evals_report/
    └── panda-gym/
```

## Notes

- Evaluation logs are stored in `part2/evals_report/`.
- The modified `panda-gym` package under `part2/panda-gym/` must be installed before training or evaluation.
