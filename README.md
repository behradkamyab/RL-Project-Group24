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
- **`train.py`** — the training loop. `main(...)` exposes the hyperparameters (algorithm, baseline, seed, learning rates, hidden size, γ, entropy weight, …) and saves the trained model plus the return/timing history into `results/`.
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
    <th align="center" width="240">REINFORCE (vanilla)</th>
    <th align="center" width="240">REINFORCE + EMA baseline (&alpha;=0.05)</th>
    <th align="center" width="240">Refined Actor-Critic</th>
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
    <td align="center">Longest, smoothest gait; best mean, but on par with EMA.</td>
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

## Part 2 — PandaGym Push

Part 2 trains a Franka-Emika Panda arm to push a cube to a goal (`PandaPush-v3`, dense reward) with stable-baselines3, and studies **sim-to-real transfer** between two versions of the task:

- **source** — the cube weighs **1.0 kg** (the "simulation" we train in),
- **target** — the cube weighs **5.0 kg** (the "real" environment we want to transfer to).

A policy trained only on the light source cube tends to over-push the heavier target cube, so we apply **domain randomization** over the cube mass during training and measure how well each strategy closes the source→target gap.

**What we built**

- **`train_sb3.py`** — trains SAC or PPO on `PandaPush-v3` (`--algo`, `--env-type source|target`, `--timesteps`, `--seed`, `--net-arch`, `--learning-rate`, `--resume`). The randomization strategy is selected with `--sampling-strategy none|udr|adr` and bounded by `--mass-range MIN MAX` (default `0.5 4.5`). SAC uses gSDE exploration and a replay buffer; both default to `512,512` networks. Checkpoints land in `models/checkpoints/`, the final model in `results/`, and a heartbeat log (for long cloud runs) in `training.log`.
- **`rand_wrapper.py`** — `RandomizationWrapper`, a `gym.Wrapper` that resamples the cube mass on every `reset` and applies it via PyBullet `changeDynamics`:
  - `udr` — samples the mass uniformly from the full `mass-range` each episode.
  - `adr` — starts from a narrow range around the source mass and **automatically widens it** as the policy succeeds: it periodically pushes a boundary outward when the success rate at that boundary clears `adr_target`, and pulls it back in when performance drops below `adr_perf_low` (single-sided by default, since the gap is one-directional toward the heavier target).
- **`eval_sb3.py`** — evaluates a saved model over N episodes (default 50) on a chosen `--env-type`, reporting mean/std/variance of return and the success rate. `--seed-offset` shifts the per-episode seeds for multi-seed evaluation.
- **`run_multi_seed.sh`** — runs `eval_sb3.py` at three seed offsets (0/1000/2000) and tees the output into `evals_report/<model>.<env>.evallog.txt`.
- **`learning_curve.py`** — reconstructs return/success curves over training by evaluating every checkpoint in a prefix, writing a `.csv` + `.png`; `--baseline-model` overlays flat reference lines for final models without a checkpoint series.
- **`test_random_policy.py`** — random-policy rollout on `PandaPush-v3` to get familiar with the task.

### Setup

The `part2/panda-gym/` package is **provided by the course professor** — it is a modified panda-gym in which `PandaPush-v3` accepts a `type="source"|"target"` argument that sets the cube mass (1.0 kg vs 5.0 kg). We use it as given; our work is the training, randomization, and evaluation code in `part2/`. Install it in editable mode, then the project requirements:

```bash
cd part2/panda-gym
pip install -e .
cd ..
pip install -r ../requirements.txt
```

### How to run

Train SAC on the source environment with each strategy (1M steps, the configuration behind our reported results):

```bash
#Task 4 
python train_sb3.py --algo ppo --sampling-strategy none --env-type source --timesteps 500000
python train_sb3.py --algo sac --sampling-strategy none --env-type source --timesteps 500000

#Task 5
python train_sb3.py --algo sac --sampling-strategy none --env-type source --timesteps 1000000
python train_sb3.py --algo sac --sampling-strategy none --env-type target --timesteps 1000000

#Task 6
python train_sb3.py --algo sac --sampling-strategy udr --env-type source --timesteps 1000000 --mass-range 0.5 6.0
python train_sb3.py --algo sac --sampling-strategy adr --env-type source --timesteps 1000000 --mass-range 0.5 6.0
```

Then evaluate, e.g. transfer to the heavy target cube across three seed offsets:

```bash
# Task 4 
bash run_multi_seed.sh results/ppo_push_none_source_500k source
bash run_multi_seed.sh results/sac_push_none_source_500k source

# Task 5
bash run_multi_seed.sh results/sac_push_none_source_1000k source 
bash run_multi_seed.sh results/sac_push_none_source_1000k target 
bash run_multi_seed.sh results/sac_push_none_target_1000k target

# Task 6
bash run_multi_seed.sh results/sac_push_udr_source_1000k source
bash run_multi_seed.sh results/sac_push_udr_source_1000k target
bash run_multi_seed.sh results/sac_push_adr_source_1000k source
bash run_multi_seed.sh results/sac_push_adr_source_1000k target
```

> `train_sb3.py` requires a CUDA GPU (it exits otherwise); training was done on a T4 GPU(Lightning ai). Evaluation runs on CPU.

### Results

Each row is the mean over three evaluation seed offsets (0/1000/2000), 50 episodes each. **Source→target is the sim-to-real test**: train on the 1.0 kg cube, evaluate on the 5.0 kg cube.

| Algo | Strategy | Steps | Eval env | Success rate | Mean return |
| --- | --- | --- | --- | --- | --- |
| SAC | none | 1M | source | 100% | -0.37 |
| SAC | none | 1M | **target** (transfer) | **~93%** | -0.72 |
| SAC | udr | 1M | **target** (transfer) | **100%** | -0.55 |
| SAC | adr | 1M | **target** (transfer) | **100%** | -0.43 |
| SAC | none | 1M | target (oracle, trained on target) | 100% | -0.42 |
| SAC | none | 500k | source | 56% | -2.37 |
| PPO | none | 500k | source | ~19% | -3.54 |

Takeaways:

- **Domain randomization closes the sim-to-real gap.** Trained only on the source cube, SAC drops to ~93% success with higher-variance returns when faced with the heavier target cube. Both UDR and ADR recover **100%** target success, and **ADR (-0.43)** essentially matches the oracle that was trained directly on the target (-0.42).
- **SAC ≫ PPO here.** With our budget SAC solves the task (100% on source at 1M steps), while PPO struggled to push the cube reliably (~19%), so SAC is the algorithm for the randomization study.
- **More training helps.** SAC on source rises from 56% (500k) to 100% (1M) success.

Eval logs for every configuration above are kept under [`part2/evals_report/`](part2/evals_report/).

## Project structure

```
RL-Project-Group24/
├── README.md
├── requirements.txt
├── part1/                      
│   ├── part1.ipynb             
│   ├── agent.py                
│   ├── train.py                
│   ├── visualize.py            
│   ├── test_random_policy.py   
│   ├── results/                
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
