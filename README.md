# Atari Pong — Deep Q-Network (DQN) Recreation

A faithful PyTorch reproduction of the Nature DQN paper by DeepMind, training an AI agent to play Atari Pong from raw pixels using deep reinforcement learning.

---

## Paper Reference

> Mnih, V., Kavukcuoglu, K., Silver, D., et al. (2015).  
> **Human-level control through deep reinforcement learning.**  
> *Nature*, 518, 529–533.  
> https://doi.org/10.1038/nature14236

This paper was a landmark in AI research — it demonstrated that a single neural network architecture, trained end-to-end from raw pixel input using reinforcement learning, could achieve human-level performance across 49 different Atari games without any game-specific engineering. It introduced the key innovations of experience replay and a separate target network that stabilise Q-learning with deep neural networks.

---

## Project Overview

This project reproduces the core DQN pipeline from the Nature paper and applies it to Atari Pong (`ALE/Pong-v5`). The agent observes raw game frames, learns to associate visual states with expected future rewards, and improves its play policy through trial and error — no hand-crafted rules, no game knowledge baked in.

**Goals:**
- Faithfully implement the Nature DQN architecture in PyTorch
- Train an agent to achieve positive reward on Pong from scratch
- Provide a clean, well-documented codebase for learning and experimentation

---

## Architecture

### NatureDQN (Primary)

The network takes a stack of 4 preprocessed grayscale frames as input and outputs Q-values for each possible action.

```
Input:  (batch, 4, 84, 84)  — 4 stacked greyscale frames

Conv1:  32 filters, 8×8 kernel, stride 4  →  (batch, 32, 20, 20)  + ReLU
Conv2:  64 filters, 4×4 kernel, stride 2  →  (batch, 64,  9,  9)  + ReLU
Conv3:  64 filters, 3×3 kernel, stride 1  →  (batch, 64,  7,  7)  + ReLU

Flatten:                                   →  (batch, 3136)

FC1:    512 units                          →  (batch, 512)          + ReLU
FC2:    n_actions units (linear)           →  (batch, n_actions)

Output: Q-value estimate for each action
```

- **Weight initialisation:** Orthogonal init (gain √2) for conv layers, Kaiming uniform for linear layers
- **Normalisation:** uint8 pixel inputs are divided by 255.0 on the forward pass
- **VRAM footprint:** ~6.5 MB for network weights — well within the 8GB VRAM budget

### DuelingDQN (Optional Upgrade)

An optional drop-in variant implementing the Dueling Network Architecture (Wang et al., 2016). The FC head is replaced with two parallel streams:

- **Value stream** V(s) → scalar
- **Advantage stream** A(s, a) → (n_actions,)

Combined as: `Q(s,a) = V(s) + A(s,a) − mean_a[A(s,a)]`

All convolutional layers are identical to NatureDQN. Swap `architecture="dueling"` in the factory function to use it.

---

## How It Works

### DQN Training Loop

Each step of training follows this sequence:

1. **Observe** — receive the current stacked frame observation from the environment
2. **Act** — select an action using epsilon-greedy policy (random with probability ε, greedy otherwise)
3. **Store** — save the transition `(obs, action, reward, next_obs, done)` in the replay buffer
4. **Sample** — draw a random mini-batch of 32 transitions from the replay buffer
5. **Compute TD target** — `r + γ · max_a Q_target(s', a)` using the frozen target network
6. **Backprop** — compute Huber loss between predicted Q-values and TD targets, update online network

### Replay Buffer

- Stores the last 10,000 transitions (configurable, paper default is 1,000,000)
- Observations stored as `uint8` to minimise RAM usage (4× smaller than float32)
- Random sampling breaks temporal correlation between consecutive transitions — critical for stable training

### Epsilon-Greedy Exploration

- Starts at ε = 1.0 (fully random)
- Decays linearly to ε = 0.05 over 100,000 steps
- Ensures the agent explores widely early in training before exploiting learned knowledge

### Target Network

- A second copy of the online network with frozen weights
- Used exclusively to compute TD targets — prevents the moving-target instability of bootstrapping against the same network being updated
- Hard update: every 1,000 steps, online network weights are copied directly to the target network

---

## Environment Preprocessing Pipeline

Raw Atari frames go through four wrappers in sequence before reaching the agent:

### 1. `NoopResetWrapper`
On each episode reset, takes 1–30 random no-op actions before returning the first observation. This randomises the starting state and prevents the agent from memorising a fixed opening sequence.

### 2. `MaxAndSkipWrapper`
Repeats each selected action for 4 consecutive frames and returns the pixel-wise maximum of the last 2 frames. Frame skipping speeds up training (the agent makes 4× fewer decisions) and max-pooling removes the sprite flickering that is common in Atari games.

### 3. `WarpFrameWrapper`
Converts the RGB frame to grayscale using `cv2.COLOR_RGB2GRAY` and resizes it to 84×84 pixels using area interpolation. Reduces input dimensionality from `(210, 160, 3)` to `(84, 84, 1)`.

### 4. `FrameStackWrapper`
Stacks the last 4 processed frames along the channel axis, producing a `(4, 84, 84)` observation. This gives the agent temporal information — it can infer velocity and direction of moving objects from the frame stack.

**Final observation shape entering the network:** `(4, 84, 84)`

---

## Project Structure

```
atari-recreation/
│
├── agent/
│   ├── networks.py          # NatureDQN and DuelingDQN architectures + build_network factory
│   └── agent.py             # ReplayBuffer, EpsilonSchedule, DQNAgent
│
├── environment/
│   ├── wrappers.py          # NoopReset, MaxAndSkip, WarpFrame, FrameStack wrappers + make_env
│   └── environment.py       # AtariEnvironment class + make_atari_env factory
│
├── train.py                 # Training loop with TensorBoard logging, checkpointing, and resume support
├── requirements.txt         # Python dependencies
├── ACKNOWLEDGEMENTS.md      # Credits and references
├── .gitignore               # Ignored files (venv, checkpoints, ROM cache)
└── .venv/                   # Python virtual environment (not committed)
```

---

## Hardware Requirements

| Component | Minimum | Tested On |
|---|---|---|
| GPU | NVIDIA 8GB VRAM | RTX 3070 Ti 8GB |
| RAM | 16GB | 16GB DDR4 |
| OS | Windows 10/11 or Linux | Windows 11 |
| Python | 3.10+ | 3.14.5 |
| CUDA | 11.8+ | 12.8 |

---

## Installation

### 1. Clone the repository

```bat
git clone https://github.com/yourusername/atari-recreation.git
cd atari-recreation
```

### 2. Create a virtual environment

**Command Prompt (Windows):**
```bat
python -m venv .venv
```

### 3. Install PyTorch with CUDA support

```bat
".\.venv\Scripts\python.exe" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

### 4. Install remaining dependencies

```bat
".\.venv\Scripts\python.exe" -m pip install gymnasium[atari] autorom[accept-rom-license] opencv-python-headless tensorboard
```

### 5. Verify installation

```bat
".\.venv\Scripts\python.exe" -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
".\.venv\Scripts\python.exe" -c "import ale_py; import gymnasium as gym; gym.register_envs(ale_py); env = gym.make('ALE/Pong-v5'); print('Pong OK, actions:', env.action_space.n); env.close()"
```

> **Note on path quoting:** Use double quotes around paths in Command Prompt. Use single quotes in PowerShell. The examples above use Command Prompt syntax.

---

## Training

### Start a training run with default settings

```bat
".\.venv\Scripts\python.exe" train.py
```

### Available arguments

| Argument | Default | Description |
|---|---|---|
| `--total_steps` | 500000 | Total environment steps to train for |
| `--lr` | 1e-4 | Adam optimizer learning rate |
| `--seed` | 42 | Random seed for reproducibility |
| `--log_dir` | `runs/pong_dqn` | TensorBoard log directory |
| `--checkpoint_dir` | `checkpoints` | Directory to save model checkpoints |
| `--resume` | None | Path to a `.pth` checkpoint file to resume training from |

### Resume from a checkpoint

```bat
".\.venv\Scripts\python.exe" train.py --resume checkpoints/dqn_episode_400.pth --total_steps 1000000 --log_dir runs/pong_dqn_run2 --checkpoint_dir checkpoints/run2
```

### Monitor training with TensorBoard

Open a **separate** Command Prompt window and run:

```bat
".\.venv\Scripts\tensorboard.exe" --logdir runs
```

Then open `http://localhost:6006` in your browser. You will see three live charts:

- **Reward/episode** — the most important metric; watch this climb over time
- **Loss/episode** — average Huber loss per episode; should trend downward
- **Epsilon/episode** — exploration rate decaying from 1.0 to 0.05

> **Tip:** TensorBoard automatically detects all subdirectories under `runs/` and plots them together with different colours, making it easy to compare runs side by side.

### Checkpoints

Model weights are saved automatically every 100 episodes to the `--checkpoint_dir` directory as `dqn_episode_N.pth`. Use separate `--checkpoint_dir` and `--log_dir` values for each run to avoid overwriting previous results.

---

## Expected Training Progress

| Steps | Expected Reward | What's Happening |
|---|---|---|
| 0 – 50k | ~−21 | Agent acting near-randomly, ε still high |
| 50k – 150k | −18 to −10 | Agent starts blocking shots, ε decaying |
| 150k – 300k | −10 to 0 | Agent winning occasional points |
| 300k – 500k | 0 to +10 | Agent consistently competitive |

> Pong has a reward range of −21 (worst) to +21 (perfect). Reaching positive reward means the agent is winning more points than it concedes.

---

## Results

| Metric | Value |
|---|---|
| Total steps trained | ~700,000 (500k run 1 + resumed run 2) |
| Best observed episode reward | −7 |
| Final episode reward (run 1) | −7 at episode 860 |
| Epsilon at end of run 1 | 0.05 (fully decayed) |
| Training time | ~5 hours total |
| Hardware | RTX 3070 Ti, 8GB VRAM, Windows 11 |

**Key observations:**
- Agent reward improved from −21 (random play) to −7 over ~500k steps
- Epsilon fully decayed to 0.05 by step ~100,000 — agent operating near-greedily for the majority of training
- Loss trended consistently downward throughout training, indicating stable learning
- Reward improvement began around episode 400–500, consistent with replay buffer filling and epsilon decay completing

---

## Acknowledgements

See [ACKNOWLEDGEMENTS.md](ACKNOWLEDGEMENTS.md).