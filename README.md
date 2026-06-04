# Atari Recreations Project

This repository contains code for recreating Atari environments using various reinforcement learning techniques. The project includes a training script, model checkpoints, and validation scripts to evaluate the trained models.

## Table of Contents
1. [Project Structure](#project-structure)
2. [Dependencies](#dependencies)
3. [Training](#training)
4. [Evaluation](#evaluation)
5. [Usage](#usage)

## Project Structure

The project is structured as follows:

- `agent/`: Contains the agent implementation.
  - `networks.py`: Neural network models used by the agent.
  - `agent.py`: Main agent class and training logic.
- `environment/`: Environment wrappers for Atari games.
  - `atari_envs.py`: Custom environments for Atari games.
- `config/`: Configuration files for hyperparameters and other settings.
- `runs/pong_dqn/`: Directory containing trained models for Pong using DQN.
- `checkpoints/`: Checkpoint files saved during training.
- `tensorboard_logs/`: TensorBoard logs for monitoring training progress.
- `.gitignore`: Specifies files to be ignored by Git.

## Dependencies

The project requires the following Python packages:

- `numpy`
- `torch`
- `gymnasium`

You can install these dependencies using pip:

```sh
pip install numpy torch gymnasium
```

## Training

To train a model, run the training script:

```sh
python train.py
```

This will save trained models in the `runs/pong_dqn/` directory.

## Evaluation

To evaluate a trained model, use the evaluation script:

```sh
python eval.py --model runs/pong_dqn/model.pth
```

## Usage

1. Clone this repository.
2. Install dependencies using pip.
3. Train a model by running `train.py`.
4. Evaluate the model using `eval.py`.