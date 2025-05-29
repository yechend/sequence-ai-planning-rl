# Agents Folder Overview

This directory contains different agent implementations and supporting modules for playing and training strategies in the Sequence game environment.

---
## Folder Structure
```plaintext
agents/
│
├── generic/
│   ├── blockerAgent.py         # Agent that blocks opponent strategies
│   ├── first_move.py           # Agent that handles initial moves heuristically
│   ├── random.py               # Agent that makes random valid moves
│   └── timeout.py              # Agent that times out or does nothing
│
├── t_068/
│   ├── 26win heuristic.py      # Heuristic-based agent optimised for 26-win scenarios
│   ├── Qlearning1.py           # Q-Learning based agent implementation
│   ├── curriculum_trainer.py   # Curriculum training for policy learning
│   ├── myTeam.py               # Main entry point for the final agent
│   ├── preFinal.py             # Preliminary version of Two-Step Greedy
│   ├── sequence_state.py       # State representation for the Sequence game
│   ├── train_sequence_policy.py# Training against random for policy learning
│   ├── weights1.json           # Pre-trained weights or Q-values
│   └── __pycache__/            # Auto-generated cache for faster imports
│
└── AgentsREADME.md             # Documentation for individual agents
```
---

## Agent Categories

### Baseline Agents (`generic/`)
Provide simple behaviours for benchmarking:
- `random.py`: Random legal moves
- `blockerAgent.py`: Defensive strategy blocking the opponent
- `first_move.py`: Specialised logic for the game's first move
- `timeout.py`: Dummy agent with timeout behaviour

### Custom Agents (`t_068/`)
Advanced or learning-based agents developed for competition:
- `26win heuristic.py`: AI Method 3 - Heuristic agent optimised for 26-win strategies
- `Qlearning1.py`: AI Method 3 - Reinforcement Learning agent using Q-Learning
- `preFinal.py`: AI Method 1 - Preliminary heuristic model
- `myTeam.py`: AI Method 1 - **Final Agent Two-Step Greedy Best-First Search**

### Training and Offline Policy Utilities
- `sequence_state.py`: AI Method 1 - Encodes the game state for learning and inference
- `curriculum_trainer.py`: AI Method 1 - Trainer for curriculum-based offline learning policy
- `train_sequence_policy.py`: AI Method 1 - Script to train a simple offline learning policy
- `weights1.json`: AI Method 3 - Stores learned weights or Q-values for inference

---

## Entry Point

To use the main team agent:

```python
from agents.t_068.myTeam import myAgent
