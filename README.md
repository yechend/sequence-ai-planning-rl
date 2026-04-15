# Sequence AI Agent – Reinforcement Learning & Planning

## Overview

<p align="center"> <img src="img/sequence.png" alt="Picture of Sequence board" width="400"> </p>

This project develops a **high-performance AI agent** for the board game *Sequence*, combining reinforcement learning, heuristic search, and strategic planning to operate in a stochastic, adversarial environment.

The agent is designed to:

* Make optimal decisions under uncertainty
* Model and respond to opponent behaviour
* Balance short-term rewards with long-term strategic positioning

**Result:** Achieved **78%+ win rate** and ranked **Top 7 / 94 agents** in a competitive evaluation setting.

---

## Key Features

* **Hybrid AI Architecture**

  * Heuristic-based search for fast decision-making
  * Reinforcement learning for policy optimisation
  * Opponent-aware strategy for adversarial play

* **Efficient Decision Engine**

  * Real-time action selection under strict time constraints
  * Optimised state evaluation and pruning

* **Robust Evaluation Framework**

  * Controlled experiments across multiple agent configurations
  * Reproducible benchmarking and performance comparison

---

## System Architecture
## Project Structure

```text
sequence-ai-planning-rl/
├── agents/                    # AI agent implementations (heuristic, RL, MCTS, hybrid)
│   ├── ai_agent/              # Custom high-performing agents developed in this project
│   └── generic/               # Baseline/reference agents for benchmarking
│
├── docker/                    # Docker environment for reproducible execution
│   ├── Dockerfile             # Container definition
│   └── *.sh                   # Scripts for setup, execution, and cleanup
│
├── img/                       # Images used in documentation and README
│   └── sequence.png           # Game board visualisation
│
├── Sequence/                  # Core game environment implementation
│   ├── sequence_model.py      # Game logic and state transitions
│   ├── sequence_utils.py      # Helper functions and constants
│   ├── sequence_displayer.py  # Visualisation and rendering
│   └── sequence.md            # Game rules and environment documentation
│
├── game.py                    # Entry point for game setup and configuration
├── general_game_runner.py     # Main simulation engine for running matches
├── template.py                # Base template for implementing new agents
├── utils.py                   # Shared utilities across the project
│
├── requirements.txt           # Python dependencies
├── README.md                  # Project overview and usage instructions
└── LICENSE                    # License information
```
### Core Components
- **Game Environment (`sequence_model.py`)**  
  Handles state transitions, rule enforcement, and game logic

- **Decision-Making Agents (`agents/`)**  
  Implements multiple AI strategies including heuristic search, MCTS, and RL-based approaches

- **Simulation Engine (`general_game_runner.py`)**  
  Runs matches, evaluates agents, and manages experiments

- **Evaluation Utilities (`sequence_utils.py`)**  
  Provides shared scoring logic and helper functions

---

## Approach

The agent combines multiple AI paradigms:

### 1. Heuristic Search

* Evaluates board states using domain-specific scoring
* Enables fast, interpretable decision-making

### 2. Reinforcement Learning

* Learns improved policies through iterative evaluation
* Captures long-term reward dynamics

### 3. Opponent Modelling

* Anticipates opponent strategies
* Adapts actions to competitive dynamics

This hybrid approach allows the agent to outperform single-method baselines.

---

## Performance

* **Win Rate:** 78%+
* **Ranking:** Top 7 / 94 agents
* Demonstrated strong generalisation across diverse opponents

Key insight:

> Combining search with reinforcement learning significantly improves both stability and performance compared to standalone methods.

---

## Getting Started

### Install Dependencies

Ensure Python 3.8+ is installed.

```bash
pip install func-timeout pytz GitPython
```

Optional:

```bash
python -m venv venv
source venv/bin/activate  # macOS / Linux
pip install func-timeout pytz GitPython
```

---

### Run the Environment

```bash
python general_game_runner.py -g Sequence
```

Run with custom agents:

```bash
python general_game_runner.py -g Sequence -a agents.generic.random,agents.generic.first_move
```

---

## Experimentation

The framework supports flexible evaluation:

```bash
python general_game_runner.py -h
```

Capabilities include:

* Running multiple simulations
* Logging and replaying games
* Benchmarking different agent strategies

---

## Design Considerations

* Real-time decision constraints (≤ 1s per move)
* Trade-off between exploration and exploitation
* Scalability of evaluation across multiple strategies

---

## Future Improvements

* Deep reinforcement learning (DQN / policy gradient)
* Monte Carlo Tree Search integration
* Enhanced opponent modelling using probabilistic methods

---

## License

This project is licensed under the MIT License.
