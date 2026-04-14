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

### Core Components

* `sequence_model.py`
  Game environment and state transition logic

* `agents/...`
  AI agent implementations (search, RL, hybrid strategies)

* `general_game_runner.py`
  Simulation engine for running matches and experiments

* `sequence_utils.py`
  Shared constants and helper utilities

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
