# Sequence Game Environment

## Overview

This repository provides a configurable environment for developing and evaluating AI agents in the game **Sequence**, a strategic, turn-based board game with elements of planning, stochasticity, and adversarial decision-making.

The environment supports:

* Simulation of full game states and transitions
* Evaluation of agent strategies under competitive settings
* Integration of custom AI agents for experimentation and benchmarking

---

## Environment Structure

The core environment handles game logic, state transitions, and action validation. Custom agents interact with the environment via a defined interface.

### Key Components

* `sequence_model.py`
  Implements the core game logic, including state representation and legal action generation.
  The `getLegalActions()` function defines the available actions at each turn.

* `agents/...`
  Contains agent implementations. Each agent must define:

  * `__init__(self, _id)`
  * `SelectAction(self, actions, rootstate)`

* `general_game_runner.py`
  Entry point for running simulations, benchmarking agents, and configuring experiments.

* `sequence_utils.py`
  Defines constants and helper functions used across the environment.

---

## Game Mechanics

Sequence is a two-player, partially stochastic board game involving:

* Strategic placement of pieces on a fixed grid
* Card-driven action selection
* Formation of sequences to score points

For full gameplay rules:
https://en.wikipedia.org/wiki/Sequence_(game)

Playable reference:
https://play-sequence.com/

---

## Execution Model

* Turn-based gameplay between two agents
* Each agent selects an action based on the current game state
* The environment updates the state and proceeds to the next turn

### Time Constraints

* Each move must be returned within **1 second**
* A short initialization window is available at the start for setup (e.g., loading models or policies)

---

## Running the Environment

### Install Dependencies

Ensure Python 3.8+ is installed.

```bash
pip install func-timeout pytz GitPython
```

Optional (recommended):

```bash
python -m venv venv
source venv/bin/activate  # macOS / Linux
pip install func-timeout pytz GitPython
```

---

### Quick Start

Run a default game:

```bash
python general_game_runner.py -g Sequence
```

Run with custom agents:

```bash
python general_game_runner.py -g Sequence -a agents.generic.random,agents.generic.first_move
```

---

## Experimentation & Debugging

The runner supports flexible configurations:

```bash
python general_game_runner.py -h
```

Common options:

* `-t` → text mode (no GUI)
* `-p` → print logs to terminal
* `-s` → save game replay
* `-l` → save execution logs

This enables:

* Rapid debugging
* Batch simulations
* Performance evaluation across agents

---

## Design Considerations

When developing agents:

* Ensure actions are returned within time constraints
* Avoid unnecessary computation outside decision steps
* Maintain reproducible and stable evaluation

The environment supports a wide range of approaches:

* Heuristic and search-based planning
* Reinforcement learning
* Hybrid decision-making systems

---

## Notes

* Agent outputs (logs/errors) may be visible during evaluation
* Code should be robust and avoid runtime exceptions
* The environment is extensible for experimentation with different AI techniques
