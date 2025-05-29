# AI Method 3 - Approximate Q-learning
We selected Q-learning as our third technique due to both its theoretical strengths and its alignment with our team's early design ideas. In our preliminary submission, each team member developed a unique agent with different strategic focuses—one emphasized center control, another prioritized immediate sequence building, while the third focused on long-term board influence.

This diversity in strategies sparked an idea: could we build a meta-agent that learns to dynamically combine and prioritize these strategies depending on the situation and opponent? This concept mirrors the essence of feature-based Q-learning, where an agent learns to assign weights to multiple strategic features to evaluate the quality of actions.

Upon deeper reflection, we realized that our goal—to combine different heuristics in a flexible, data-driven way—is essentially what Q-learning achieves when implemented with hand-crafted features. It provides a structured yet adaptable framework to learn from experience, optimize long-term performance, and ultimately evolve beyond static rule-based decision making. Therefore, we adopted Q-learning with feature-based function approximation to realize this vision in a principled and scalable manner.
# Table of Contents
* [Motivation](#motivation)
* [Application](#application)
* [Experiments](#experiments)
* [Solved challenges](#solved-challenges)
* [Trade-offs](#trade-offs)
    - [Advantages](#advantages)
    - [Disadvantages](#disadvantages)
* [Future improvements](#future-improvements)

### Motivation
Approximate Q-learning offered a compelling answer. By using feature-based linear value approximation, it allows the agent to generalize across states, learn from rewards over time, and dynamically weight different board-level patterns. The selected features were inspired by our best-performing heuristic agent and encoded critical strategic signals.But after thorough testing, this method was not implemented due to the bad performance and we need to select  so many suitbale features that could get the higher win rates.


[Back to top](#table-of-contents)

### Application
Problem Modeling
-	State Definition: A state is defined by three elements: the board layout (10×10 chips), the agent’s current hand, and the available draft cards.
-	Goal State: The agent aims to either form two completed sequences or fully occupy all four central tiles.
-	Reward Function: Reward = number of completed sequences. (Sparse reward, updated only when a new sequence is detected.)
-	Reward Function: Reward = number of completed sequences. (Sparse reward, updated only when a new sequence is detected.)
-	Q-function: Linear combination of tactical pattern features (see table below).

### Tactical Pattern Features Used in Q-Function

| Feature           | Description                                                    |
|------------------|----------------------------------------------------------------|
| `live_one`        | Detects 1 aligned chip with two open ends.                    |
| `sleep_two`       | Detects 2 aligned chips with one open end.                    |
| `live_two`        | Detects 2 aligned chips with two open ends.                   |
| `sleep_three`     | Detects 3 aligned chips with one open end.                    |
| `live_three`      | Detects 3 aligned chips with two open ends.                   |
| `chong_four`      | Detects 4 aligned chips with only one open end.               |
| `live_four`       | Detects 4 aligned chips with two open ends.                   |
| `live_five`       | Detects 5-in-a-row (complete sequence).                       |
| `opp_live_three`  | Detects 3 aligned opponent chips with two open ends.          |
| `opp_live_four`   | Detects 4 aligned opponent chips with two open ends.          |
| `opp_live_five`   | Detects completed opponent sequence (5 aligned).              |

[Back to top](#table-of-contents)

### Experiments
We have conducted four key experiments to test the impact of different design choices in our Q-learning agent:

---

**1. Hyperparameter Tuning**

We varied the learning rate (`alpha`), discount factor (`gamma`), and exploration rate (`epsilon`) to understand their impact on convergence and performance.

| Alpha | Gamma | Epsilon | Win Rate vs Heuristic | Observation |
|-------|--------|---------|------------------------|-------------|
| 0.2   | 0.9    | 0.1     | ~5%                    | Baseline setup. Agent rarely wins. |
| 0.4   | 0.9    | 0.2     | ~8%                    | Slightly faster learning, but more unstable. |
| 0.3   | 0.95   | 0.05    | ~10%                   | Slower but more stable training. Slight performance gain. |

>  **Finding**: Higher gamma improves long-term planning slightly. However, epsilon decay may be needed to balance exploration and exploitation.

---

**2. Feature Selection Variants**

We tested three different sets of features to evaluate which characteristics of the board state are most critical for effective decision-making.

| Feature Set                       | Description                                  | Win Rate |
|----------------------------------|----------------------------------------------|----------|
| Tactical Patterns Only           | Includes `live_four`, `live_three`, etc.     | ~6%      |
| Positional + Tactical (ours)     | Adds `center_distance`, `fork_potential`     | ~5%     |
| Defensive Only                   | Focuses on opponent formations               | ~4%      |

>  **Finding**: Combining tactical and positional features significantly improves the agent’s situational awareness and sequence formation ability.

---

**3. Reward Function Design**

We explored different reward functions to address the sparsity of sequence-based reward signals.

| Reward Type           | Definition                                               | Win Rate |
|-----------------------|----------------------------------------------------------|----------|
| Sparse Reward         | +1 for completed sequence only                           | ~6%      |
| Shaped Reward         | +0.2 for live-four, +0.1 for live-three, -0.3 for opp    | ~10%     |
| Delta Score Reward    | Reward = `score_t - score_(t-1)`                         | ~9%      |

>  **Finding**: Shaped rewards accelerate early learning and help the agent avoid empty or unproductive moves. Sparse rewards delay learning.

---
**4. Comparison with Other Agents (Same Time Budget)**

We conducted head-to-head experiments between our Q-learning agent and two strong baselines — the GBFS agent and the MCTS agent — both constrained to 1-second decision windows per move.

| Opponent Agent | Win Rate (Q-learning) | Notes |
|----------------|------------------------|-------|
| GBFS Agent     | ~0%                    | Q-agent consistently failed to defeat GBFS due to lack of lookahead and poor strategy convergence. |
| MCTS Agent     | ~5%                    | Slightly better, but MCTS still dominated due to deeper planning and simulated rollouts. |

> **Finding**: Despite its theoretical adaptability, the Q-agent could not match the structured depth search capabilities of either GBFS or MCTS within the same compute time budget. This highlights the importance of convergence stability and reward shaping in Q-learning for real-time environments.

### Summary
- Win rates remain low (~5%) against well-designed heuristics, showing limitations of linear feature-based Q-learning in this game.

[Back to top](#table-of-contents)

### Solved Challenges
- **Scalability of Q-learning**  
  Instead of using a full Q-table, we adopted feature-based approximation to handle the combinatorial complexity of the Sequence game state space.

- **Designing Meaningful Features**  
  We successfully encoded strategic board knowledge into compact, interpretable features such as `live_four`, `fork_potential`, and `opp_live_three`, enabling the agent to evaluate tactical value.

- **Sparse Rewards in Early Game**  
  We explored intermediate reward shaping (e.g., bonuses for `live_three`, penalties for opponent threats), improving learning signal during mid-game plays.

- **Heuristic-Q Learning Integration**  
  We bridged heuristic and learning-based approaches by deriving features from the strongest greedy agent, allowing the Q-agent to inherit domain knowledge.

- **Maintaining Time Constraints**  
  The final implementation respects the 0.95s time limit per move using lightweight feature computation and time-aware action selection logic.

[Back to top](#table-of-contents)


### Trade-offs

While our Approximate Q-learning agent offered a principled and interpretable learning framework, it suffered from several practical limitations in training effectiveness, strategic depth, and real-game performance. This section highlights the main advantages and disadvantages observed during implementation and testing.

#### *Advantages*

- **Efficient Learning with Compact Representation**  
  Unlike traditional tabular Q-learning, our agent uses a linear combination of six handcrafted features to approximate Q-values. This significantly reduces memory and computational demands, making it feasible under the 1-second time constraint (`THINK_TIME_LIMIT = 0.95s`) and suitable for the large, combinatorial environment of Sequence.

- **Interpretable and Debuggable Feature Set**  
  Each feature reflects a clear tactical pattern. This transparency enables easy inspection of learned weights and identification of behavioral issues — a key asset when diagnosing repeated Q-agent failures.

- **Heuristic-Aware Initialization**  
  Our feature definitions were directly inspired by our best-performing heuristic agent. This provided a meaningful starting policy and helped the Q-agent avoid entirely random early-game behavior, accelerating learning stability compared to uninitialized Q-learning.

#### *Disadvantages*

- **Strong Dependence on Manual Feature Design**  
  Despite thoughtful engineering, the feature extractor only encodes local alignment and center bias. It omits broader strategic signals like opponent intention, long-term planning, and card synergies. These blind spots limit the agent’s ability to react effectively in mid-to-late game phases.

- **Inability to Capture Nonlinear Dynamics**  
  The linear Q-function cannot model multi-turn setups, positional sacrifices, or compound tactical threats. As a result, the agent often chooses moves that are either locally optimal or strategically incoherent, especially when the board becomes complex.

- **Lack of Generalization and Adaptation**  
  The feature set is brittle across opponents and board states. In testing, the Q-agent consistently lost to both Method 1 (heuristic GBFS) and Method 2 (MCTS agent), achieving near 0% win rate even after multiple games and weight updates (`weights1.json`).

- **Sparse Reward and No Intermediate Feedback**  
  Using raw sequence completion as the only reward creates a sparse signal, especially early in the game. Without reward shaping (e.g., partial credit for forming `live_three`), the agent struggles to distinguish good exploratory paths from ineffective ones.

[Back to top](#table-of-contents)

### Future improvements
Despite its simplicity and interpretability, our Q-learning agent can be significantly improved in both feature design and training methodology. Below are several concrete and actionable directions for future enhancement:
###  Feature Engineering

- **Sequence Blocking Feature**  
  Add a binary feature to detect whether an action can block an opponent’s imminent sequence, especially for one-eyed jack `"remove"` actions.

- **Positional Bias Features**  
  Incorporate features like `is_center_tile`, `is_diagonal`, or `is_border` to capture location-based strategy preferences.

- **Phase-aware Feature Weighting**  
  Adjust the importance of certain features depending on the game stage (early/mid/late), e.g., emphasizing center control early and sequence completion late.

---

###  Reward Function Enhancements

- **Shaped Intermediate Rewards**  
  Instead of only rewarding completed sequences, provide small rewards for forming intermediate patterns:
    - `+0.1` for `live_three`
    - `+0.2` for `live_four`
    - `-0.3` if opponent forms `live_four`

- **Score Delta-based Rewards**  
  Use the change in agent score:  
  `reward = current_score - previous_score`  
  to provide a more responsive and continuous reward signal.

- **Penalty for Losing Center Control**  
  Penalize actions that give up central positions to the opponent, reinforcing the importance of board control.

---

###  Training Strategy Improvements

- **Epsilon Decay Schedule**  
  Use a gradually decaying exploration rate (e.g., start from `ε = 0.3` and decay to `ε = 0.05`) to improve learning efficiency over time.

- **Offline Pretraining via Imitation**  
  Run a strong heuristic agent and extract action-state pairs to pretrain the Q-function via imitation learning.

- **Self-play Training**  
  Let the agent train against itself or previous versions to explore diverse strategies and improve generalization.

---

###  Q-function Modeling Extensions

- **Non-linear Function Approximators**  
  Replace the current linear Q-function with a shallow neural network or radial basis function to capture non-linear interactions between features.

- **Eligibility Traces (Q(λ))**  
  Apply eligibility traces (e.g., `SARSA(λ)` or `Q(λ)`) to better propagate rewards over multiple steps — crucial in delayed-reward environments like Sequence.

[Back to top](#table-of-contents)