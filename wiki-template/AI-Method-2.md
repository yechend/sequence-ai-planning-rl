# AI Method 2 - Monte Carlo Tree Search

This project focuses on building a strategic agent for the board game Sequence, initially using a customised Two-Step Greedy Best-First Search (GBFS). While GBFS proved efficient under the strict 1-second-per-move constraint—achieving a strong win rate of 87.5% (35/40 games)—it was ultimately limited by its shallow search depth, lack of long-term planning, and inability to anticipate opponent responses beyond two moves.

To address these shortcomings, we experimented with a Monte Carlo Tree Search (MCTS) framework with multi-turn foresight and dynamic exploration of the decision space. Our current MCTS approach focuses on multi-turn simulation-based planning, ideal for capturing deeper strategic foresight and responding flexibly to different board states, marking a significant step toward more advanced and adaptable Sequence agents


# Table of Contents
  * [Motivation](#motivation)
  * [Application](#application)
  * [Experiments](#experiments)
  * [Solved Challenges](#solved-challenges)
  * [Trade-offs](#trade-offs)     
     - [Advantages](#advantages)
     - [Disadvantages](#disadvantages)
  * [Future Improvements](#future-improvements)
 
### Motivation  

MCTS addresses the limitation of our previous GBFS-based agent by enabling deeper multi-turn simulations, probabilistic exploration, and adaptive decision-making. Unlike static heuristics, MCTS balances exploitation of strong known moves with exploration of less obvious strategies, offering a more strategic and flexible approach.

By integrating enhanced heuristics into the rollout policy, MCTS allows the agent to better evaluate threats and plan sequence formation. This makes it a natural progression for building a more robust and intelligent Sequence-playing agent. But after thorough testing, this method was not implemented due to xxxxx

[Back to top](#table-of-contents)

### Application

In our MCTS-based agent, the decision-making process is structured around four core components:

- **Tree Policy**  
  We adopt the Upper Confidence Bound for Trees (UCT) strategy to traverse the tree. This policy balances exploration and exploitation by prioritising nodes with high value and low visit count.

- **Simulation Policy**  
  During rollouts, the agent uses a simplified greedy heuristic that:
  - Prioritises completing sequences (5-in-a-row),
  - Blocks opponent threats (e.g., 4 aligned with one open end),
  - Favours placements near the board centre.

This policy allows for efficient, goal-directed simulations without relying on random playouts.

- **Reward Function**  
  We define a sparse terminal reward:
  - `1` if the simulation results in a win (achieving two sequences),
  - `0` otherwise (loss, draw, or timeout).
  
  For non-terminal states in rollouts, we apply a heuristic score to approximate win potential, enabling early backpropagation even if the game doesn't reach completion.

- **Simulation Depth and Iteration Budget**  
  - Rollout Depth: Simulations proceed up to 5 combined turns (player + opponent), or until a win condition is detected.
  - Iteration Budget:
    - Up to 400 iterations per root node during the 15s pregame phase.
    - Runtime-limited to ~0.9s per move during gameplay, typically allowing ~50–100 iterations depending on board complexity.

[Back to top](#table-of-contents)

### Experiments
The initial performance of the model is winning 29 games out of 40 games (72.5%).

We have conducted six experiments to test potential improvements:

**1.Basic MCTS**

We first implemented a basic MCTS agent using a standard UCT policy without any pregame computation. This agent relied on random rollouts and default simulation depth, with no domain-specific enhancements.
  
- **Tree Policy**: UCT with exploration constant `c = 1.41`.
- **Simulation**: Purely random playouts until rollout depth or terminal state.

The agent achieved a 22/40 win rate, which was consistently lower than our previous GBFS agent. It frequently failed to detect long-term threats or form strategic sequences due to the randomness of its simulations.

This baseline demonstrated the importance of adding domain knowledge to guide simulations, as random playouts diluted learning signals and led to poor decision quality.

**2. Heuristic-Guided Rollouts (Implemented)**

To improve the simulation policy, we replaced random playouts with a greedy, heuristic-based simulation. During each rollout, the agent:
- Attempted to complete its own sequences.
- Blocked opponent threats such as 4-in-a-row with open ends.
- Favoured central positions to increase alignment flexibility.

This change led to a slight performance improvement, reaching 24–25 wins out of 40 games on average. While this improvement is not enough to outperform GBFS, it significantly improved rollout consistency and convergence, showing that goal-directed rollouts are far more effective than random ones.
  <img width="1478" alt="image" src="https://github.com/user-attachments/assets/808be35f-950d-43d5-a2c6-53af5dc77217" />

**3. Early Heuristic Backpropagation**

In many rollouts, simulations would not reach a terminal game outcome due to time limits or rollout depth. To improve reward signal quality, we modified the backpropagation strategy:
- If the game did not reach a win/loss, we returned a heuristic evaluation of the resulting board.
- This heuristic included metrics such as alignment potential and fork possibilities.

This tweak helped **stabilise backpropagation and improved rollout feedback quality. Although it didn’t drastically improve win rate alone, it laid the groundwork for more informed tree growth and better node selection.

**4. Extended Rollout Depth (5 → 7)**

We experimented with increasing the rollout depth from 5 to 7 turns (including both agent and opponent actions) in hopes of capturing deeper strategic consequences. While this allowed the agent to simulate more future interactions and occasionally detect stronger plays, it also introduced two major downsides:
  - Fewer rollouts due to longer simulations per iteration
  - More frequent timeouts, especially in complex board states （有没有尝试别的方法解决timeout issues?)

Overall, the gain in depth came at the cost of reduced exploration, which negatively impacted the breadth of the search tree.

**5. Aggressive Tree Expansion**

We lowered the UCT `c_param` to reduce exploration and prioritise exploiting higher-value nodes. The rationale was that with tight time budgets, we should favour reliable known moves over uncertain alternatives. This adjustment led to more consistent decisions and marginally improved game outcomes. However, it introduced a new risk: the agent sometimes became trapped in local optim*, repeatedly choosing a suboptimal move because it lacked exploration data on alternatives.

This experiment highlighted the delicate trade-off between exploration and exploitation, especially under strict timing constraints.

**6. Other Implemented Improvements after Experiments**

To improve MCTS efficiency and reuse, we introduced three key modifications: First, **Ignore Draft Matching** simplifies tree traversal by focusing solely on placement locations, disregarding the specific draft card used—this reduces node duplication and improves subtree matching. Second, a **Unified Rollout Depth Limit** ensures consistent simulation depth by capping all rollouts to a fixed number of player actions (e.g., 5), rather than varying with draft count, leading to more stable and comparable planning. Lastly, **Dynamic Child Expansion** pre-generates all legal moves when the hand changes, improving rollout consistency and enabling better reuse across similar game states.

**7. Comparison with GBFS (Same Time Budget)**

We conducted head-to-head matches between the MCTS agent and our final GBFS agent, both restricted to 1-second decision windows per move. The MCTS agent usually lost, achieving a win rate of ~40–50%. This confirmed that, while MCTS offers theoretical advantages in deep planning, the limited iteration budget and simulation overhead lead to suboptimal performance.

### Solved Challenges

**1. High Branching Factor Under Time Constraint**  
The 10×10 board combined with draft-hand combinations results in a large branching factor, making it difficult to search deeply within the 1-second time limit.  
- Solution: Applied heuristic-based pruning to rank all legal placements using `HeuristicBoard`, retaining only top-K actions for regular and Jack cards to significantly reduce the branching factor.

**2. Tree Reuse Difficulty Due to Partial Observability**  
Frequent updates to hand and draft cards, along with unknown opponent state, prevent subtree reuse across turns.  
- Solution:  
  - Ignored draft card identity, focusing only on positional updates.  
  - Used a fixed rollout draft pool to reduce subtree mismatch from draw randomness.  
  - Applied incremental expansion: new children are generated only for changed positions, improving subtree consistency.

**3. Balancing Heuristics in UCT**  
Over-reliance on heuristic scores early on limited exploration, while pure UCT suffered from reduced performance.  
- Solution: Integrated Progressive Bias into the UCT formula, with adaptive decay of heuristic weights based on visit count, enabling early focus and late-game exploration.

**4. Rollout Evaluation Bias (Random vs. Greedy)**  
Purely random simulations created noisy evaluations, while greedy rollouts risked local optima and narrow exploration.  
- Solution: Adopted a hybrid **heuristic-greedy + depth-capped (6 steps)** simulation policy, blending tactical foresight with variability to stabilise outcomes.

**5. Redundant Heuristic Computation Bottlenecks**  
Frequent re-evaluation of heuristic functions (e.g., `HeuristicBoard`, `CountAlignedChips`) during expansion and backpropagation slowed down search.  
- Solution: Partial mitigation was deployed - heuristic evaluations are batched before sorting child nodes

[Back to top](#table-of-contents)

### Trade-offs 

While our MCTS agent offers deeper foresight and flexible decision-making, it comes with computational and architectural trade-offs. This section reflects how the agent balances strategic planning, runtime efficiency, and practical implementation under time-constrained gameplay conditions.

#### *Advantages*  

- **Deep Strategic Reasoning**  
  MCTS enables a 5-step simulation and evaluation, allowing the agent to discover future winning paths, detect forks, and block threats several turns in advance.

- **Efficient Search with Heuristic Guidance**  
  Integration of Progressive Bias and heuristic-driven rollouts accelerates convergence toward high-value actions while preserving diversity in exploration.

- **Improved Draft Robustness**  
  Ignores draft card identities in tree node keys, focusing on chip positions. This enables subtree reuse and reduces node duplication across stochastic hand/draft transitions.

- **Modular & Scalable**  
  Supports dynamic expansion of valid actions, unified rollout depth, and flexible reward shaping. Easily extendable to incorporate future rollout policies or adaptive tree scoring.

#### *Disadvantages*

- **High Computational Overhead**  
  Tree expansion, UCB scoring, and rollout evaluations are resource-intensive, especially in complex game states, sometimes limiting search depth under the 1-second runtime.

- **Partial Tree Reusability Across Turns**  
  Due to constantly changing hands and drafts, subtree reuse is limited unless abstraction or matching simplification (e.g. ignoring draft) is enforced

- **Sparse Reward Signal**  
  Terminal-only reward assignment makes backpropagation sparse, relying heavily on heuristics in early tree levels, which can introduce estimation bias.

- **Underexplored Opponent Modelling**  
  Our agent does not explicitly simulate or infer the opponent’s cards or likely actions, which could miss defensive opportunities in critical states.

- **Overhead in Preprocessing and Expansion**  
  Dynamic child generation and pruning (Top-K based on heuristic scores) introduce preprocessing latency that must be tightly controlled to avoid runtime violations.

[Back to top](#table-of-contents)

### Future improvements  

- **Precompute Strategic Tree in Pregame Phase**  
  Leverage the 15-second pregame window to run MCTS with deep rollouts (e.g. 400–600 iterations), building a well-initialised root tree. This allows early moves to benefit from extensive planning without violating runtime constraints during actual gameplay.

- **Offline Self-Play Training for Policy Guidance**  
  Similarly, well-designed offline learning policy training could also benefit MCTS in its expansion.

- **Differential State Updates**  
  Replace full board deep copies with delta updates (only modify changed positions and card lists). This optimisation is critical to maintain MCTS feasibility during the pregame simulations, especially under large branching factors.

- **Dynamic Branch Pruning with Adaptive Top-K**  
  Adjust the number of expanded child nodes (Top-K) based on time remaining and board complexity. Nodes with high score variance keep more branches for exploration, while stable nodes are aggressively pruned. 

- **Heuristic-to-UCT Weight Decay**  
  Early in MCTS, emphasise heuristic guidance (bias); as simulations progress, decay the heuristic weight to rely more on visit statistics. This dynamic tuning aligns exploration-exploitation with tree maturity.

[Back to top](#table-of-contents)
