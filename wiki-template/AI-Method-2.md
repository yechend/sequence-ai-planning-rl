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

We conducted head-to-head matches between the MCTS agent and our final GBFS agent, both restricted to 1-second decision windows per move. The MCTS agent usually lost, achieving a win rate of ~40–50%. This confirmed that, while MCTS offers theoretical advantages in deep planning, the limited iteration budget and simulation overhead prevented it from consistently outperforming the more tailored, domain-optimised GBFS strategy.

### Solved Challenges

**1. 1-Second Timeout Constraint**
   
   Deep lookahead and complex evaluation exceeded the 1-second per-move limit.  
   - Solution: Inserted early runtime cutoffs using `time.perf_counter()` and used a fallback to random valid actions if computation exceeded 0.95s.

**2. Heuristic Narrowness**

   Initial heuristics strongly favoured central control, ignoring long-term sequence building.  
   - Solution: Added directional alignment scoring and fork detection to encourage flexible positioning and overlapping potential.

**3. Draft/Hand Uncertainty**

   Unpredictable draft cards after action execution introduced uncertainty.  
   - Solution: Simulated future drafts by sampling from the unseen deck to approximate a more realistic next-step state.

**4. Dead Card Stagnation**
   Players could hold unusable cards that offered no placement opportunities.  
   - Solution: Implemented discard logic that identifies and removes dead cards. The agent simulates a trade and re-evaluates the best moves.

[Back to top](#table-of-contents)


### Trade-offs 

While our Two-Step GBFS agent achieves a strong balance between tactical foresight and real-time decision-making, the design involves several trade-offs. This section outlines the primary advantages and disadvantages observed throughout development and testing. These insights help contextualise the agent's strengths and the practical limitations it faces within a constrained game environment.

#### *Advantages*  

- **Strong Performance in Constrained Settings**  
  Achieves high win rates (up to **87.5%**, and **35/40** in #submission conditions using only 1-second decision windows.

- **Interpretable Heuristic Design**  
  Uses a transparent, domain-informed heuristic function to evaluate alignment potential, centre control, and fork creation. Easy to debug and extend.

- **Robust to Partial Information**  
  Simulates unknowns (e.g., draft card outcomes) using sampling from the unseen deck; does not rely on full knowledge of opponent’s hand.

- **Modular & Extendable**  
  Clean architecture allows logical integration of wildcard logic, discard strategies, opponent-aware penalties, and soft policy guidance if required.

- **Realistic Game Integration**  
  Models Sequence-specific mechanics: wildcard jacks, two-dead-card trades, central control, etc., accurately reflecting game rules.

#### *Disadvantages*

- **Limited Long-Term Planning**  
  Shallow two-step depth cannot anticipate complex multi-turn strategies or future threats.

- **No Strategic Initialisation (Pregame 15s Underused)**  
  Misses the opportunity to build a long-horizon strategy tree, precompute threat maps during the 15-second pregame period, or load a well-trained offline policy model.

- **No True Opponent Modelling**  
  Lacks forecasting of opponent actions. Only reacts to the current board without predicting counterplay or blocking threats in advance.

- **Static Heuristic**  
  The same evaluation is used regardless of game phase. Doesn’t adapt weighting for early control vs. late-game closing moves. 

- **Ineffective Offline Policy Integration**  
  Attempts to integrate self-play-trained policies failed due to poor generalisation and weak inference under complex states.

- **Discard Logic Myopia**  
  Heuristic-based discard may undervalue cards that are not immediately useful but may be critical later (e.g., edge or corner cards).

- **No Heuristic Weight Learning or Tuning**
  All scoring parameters were manually tuned and fixed; no data-driven learning was used to optimise them over time.

[Back to top](#table-of-contents)

### Future improvements  

- **Hybrid MCTS + GBFS Heuristic Search**  
  Use Monte Carlo Tree Search during the 15s pregame to construct a strategic tree. Pair this with fast GBFS pruning in real-time. This would balance deep planning with rapid execution

- **Policy Iteration and Adaptive Heuristics**  
  Use policy gradient or imitation learning to dynamically adjust heuristic weights based on the game phase (early, mid, late) and board context, improving decision adaptability.

- **Opponent Forecasting and Threat Mapping**  
  Model common opponent sequences, use probabilistic inference (e.g., jack threats), and precompute responses for known patterns.

- **Parallel Rollouts or Heuristic Rollout Evaluation**  
  Explore multiple branches through shallow tree rollouts, guided by heuristics. This allows richer strategic exploration without violating the 1-second action time constraint. 

- **Improving Offline Policy Learning**  
  - Replace the majority-random curriculum (48% random) with a stronger and more diverse opponent pool, including high-performing GBFS variants, scripted better defensive agents, and q learning agents.
  - Dynamically reweight policy and value losses during training to refine both action selection and win-rate prediction.
  - Augment training data using symmetry-based transformations (e.g., board flips, rotations) to improve generalisation.
  - Use more sophisticated models and training logics, such as using value head as a simulation critic during search, phase-aware feature engineering, and  contrastive learning or ranking Loss to better learn action preferences, especially under constrained choices.

[Back to top](#table-of-contents)
