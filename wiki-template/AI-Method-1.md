# AI Method 1 - Two-Step Greedy Best-First Search

This project focuses on building a competitive agent for the board game Sequence using a strategic AI-based approach. While various techniques such as Monte Carlo Tree Search (MCTS) and Q-learning were explored during development, our final agent employs a customised Two-Step Greedy Best-First Search (GBFS). This technique consistently outperformed alternatives in terms of win rate and decision reliability within the given time constraints.

Our final agent incorporates realistic gameplay elements such as random draft card draws, dead card identification, and discard logic. We also experimented with offline policy model guidance to aid move selection, although  it was not deployed in the final agent beyond experimental trials. The best recorded performance of our final agent under #submission achieved a win count of 35 out of 40 games.

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

GBFS was chosen for its ability to perform focused heuristic evaluations in a limited time window (≤ 1s per move). The two-step lookahead extends the depth of planning without exponential time cost. It allowed the agent to prioritise high-impact sequences while retaining reactivity.

Alternative approaches considered:
-	Q-learning: Ultimately rejected, as Q-learning failed to converge meaningfully due to sparse, delayed rewards and ineffective function approximation.
-	MCTS: Initially promising, but led to inconsistent results due to noisy rollouts and variance in simulation depth.
  
Hence, the GBFS with a two-step enhancement was favoured as the most balanced approach for search depth and runtime feasibility.

[Back to top](#table-of-contents)

### Application  
We modelled the problem as follows:
-	State Definition: A state is defined by three elements: the board layout (10×10 chips), the agent’s current hand, and the available draft cards.
-	Goal State: The agent aims to either form two completed sequences or fully occupy all four central tiles.
-	Heuristic Function:
    -	Encourage proximity to centre: adds positional bonus for tiles closer to the centre.
    - Prioritise alignment potential:
      - +200 for a complete sequence.
      -	+90 for four aligned chips with open ends.
      -	+50 for three aligned with two open ends.
      -	+20 for two aligned with two open ends.
    - Detect and reward fork potential (i.e., multiple alignments).

No explicit modeling of opponent behavior is used—strategy assumes a reactive opponent, simplifying planning while still performing competitively.

[Back to top](#table-of-contents)

### Experiments
The initial performance of the model is winning 29 games out of 40 games (72.5%).
<img width="1378" alt="image" src="https://github.com/user-attachments/assets/dceabbe0-d8a8-485c-b589-9e76f6980f27" />

We have conducted seven thorough experiments to test potential improvements:

**1. Advanced Wildcard Strategies**

We experienced a version implementing the following:
- **Two-Eyed Jacks (wild cards)**: Used to place a chip anywhere. Prioritise completing a sequence, blocking an opponent’s win, or occupying high-value spots (e.g. forks or centre).
- **One-Eyed Jacks (removers)**: Remove opponent chips that are critical—part of a 4-in-a-row or occupying strategic positions. Always override normal logic if an immediate threat is detected.
  
To comply with a one-second-per-move constraint, this had to be implemented under `GeneratePlacingActions` rather than `SelectionAction`. In local testing, the updated agent won 21 out of 40 matches against the initial baseline model, demonstrating moderate improvement. However, during official submission trials, its performance dropped with a win rate of 27/40, suggesting limited gains under varied opponent conditions. We proposed the causes to be two main factors: over-prioritising wildcard, which might lead to ignoring better tactical placement, and the inherent limitation of the one-step decision process, which lacks the contextual foresight to evaluate whether removing or placing a chip offers a more strategic long-term benefit.

<img width="1378" alt="image" src="https://github.com/user-attachments/assets/18665a14-ca99-4931-8cff-e6899daae822" />

**3. Multi-Step Search - 3-Step Extension**

To further enhance the agent’s planning capability, we extended our search depth from two steps to a three-step Greedy Best-First Search. The rationale was to better anticipate the outcome of a sequence of plays and better account for opponent threats or opportunities arising after our initial two actions. This would theoretically allow our agent to block sequences earlier, set up forks more reliably, and avoid short-sighted placements.

To implement this, we modified the existing two-step logic to include one additional simulated action. However, despite promising expectations, this deeper search came with notable **trade-offs**:

- **Severe time constraints**: The increased branching factor caused frequent timeout violations. The agent often approached the 1-second limit and had to revert to a 2-step search mid-loop. We tested state evaluation caching and reuse, and reverted to 2-step when approaching the time limit, but didn’t sufficiently resolve this issue. 

- **Efficiency bottlenecks**: While caching and reuse of state evaluations improved speed in some cases, generating realistic opponent hands and valid moves added computational burden due to the unknown condition of the opponent's complete hand.

- **No significant gain in test performance**: We removed the 1-second limit to test the potential of this extension. On average, it achieved a win rate of 23/40, suggesting rather minor improvements due to imperfect state simulation.

While the 3-step model offered better tactical foresight in theory, it underperformed in practice due to runtime limits, computational overhead, and imperfect modelling of hidden game elements. We thus reverted to a more efficient 2-step GBFS.

**4. Opponent Modelling and Threat Blocking**

To enhance decision-making, we integrated opponent-aware heuristic tuning and explored basic card inference logic. These adjustments aim to penalise board states that are strategically advantageous to the opponent, aligning with defensive priorities.

- **Tuning Parameter**:

  The tuning parameter α controlled the weights of the opponent's heuristic score under `HeuristicScore`. We experimented with different settings with their associated win rate against the baseline model.

  - `α = 0.0`: 55% win rate (maximum performance)  
  - `α = 0.1`: 52.5%  
  - `α = 0.3`: 40%  
  - `α = 0.5`: 35%

  This suggests that focusing purely on the agent’s score (`α = 0`) was more effective in practice, likely due to the simplicity and consistency of self-oriented evaluation under tight time constraints.

- **Opponent-aware Heuristic Penalty**:  
  We further tested applying a penalty in the heuristic if the opponent had:
  - 4 aligned chips with one open end (imminent threat)
  - 3 aligned chips with two open ends (high potential threat)

  This logic was embedded into both the `HeuristicScore` function and a pre-search check to ensure emergency blocking actions are prioritised (e.g., one-eyed Jack removals or direct placements to deny key spots). However, this method provided no noticeable performance improvement. We hypothesise that the added complexity diluted the agent’s focus on its strategic buildup, especially under the tight time constraint per move.

- **Card Inference**:  

  While we considered integrating card memory to estimate the likelihood of the opponent possessing a specific card (i.e., if both copies had been seen), this was not ultimately implemented due to complexity and time cost. The agent assumes the worst-case scenario for critical threats, which is a safer but less advanced approach.

**5. Sequence Extension Bonus & Fork Detection (Implemented)**

Since the heuristic function is the core of the agent’s decision-making in the GBFS framework, we initially added a small bonus for placements that would extend to 6-in-a-row under exiting 4-in-a-row (which doesn’t count as a second sequence by rules), recognising the strategic value of overlapping sequence plans. But this did improve the performance due to its limited impact on game outcomes. We then experimented with **fork detection** by evaluating how many sequence lines that position could extend. If a tile contributes to multiple alignment directions, it’s awarded extra points to reflect forming a sequence potential. This refinement raised the local win rate to 55% against the baseline model and was subsequently adopted in our final heuristic design.

**6. Card Discard Logic (Implemented)**

We further employed a two-level discard strategy to maintain hand efficiency and avoid wasted turns:

- **Dead Card Discard**:  
  If a card cannot be legally placed on the board (i.e., all its associated positions are occupied), it is classified as a dead card and immediately discarded.

- **Low-Value Card Discard**:  
  If no dead cards are found, the agent evaluates all playable cards using the heuristic function and discards the one with the lowest strategic potential.

This deployment achieved a 57.5% win rate against our previously refined model from **5. – Sequence Extension Bonus & Fork Detection**. In online testing, it reached a peak performance of 33 wins out of 40 games. We included this refinement in our final agent.
<img width="1378" alt="image" src="https://github.com/user-attachments/assets/db5fa0ca-b5f7-4cf2-90e7-ca7f8a83150a" />
<img width="1378" alt="image" src="https://github.com/user-attachments/assets/ce6ee26c-a1e6-41b9-96ce-5145c72c0eff" />

**7. Offline Self-Play Training & Policy Networks**

To better use the 15-second pregame loading time, we explored offline policy training via a cold-start strategy. The rationale was to precompute a general decision policy to guide early-game actions before the real-time search activates, reducing reliance on online computation and improving responsiveness. The model takes encoded board states as input and outputs both action probabilities (policy head) and win likelihoods (value head).

- Initially, we deployed `train_sequence_policy.py` to train against the random agent. But its evaluation metrics (accuracy of the policy head, mean Absolute Error (MAE) of the value head, the validation accuracy of the policy head, the validation MAE of the value head) suggested severe underfitting and might not be learning meaningful action patterns.
- To resolve this, we further deployed `curriculum_trainer.py` to self-play using curriculum learning over 4,000 games against diverse opponents (random 48%, blocker 4%, and our GBFS agent 48%). Due to time limitations, this allocation hadn't been tuned properly to select the optimal split and no other agents were used to train this final model (`policy_value_model_curriculum.keras`). The final training results showed limited improvements as compared to the first training method. 
  | Metric              | Random-Only | Curriculum  |
  |-------------------------|--------------------------|------------------------|
  | `policy_accuracy`       | ~0.014 → 0.086           | 0.009 → 0.106      |
  | `value_mae`             | ~0.42 → 0.03             | 0.38 → 0.03        |
  | `val_policy_accuracy`   | max ~0.0426              | up to 0.1064      |
  | `val_value_mae`         | ~0.25                    | ~0.04              

While we acknowledge that the trained policy model (even under curriculum learning) is suboptimal, we deliberately integrated it into our agent as a hypothesis-driven experiment. Our aim was not to rely on it for full decision-making, but to test how lightweight offline models might assist GBFS in specific scenarios. We conducted the following experimental trials under the removal of the 1s constraint:

1. **Filtering by Top-10, Top-3, Top-1 Policy Prediction**  
   → This resulted in poor action diversity and failed to significantly guide optimal choices.

2. **Tuning Policy-Heuristic Weighting (α = 0.1 vs. 0.9)**  
   → No gain in win rate; higher policy influence sometimes led to irrelevant moves.

3. **Soft Bonus Integration (e.g., +0.1 × policy score)**  
   → Minor improvement in some settings, but still inconsistent due to poor policy.
   
The results supported our hypothesis that the naively trained policy model tends to overfit to simple board patterns and fails to generalise well to more complex situations. It provided no noticeable performance improvement and, in several cases, selected clearly suboptimal moves despite better available options. Additionally, the strict 1-second decision limit made it impractical to perform deeper simulations or corrections based on policy suggestions.

Although the model did not enhance gameplay performance, these experiments offered valuable insights. They highlighted key limitations of offline-trained policies and helped inform how future self-play or imitation learning approaches might be better designed and integrated.

**8. Teacher Cai**

**9. ABC**

[Back to top](#table-of-contents)

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
