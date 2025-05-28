# AI Method 1 - Two-Step Greedy Best-First Search

This project focuses on building a competitive agent for the board game Sequence using a strategic AI-based approach. While various techniques such as Monte Carlo Tree Search (MCTS) and Q-learning were explored during development, our final agent employs a customised Two-Step Greedy Best-First Search (GBFS). This technique consistently outperformed alternatives in terms of win rate and decision reliability within the given time constraints.

Our final agent incorporates realistic gameplay elements such as random draft card draws, dead card identification, and discard logic. We also experimented with offline policy model guidance to aid move selection, although  it was not deployed in the final agent beyond experimental trials. The best recorded performance of our final agent under #submission achieved a win count of 35 out of 40 games.

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

GBFS was chosen for its ability to perform focused heuristic evaluations in a limited time window (≤ 1s per move). The two-step lookahead extends the depth of planning without exponential time cost. It allowed the agent to prioritise high-impact sequences while retaining reactivity.

Alternative approaches considered:
-	Q-learning: Ultimately rejected, as Q-learning failed to converge meaningfully due to sparse, delayed rewards and ineffective function approximation.
-	MCTS: Initially promising, but led to inconsistent results due to noisy rollouts and variance in simulation depth.
  
Hence, the GBFS with a two-step enhancement was favoured as the most balanced approach for search depth and runtime feasibility.

[Back to top](#table-of-contents)

### Application  
Problem Modeling
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
- Two-Eyed Jacks (wild cards): Used to place a chip anywhere. Prioritise completing a sequence, blocking an opponent’s win, or occupying high-value spots (e.g. forks or centre).
- One-Eyed Jacks (removers): Remove opponent chips that are critical—part of a 4-in-a-row or occupying strategic positions. Always override normal logic if an immediate threat is detected.
  
To comply with a one-second-per-move constraint, this had to be implemented under `GeneratePlacingActions` rather than `SelectionAction`. In local testing, the updated agent won 21 out of 40 matches against the initial baseline model, demonstrating moderate improvement. However, during official submission trials, its performance dropped with a win rate of 27/40, suggesting limited gains under varied opponent conditions. We proposed the causes to be two main factors: over-prioritising wildcard, which might lead to ignoring better tactical placement, and the inherent limitation of the one-step decision process, which lacks the contextual foresight to evaluate whether removing or placing a chip offers a more strategic long-term benefit.

<img width="1520" alt="image" src="https://github.com/user-attachments/assets/18665a14-ca99-4931-8cff-e6899daae822" />

**3. Multi-Step Search**

**4. Opponent Modelling and Threat Blocking**

**5. Refined Heuristic Evaluation**

**6. Card Discard Logic**

**7. Offline Self-Play Training & Policy Networks**

**8. Cai Loashi**

**9. ABC**
[Back to top](#table-of-contents)

### Solved Challenges

[Back to top](#table-of-contents)


### Trade-offs  
#### *Advantages*  


#### *Disadvantages*

[Back to top](#table-of-contents)

### Future improvements  

[Back to top](#table-of-contents)
