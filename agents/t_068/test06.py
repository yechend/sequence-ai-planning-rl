from template import Agent
from Sequence.sequence_model import SequenceGameRule as GameRule
import time, random
from copy import deepcopy
from Sequence.sequence_model import COORDS
from Sequence.sequence_utils import EMPTY

MAX_THINK_TIME = 0.95
SAFETY_BUFFER = 0.02
CENTER_COORDS = [(4, 4), (4, 5), (5, 4), (5, 5)]

try:
    policy_model = load_policy()  # Assume function exists to load network weights
except Exception as e:
    policy_model = None  # If not available, proceed without real network (use dummy logic)

# Data structures for caching and precomputed openings
opening_move_values = {}       # cache for precomputed opening move evaluations
opening_policy = {}            # optional deeper opening strategy cache (state -> best move)
heuristic_cache = {}           # cache for heuristic evaluations to avoid recomputation
precomputed_openings_done = False

def precompute_openings(initial_state, agent_id=0, simulations_per_move=30):
    """Use the 15-second pre-game phase to simulate opening moves and cache their values."""
    global precomputed_openings_done, opening_move_values, opening_policy
    # Generate all possible first actions from the initial state (based on agent's hand)
    legal_actions = GenerateActions(initial_state, agent_id)  # assume this function exists
    # Monte Carlo simulate outcomes for each action to estimate its value
    for action in legal_actions:
        wins = 0
        total_score = 0.0
        for sim in range(simulations_per_move):
            # Simulate a game (or partial game) after taking this action
            state_copy = initial_state.copy()       # assume state has copy method
            simulate_action_inplace(state_copy, action, agent_id)  # apply our move
            # Play out the rest of the game (or some depth) with random/greedy policy
            result = simulate_greedy_game(state_copy, agent_id)   # simulate to end or fixed depth
            # `result` could be final outcome (win=1, loss=0) or heuristic score difference
            if result == agent_id or result == 1:   # if we treat result as win flag
                wins += 1
            elif result == -1 or result == (1 - agent_id):
                # Loss for our agent (depending on simulate_greedy_game return format)
                wins += 0
            elif isinstance(result, (int, float)):
                total_score += result
        # Estimate win rate or average score for this action
        if wins > 0 or simulations_per_move > 0:
            action_value = wins / simulations_per_move if wins > 0 else total_score / simulations_per_move
        else:
            action_value = 0
        opening_move_values[action] = action_value
    # Optionally, choose the best opening action and store in opening_policy
    if opening_move_values:
        best_opening = max(opening_move_values, key=opening_move_values.get)
        opening_policy[initial_state.hash()] = best_opening  # store by state hash if available
    precomputed_openings_done = True

# --- Heuristic Evaluation Function with Advanced Features ---
def HeuristicBoard(state, agent_id):
    """Evaluate the board state with advanced heuristics for the given agent."""
    global heuristic_cache
    opp_id = 1 - agent_id
    # Use board configuration as cache key (ignoring cards in hand for heuristic)
    board_key = state.board.hash_state() if hasattr(state.board, "hash_state") else tuple(state.board.chips_layout())
    # Include agent_id to differentiate perspective if needed
    cache_key = (board_key, agent_id)
    if cache_key in heuristic_cache:
        return heuristic_cache[cache_key]
    score = 0

    # Count completed sequences for each player (assume function or property available)
    my_seq = state.board.count_sequences(agent_id) if hasattr(state.board, "count_sequences") else count_sequences(state.board, agent_id)
    opp_seq = state.board.count_sequences(opp_id) if hasattr(state.board, "count_sequences") else count_sequences(state.board, opp_id)
    # Large reward/penalty for completed sequences
    score += my_seq * 1000
    score -= opp_seq * 1000
    # If either side has reached the win condition (e.g., 2 sequences), handle accordingly
    # (Here we assume game ends at 2 sequences for a player)
    if my_seq >= 2:
        heuristic_cache[cache_key] = 99999  # winning state
        return 99999
    if opp_seq >= 2:
        heuristic_cache[cache_key] = -99999  # losing state
        return -99999

    # Count runs of length 2,3,4 (contiguous chips) for both players, not yet sequences
    # and consider central control, forks, etc.
    my_runs4 = count_runs(state.board, agent_id, 4)
    opp_runs4 = count_runs(state.board, opp_id, 4)
    my_runs3 = count_runs(state.board, agent_id, 3)
    opp_runs3 = count_runs(state.board, opp_id, 3)
    my_runs2 = count_runs(state.board, agent_id, 2)
    opp_runs2 = count_runs(state.board, opp_id, 2)

    # Base run scores: emphasize runs of 4 strongly, then 3, then 2
    score += my_runs4 * 50  - opp_runs4 * 50
    score += my_runs3 * 15  - opp_runs3 * 15
    score += my_runs2 * 5   - opp_runs2 * 5

    # Central control: bonus for chips in center region (e.g., middle 4x4 of board)
    center_coords = [(r, c) for r in range(state.board.rows) for c in range(state.board.cols)
                     if 3 <= r <= (state.board.rows-4) and 3 <= c <= (state.board.cols-4)]
    center_control_my = sum(1 for pos in center_coords if state.board.get(pos) == agent_id)
    center_control_opp = sum(1 for pos in center_coords if state.board.get(pos) == opp_id)
    score += center_control_my * 2
    score -= center_control_opp * 2

    # Fork potential: count positions (chips) that contribute to multiple lines of 5 for agent
    my_fork_bonus = 0
    opp_fork_threats = 0
    # Evaluate each chip of agent: how many potential sequence lines include this chip
    for pos in state.board.get_positions(agent_id):
        lines = lines_through_position(pos)  # all sequence-lines (5 in a row sets) that pass through pos
        count_open_lines = 0
        for line in lines:
            # Check if this line is still open for agent (no opponent chip blocking)
            blocked = any(state.board.get(p) == opp_id for p in line)
            if not blocked:
                # Check how many of agent's chips in this line
                agent_count = sum(1 for p in line if state.board.get(p) == agent_id or state.board.is_corner(p))
                if agent_count >= 2:  # at least part of a run
                    count_open_lines += 1
        if count_open_lines >= 2:
            my_fork_bonus += 1  # chip is part of two potential sequences
    # Opponent fork threat: count if opponent has any position that lies in multiple open lines
    for pos in state.board.get_positions(opp_id):
        lines = lines_through_position(pos)
        count_open_lines = 0
        for line in lines:
            blocked = any(state.board.get(p) == agent_id for p in line)
            if not blocked:
                opp_count = sum(1 for p in line if state.board.get(p) == opp_id or state.board.is_corner(p))
                if opp_count >= 2:
                    count_open_lines += 1
        if count_open_lines >= 2:
            opp_fork_threats += 1

    score += my_fork_bonus * 10  # reward our fork opportunities
    score -= opp_fork_threats * 10  # penalize opponent forks

    # Phase-based adjustment: alter weights if late game vs early game
    total_chips_placed = len(state.board.get_positions(agent_id)) + len(state.board.get_positions(opp_id))
    if my_seq > 0 or opp_seq > 0 or total_chips_placed > 40:
        # Late game: increase weight on immediate threats (runs4) and sequences, less on long-term (center)
        score += my_runs4 * 20  - opp_runs4 * 20   # extra emphasis
        score += my_runs3 * 5   - opp_runs3 * 5    # moderate
        # Early game: else case - already covered by base weights (which emphasize center, forks etc.)

    heuristic_cache[cache_key] = score
    return score

# Helper functions for heuristic (assuming no heavy new classes)
# We'll implement count_sequences and count_runs if not provided by state.board:
def count_sequences(board, player_id):
    """Count how many 5-in-a-row sequences player_id has on the board."""
    count = 0
    # Check all possible starting positions and directions for 5-length lines
    directions = [(0,1),(1,0),(1,1),(1,-1)]
    rows = board.rows if hasattr(board, "rows") else len(board.grid)
    cols = board.cols if hasattr(board, "cols") else len(board.grid[0])
    for r in range(rows):
        for c in range(cols):
            for dr, dc in directions:
                # Only count sequences starting from smallest index to avoid double counting
                if not (0 <= r-dr < rows and 0 <= c-dc < cols):
                    # make sure this is a start (previous cell in direction is out of bounds)
                    seq = True
                    for i in range(5):
                        nr, nc = r + dr*i, c + dc*i
                        if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                            seq = False
                            break
                        cell = board.get((nr, nc))
                        # treat corner as wildcard chip for both players
                        if board.is_corner((nr, nc)):
                            # corner counts as filled by player
                            continue
                        if cell != player_id:
                            seq = False
                            break
                    if seq:
                        count += 1
    return count

def count_runs(board, player_id, length):
    """Count runs of a given length (contiguous chips) for player_id, not blocked by opponent."""
    runs = 0
    directions = [(0,1),(1,0),(1,1),(1,-1)]
    rows = board.rows if hasattr(board, "rows") else len(board.grid)
    cols = board.cols if hasattr(board, "cols") else len(board.grid[0])
    for r in range(rows):
        for c in range(cols):
            for dr, dc in directions:
                # Only check runs starting at this position in the negative direction is either out of bounds or an opponent (to avoid counting mid-runs multiple times)
                prev_r, prev_c = r-dr, c-dc
                if 0 <= prev_r < rows and 0 <= prev_c < cols:
                    # If previous cell in this direction is within bounds, ensure it is blocked or out-of-line
                    prev_val = board.get((prev_r, prev_c))
                    if prev_val == player_id or board.is_corner((prev_r, prev_c)):
                        continue  # if previous in line is also our chip or a corner, then (r,c) is mid-run, skip
                # Now count contiguous chips (with corners as our chips) from this start
                count = 0
                blocked = False
                for i in range(0, length):
                    nr, nc = r + dr*i, c + dc*i
                    if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                        blocked = True
                        break
                    cell = board.get((nr, nc))
                    if cell == player_id or board.is_corner((nr, nc)):
                        count += 1
                    elif cell == 1 - player_id:
                        blocked = True
                        break
                    else:
                        # empty space breaks contiguous run for counting
                        blocked = True
                        break
                if not blocked and count == length:
                    # Ensure the sequence is not longer than 'length' (next cell is not also our chip)
                    next_r, next_c = r + dr*length, c + dc*length
                    if 0 <= next_r < rows and 0 <= next_c < cols:
                        next_val = board.get((next_r, next_c))
                        if next_val == player_id or board.is_corner((next_r, next_c)):
                            continue  # run is actually longer than 'length'
                    runs += 1
    return runs

# If not provided, we define a quick way to get all sequence lines passing through a position:
def lines_through_position(pos):
    """Return all 5-length lines (list of positions) on the board that include the given position."""
    lines = []
    row, col = pos
    directions = [(0,1),(1,0),(1,1),(1,-1)]
    # For each direction, find the 5-length line that includes pos (if any)
    for dr, dc in directions:
        # find starting point by going backward up to 4 steps
        for back in range(5):
            start_r = row - dr*back
            start_c = col - dc*back
            line = [(start_r + dr*i, start_c + dc*i) for i in range(5)]
            # Check bounds
            if all(0 <= r < (state.board.rows if hasattr(state.board, "rows") else len(state.board.grid)) and
                   0 <= c < (state.board.cols if hasattr(state.board, "cols") else len(state.board.grid[0]))
                   for (r, c) in line):
                if pos in line:
                    lines.append(line)
                    break
    return lines

# --- Two-Step Lookahead Search with Monte Carlo Tree Search Integration ---
def TwoStepLookaheadSearch(state, agent_id):
    """Select the best action using a limited-depth search with opponent modeling and MCTS rollouts."""
    legal_actions = GenerateActions(state, agent_id)
    best_action = None
    best_value = -float('inf')
    # Number of simulations per action (could adjust based on time or game phase)
    sims_per_action = 5  # e.g., do 5 rollouts for each action within 1 sec
    # If early game (no sequences yet), maybe fewer sims to save time for broad search
    my_seq = state.board.count_sequences(agent_id) if hasattr(state.board, "count_sequences") else count_sequences(state.board, agent_id)
    opp_seq = state.board.count_sequences(1-agent_id) if hasattr(state.board, "count_sequences") else count_sequences(state.board, 1-agent_id)
    if my_seq == 0 and opp_seq == 0:
        sims_per_action = 3
    for action in legal_actions:
        # If we have policy network priors, we can bias selection order by that
        prior_prob = 0
        if policy_model:
            try:
                prior_prob = policy_model.predict_policy(state).get(action, 0)
            except:
                prior_prob = 0
        # If action is obviously winning (completes second sequence), return immediately
        # Check if this action yields win (complete 2 sequences)
        # We simulate the action quickly:
        temp_state = state.copy()
        simulate_action_inplace(temp_state, action, agent_id)
        new_my_seq = temp_state.board.count_sequences(agent_id) if hasattr(temp_state.board, "count_sequences") else count_sequences(temp_state.board, agent_id)
        if new_my_seq >= 2:
            return action  # immediate win by completing second sequence
        # Evaluate action via simulations
        total_val = 0.0
        for sim in range(sims_per_action):
            # Create a fresh copy of state for simulation
            sim_state = state.copy()
            simulate_action_inplace(sim_state, action, agent_id)  # our move
            # Opponent move simulation: model opponent's response
            # We consider opponent might either play optimally (best heuristic move) or randomly to account for uncertainty
            opp_actions = GenerateActions(sim_state, 1 - agent_id)
            if opp_actions:
                # Choose opponent action:
                if sim == 0:
                    # First simulation: assume opponent plays best (minimizes our score)
                    opp_best_val = float('inf')
                    opp_best_action = None
                    for opp_action in opp_actions:
                        state_after_opp = sim_state.copy()
                        simulate_action_inplace(state_after_opp, opp_action, 1 - agent_id)
                        val = HeuristicBoard(state_after_opp, agent_id)
                        if val < opp_best_val:
                            opp_best_val = val
                            opp_best_action = opp_action
                    simulate_action_inplace(sim_state, opp_best_action, 1 - agent_id)
                else:
                    # Other simulations: pick a random opponent move (sample a possible response)
                    opp_act = random.choice(opp_actions)
                    simulate_action_inplace(sim_state, opp_act, 1 - agent_id)
            # Agent's second move (greedy continuation) - simulate one more move for our agent to see potential
            follow_actions = GenerateActions(sim_state, agent_id)
            if follow_actions:
                # Choose the action that maximizes heuristic (greedy rollout policy for our agent)
                follow_best = None
                follow_best_score = -float('inf')
                for follow_act in follow_actions:
                    follow_state = sim_state.copy()
                    simulate_action_inplace(follow_state, follow_act, agent_id)
                    score = HeuristicBoard(follow_state, agent_id)
                    if score > follow_best_score:
                        follow_best_score = score
                        follow_best = follow_act
                # Apply the follow-up move
                if follow_best:
                    simulate_action_inplace(sim_state, follow_best, agent_id)
            # After simulated sequence of moves, evaluate resulting state
            final_score = HeuristicBoard(sim_state, agent_id)
            total_val += final_score
        # Average the value over simulations and optionally incorporate policy prior as a bias
        avg_value = total_val / sims_per_action
        # Incorporate a small bias from policy prior (e.g., add prior_prob scaled)
        avg_value += 0.1 * prior_prob
        # Dual-sequence threat detection: if this action creates a "fork" (two threats)
        # Check if after our action (and before opponent response) we have multiple 4-in-a-rows
        # We'll use the temp_state from above (after our action, before opponent)
        my_runs4_after = count_runs(temp_state.board, agent_id, 4)
        if my_runs4_after >= 2 and my_seq < 2:
            # If we created two separate 4-in-a-row threats, boost value (especially if we already had 1 sequence, this might be a guaranteed win scenario)
            boost = 50
            if my_seq >= 1:
                boost = 200  # if we already have one sequence, two threats = nearly certain win next turn
            avg_value += boost
        # Opponent dual-sequence threat: if our action fails to address a major opponent fork threat, penalize
        # e.g., if opponent currently has two 4-in-a-rows (fork) and our move didn't block at least one, it's dangerous
        opp_runs4_cur = opp_runs4  # from initial state
        # We can recalc opp runs4 after our move:
        opp_runs4_after = count_runs(temp_state.board, opp_id, 4)
        if opp_runs4_after > opp_runs4_cur:
            # If opponent still has a fork threat (or gained one) after our move, slight penalty
            avg_value -= 20

        # Choose best action by highest average simulation value
        if avg_value > best_value:
            best_value = avg_value
            best_action = action
    return best_action

# --- Main Action Selection integrating all strategies ---
def SelectAction(state):
    """Decide the best action for our agent given the current state, using integrated strategies."""
    agent_id = state.agent_id  # assume agent_id indicates which player we are
    opp_id = 1 - agent_id
    global precomputed_openings_done

    # Use pre-game simulation results if not already done
    if not precomputed_openings_done:
        precompute_openings(state, agent_id)
    # If an opening policy suggestion exists for this exact state, consider using it
    state_hash = state.hash() if hasattr(state, "hash") else None
    if state_hash and state_hash in opening_policy:
        suggested_action = opening_policy[state_hash]
        # If suggestion is still legal (the board state matches expected), we can use it
        legal_actions = GenerateActions(state, agent_id)
        if suggested_action in legal_actions:
            return suggested_action

    # 1. Priority: Use One-Eyed Jack to remove a critical opponent chip (threat blocking)
    for card in state.hand:
        if card.isOneEyedJack():
            target = find_critical_opponent_chip(state, opp_id)
            if target:
                # Ensure removal is legal (not part of completed sequence)
                if not state.board.is_part_of_sequence(target):
                    return Action.remove(opp_id, target)

    # 2. Priority: Use Two-Eyed Jack (wild) for immediate win or block if available
    for card in state.hand:
        if card.isTwoEyedJack():
            best_jack_move = None
            best_jack_score = -float('inf')
            for pos in state.board.empty_positions():
                # Simulate placing at pos
                sim_state = state.copy()
                sim_state.board.place_chip(pos, agent_id)  # place chip for simulation
                # Check if this yields an immediate win (two sequences)
                new_my_seq = sim_state.board.count_sequences(agent_id) if hasattr(sim_state.board, "count_sequences") else count_sequences(sim_state.board, agent_id)
                new_opp_seq = sim_state.board.count_sequences(opp_id) if hasattr(sim_state.board, "count_sequences") else count_sequences(sim_state.board, opp_id)
                if new_my_seq >= 2:
                    # Placing here wins the game
                    return Action.place(card, pos)
                if new_opp_seq >= 2:
                    # Placing here would inadvertently give opponent a win (should not happen with wild placement, as we place our chip, but just in case, skip)
                    continue
                # Otherwise evaluate heuristic of resulting board
                score = HeuristicBoard(sim_state, agent_id)
                # If this placement also blocks an opponent immediate sequence threat, that score will be much higher than others
                if score > best_jack_score:
                    best_jack_score = score
                    best_jack_move = Action.place(card, pos)
            if best_jack_move:
                return best_jack_move

    # 3. Opponent threat block using normal card:
    # If opponent has a likely win next turn (open 4 in a row), and we have the specific card to block it, do so.
    opp_threat = find_open_four(state, opp_id)  # find an empty position that completes opponent's 5
    if opp_threat:
        # Check if we can play a card for that position (or a wild, but wild handled above)
        for card in state.hand:
            if not card.isJack() and state.board.card_position(card) == opp_threat:
                # If this card corresponds to that threatened position
                return Action.place(card, opp_threat)

    # 4. Use Monte Carlo Tree Search / Lookahead for best action otherwise
    best_action = TwoStepLookaheadSearch(state, agent_id)
    if best_action:
        return best_action

    # 5. If no action found (should not happen normally), fallback: discard a dead card if possible
    dead_cards = [card for card in state.hand if (not card.isJack()) and state.board.card_position(card) is None]
    if dead_cards:
        # If we have a card that cannot be played (both spots occupied), discard it to get a fresh card
        return Action.discard(dead_cards[0])
    # If no dead card, just discard the first card (as a denial or cycle tactic) if we must take an action
    return Action.discard(state.hand[0])

# --- Opponent Modeling Helper: Critical Threat Detection ---
def find_critical_opponent_chip(state, opp_id):
    """Find an opponent chip that is critical to remove (nearly completing a sequence or forming a dangerous fork)."""
    agent_id = 1 - opp_id
    critical_target = None
    # Check for opponent double-sequence (fork win) threat: find any empty that would give opponent 2 sequences
    # If found, target one chip from one of those sequences to remove
    for pos in state.board.empty_positions():
        # If opponent placing at pos yields two sequences, that's critical
        # Simulate opponent placing here
        sim_state = state.copy()
        sim_state.board.place_chip(pos, opp_id)
        new_opp_seq = sim_state.board.count_sequences(opp_id) if hasattr(sim_state.board, "count_sequences") else count_sequences(sim_state.board, opp_id)
        if new_opp_seq >= 2:
            # pos is a spot that would let opponent win outright with two sequences at once
            # Identify one chip from those two sequence lines to remove
            lines = lines_through_position(pos)
            for line in lines:
                if all(sim_state.board.get(p) == opp_id or sim_state.board.is_corner(p) or p == pos for p in line):
                    # line is a full sequence if pos were filled
                    # remove one of the real chips from this line (not a corner, not pos itself)
                    for p in line:
                        if state.board.get(p) == opp_id:
                            return p  # critical chip to remove
    # If no double-sequence threat, check for any opponent 4-in-a-row that could become a sequence next turn
    opp_runs4_positions = []  # collect one position from each opponent run of 4 (which is empty to complete sequence)
    # Scan for opponent runs of 4 with one empty at ends or through a corner
    directions = [(0,1),(1,0),(1,1),(1,-1)]
    rows = state.board.rows if hasattr(state.board, "rows") else len(state.board.grid)
    cols = state.board.cols if hasattr(state.board, "cols") else len(state.board.grid[0])
    for r in range(rows):
        for c in range(cols):
            for dr, dc in directions:
                # potential start of a 5-length line
                line = [(r + dr*i, c + dc*i) for i in range(5)]
                if not all(0 <= rr < rows and 0 <= cc < cols for rr, cc in line):
                    continue
                # count opponent chips and empties in this line
                opp_count = 0
                empties = []
                invalid = False
                for (rr, cc) in line:
                    if state.board.is_corner((rr, cc)):
                        # corner counts as filled by both, treat as opp chip for sequence formation
                        opp_count += 1
                    else:
                        cell = state.board.get((rr, cc))
                        if cell == opp_id:
                            opp_count += 1
                        elif cell == agent_id:
                            invalid = True
                            break  # our chip blocks this line, not a threat
                        else:
                            empties.append((rr, cc))
                    if opp_count + len(empties) < (line.index((rr, cc))+1) - 0:
                        # quick break: if by this point not enough chips to reach required length
                        pass
                if invalid:
                    continue
                # For a nearly complete sequence, opp_count should be 4 and empties 1
                if opp_count == 4 and len(empties) == 1:
                    # This line is one move away from opponent sequence
                    opp_runs4_positions.append(empties[0])
    # If any nearly-complete sequence found, try to remove one chip from that line
    if opp_runs4_positions:
        # If multiple, just take the first for now (could choose the one with most impact)
        threat_pos = opp_runs4_positions[0]
        # Find the opponent chip in that line (other than corners) that is most critical
        # For simplicity, remove one of the chips adjacent to the empty spot
        for dr, dc in directions:
            # Check neighbor on each side of empty in line direction if it's opponent chip
            nb1 = (threat_pos[0] - dr, threat_pos[1] - dc)
            nb2 = (threat_pos[0] + dr, threat_pos[1] + dc)
            if 0 <= nb1[0] < rows and 0 <= nb1[1] < cols and state.board.get(nb1) == opp_id:
                return nb1
            if 0 <= nb2[0] < rows and 0 <= nb2[1] < cols and state.board.get(nb2) == opp_id:
                return nb2
    return None

def find_open_four(state, player_id):
    """Find an empty position that completes a sequence of 5 for the given player (open four)."""
    # Similar to above but directly returns the empty position that would complete player's sequence if exists
    opp_id = 1 - player_id
    directions = [(0,1),(1,0),(1,1),(1,-1)]
    rows = state.board.rows if hasattr(state.board, "rows") else len(state.board.grid)
    cols = state.board.cols if hasattr(state.board, "cols") else len(state.board.grid[0])
    for r in range(rows):
        for c in range(cols):
            for dr, dc in directions:
                line = [(r + dr*i, c + dc*i) for i in range(5)]
                if not all(0 <= rr < rows and 0 <= cc < cols for rr, cc in line):
                    continue
                count = 0
                empty = None
                valid = True
                for pos in line:
                    if state.board.is_corner(pos):
                        count += 1  # corner counts as player chip
                    else:
                        cell = state.board.get(pos)
                        if cell == player_id:
                            count += 1
                        elif cell == opp_id:
                            valid = False
                            break
                        else:
                            if empty is None:
                                empty = pos
                            else:
                                # more than one empty
                                valid = False
                                break
                if valid and count == 4 and empty:
                    return empty
    return None
