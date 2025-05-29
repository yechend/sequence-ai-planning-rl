from template import Agent
from Sequence.sequence_model import SequenceGameRule as GameRule
from Sequence.sequence_model import COORDS
from Sequence.sequence_utils import EMPTY  # constant for empty board cell
import time, random
from math import sqrt, log
from copy import copy

# Constants for timing
MAX_THINK_TIME = 0.95 # seconds allotted per move (with buffer)

# Center (heart) coordinates for special win condition
CENTER_COORDS = [(4,4), (4,5), (5,4), (5,5)]

class myAgent(Agent):
    def __init__(self, _id):
        super().__init__(_id)
        self.id = _id
        self.rule = GameRule(2)  
        super().__init__(_id)  
        self.root = None

    # check if a given action results in a win for the given agent_id
    def isWinningMove(self, state, action, agent_id):
        """Simulate the action on a copy of board and check if it completes a sequence."""
        if action['type'] != 'place' or action['coords'] is None:
            return False
        r, c = action['coords']
        board = [row[:] for row in state.board.chips]  # shallow copy of board matrix
        board[r][c] = state.agents[agent_id].colour
        if all(board[x][y] == state.agents[agent_id].colour for (x,y) in CENTER_COORDS):
            return True
        result, _ = self.rule.checkSeq(board, state.agents[agent_id], (r, c))
        return result is not None and result['num_seq'] >= 1

    # identify if a card is "dead" (no available place on board for that card)
    def is_dead_card(self, card, board):
        """Return True if all board positions for this card are occupied."""
        if card in ['jd','jc','jh','js']:  # Jacks are never dead (wild or remove)
            return False
        for (r, c) in COORDS.get(card, []):
            if board[r][c] == EMPTY:
                return False
        return True

    # MCTS Node structure as an inner class for clarity
    class Node:
        __slots__ = ('parent', 'action', 'player', 'board', 'hand', 'draft', 'visits', 'wins', 'children', 'untried_actions', 'heuristic_cache')
        def __init__(self, parent, action, player, board, hand, draft):
            self.parent = parent       # parent Node
            self.action = action       # action leading to this node (None for root)
            self.player = player       # player ID who will move at this node (current turn)
            self.board = board         # board state (10x10 matrix of chips) at this node
            self.hand = hand           # hand of our agent at this node (if this.player == our_id)
            self.draft = draft         # draft cards available at this node
            self.visits = 0
            self.wins = 0.0           # total win score (from root's perspective)
            self.children = []        # list of child Node
            self.untried_actions = None  # list of actions yet to try from this node
            self.heuristic_cache = None   # store heuristic values for possible moves (for opponent moves)

    def SelectAction(self, actions, game_state):
        """Select an action based on MCTS with heuristic-guided search."""
        start_time = time.perf_counter()
        # 1. Immediate win check – if any available action wins the game, play it.
        for action in actions:
            if self.isWinningMove(game_state, action, self.id):
                return action  # take the winning move immediately

        # 2. Initialize root node for MCTS
        root = self._advance_root(game_state, actions)  # try to advance existing root if possible
        root_hand = game_state.agents[self.id].hand.copy()   # our agent's hand
        root_draft = game_state.board.draft.copy()           # current draft cards
        if root is not None:
            root.draft = root_draft
            root.hand = root_hand 
            self._prune_invalid_subtree(root, root_hand, root_draft)  # prune invalid actions
            self._refresh_untried_actions(root, actions)  # refresh untried actions based on current state
            for child in root.children:    
                act = child.action
                if act is not None and (act['draft_card'] not in root_draft or act['play_card'] not in root_hand):
                    root.children.remove(child) 
                
        else:
         # Copy current board state
            root_board = [row[:] for row in game_state.board.chips]
            
            root = self.Node(parent=None, action=None, player=self.id, 
                         board=root_board, hand=root_hand, draft=root_draft)
            # Prepare list of legal actions at root (our turn)
            root.untried_actions = [a for a in actions if a['type'] in ['place','discard']]
            # Sort our actions by heuristic value (descending) for prioritized expansion
            root.untried_actions.sort(key=lambda a: self.HeuristicBoard(root_board, a.get('coords'), game_state, self.id), reverse=True)
            self.root = root


        # Precompute full deck and initial seen cards for draw simulation
        plan_draft = None
        draft_cards = game_state.board.draft
        for draft in draft_cards:
            if draft in ['jd', 'jc','jh','js']:
                plan_draft = draft

        full_deck  = [
        '2c', '2d', '2h', '2s', '3c', '3d', '3h', '3s', '4c', '4d', '4h', '4s',
        '5c', '5d', '5h', '5s', '6c', '6d', '6h', '6s', '7c', '7d', '7h', '7s',
        '8c', '8d', '8h', '8s', '9c', '9d', '9h', '9s', 'tc', 'td', 'th', 'ts',
        'jc', 'jd', 'jh', 'js', 'qc', 'qd', 'qh', 'qs', 'kc', 'kd', 'kh', 'ks',
        'ac', 'ad', 'ah', 'as',
        '2c', '2d', '2h', '2s', '3c', '3d', '3h', '3s', '4c', '4d', '4h', '4s',
        '5c', '5d', '5h', '5s', '6c', '6d', '6h', '6s', '7c', '7d', '7h', '7s',
        '8c', '8d', '8h', '8s', '9c', '9d', '9h', '9s', 'tc', 'td', 'th', 'ts',
        'jc', 'jd', 'jh', 'js', 'qc', 'qd', 'qh', 'qs', 'kc', 'kd', 'kh', 'ks',
        'ac', 'ad', 'ah', 'as'
        ]

        seen_cards = set(root_hand + root_draft)
        # treat unknown cards as still in deck
        setup_available_cards = [card for card in full_deck if card not in seen_cards]
        # MCTS
        iteration = 0
        while time.perf_counter() - start_time < MAX_THINK_TIME:
            iteration += 1
            node = root
            available_cards = setup_available_cards.copy()  # fresh copy for each iteration
            # **Selection**: traverse tree to a leaf node
            # Select child nodes until a node with untried actions or terminal state is reached
            while node.untried_actions is None or len(node.untried_actions) == 0:
                # If this node has no children (terminal state) or no untried actions, break
                if not node.children:
                    break
                # UCB selection among children
                # Different players use the same formula but wins are stored from root perspective
                best_child = None
                best_ucb = -float('inf')
                for child in node.children:
                    # Calculate UCB value for child
                    if child.visits == 0:
                        ucb = float('inf')
                    else:
                        mean_value = child.wins / child.visits
                        exploration = sqrt(log(node.visits) / child.visits)
                        # Progressive bias: add heuristic term if available
                        bias = 0.0
                        if child.action is not None:
                            # Determine heuristic bias for the move
                            if node.player == self.id:
                                # Node.player is the one who made the move leading to child
                                # If node.player == self.id, then child was result of our move
                                # Use heuristic of that move (already sorted by it) as bias
                                bias_score = self.HeuristicBoard(node.board, child.action.get('coords'), game_state, node.player)
                            else:
                                # If opponent's move, use opponent's heuristic for that move
                                bias_score = self.HeuristicBoard(node.board, child.action.get('coords'), game_state, node.player)
                            bias = bias_score / (1 + child.visits)
                        ucb = mean_value + 1.414 * exploration + 0.1 * bias  # bias weight 0.1
                    if ucb > best_ucb:
                        best_ucb = ucb
                        best_child = child
                if best_child is None:
                    break
                node = best_child  # move to selected child
                # Check for terminal state 
                if node.action is not None and node.parent is not None:
                    # If the move leading to this node resulted in a win, break
                    if self._is_terminal(node, game_state):
                        break

            # **Expansion**: expand one new child from the node (if not terminal and actions remain)
            if node.untried_actions and len(node.untried_actions) > 0:
                # Take the next untried action (heuristic-prioritized)
                action = node.untried_actions.pop(0)
                # Simulate this action to create a new child node state
                child_node = self._simulate_action(node, action, game_state, available_cards)
                node.children.append(child_node)
                node = child_node  # advance to the newly expanded node

            # if the node is not terminal, perform a rollout from this state
            outcome = self._rollout(node, game_state, available_cards)

            # Backpropagation
            while node is not None:
                node.visits += 1
                # Add outcome to wins from the perspective of root agent (self.id)
                # outcome is 1 for our win, 0 for loss (0.5 or intermediate values for draw/heuristic eval)
                if node.parent is None:
                    # Root node: no move made here
                    node.wins += 0  # root.wins is unused effectively
                else:
                    # Determine reward for root: if our agent is root (self.id), outcome already in root perspective
                    node.wins += outcome
                node = node.parent

        # Select the best move from root (most visits or highest win rate)
        best_move = None
        best_score = -float('inf')
        for child in root.children:
            # We can choose by highest visit count or highest average win rate
            # Here use highest visits (robust child) as final decision, with win-rate as tiebreaker
            score = child.visits
            avg_win = child.wins / child.visits if child.visits > 0 else 0
            if score > best_score or (score == best_score and avg_win > (0 if best_move is None else best_move.wins/best_move.visits)):
                best_score = score
                best_move = child
        # If no children (should not happen unless no actions), choose a random legal action
        if best_move is None:
            self.root = None
            return random.choice(actions)
        self.root = best_move
        self.root.parent = None 
        
        if plan_draft is not None:
            best_move.action['draft_card'] = plan_draft
        print("selct", best_move.action)
        print(game_state.board.draft)
        return best_move.action

    def _prune_invalid_subtree(self, node, curr_hand, curr_draft):
        # Prune the subtree rooted at `node` by removing children with invalid tree.
        valid_children = []
        for child in node.children:
            act = child.action
            if act is None:
                self._prune_invalid_subtree(child, curr_hand, curr_draft)
                valid_children.append(child)
            elif (act['play_card'] in curr_hand) and (act['draft_card'] in curr_draft):
                self._prune_invalid_subtree(child, curr_hand, curr_draft)
                valid_children.append(child)
        node.children = valid_children


    def _advance_root(self, game_state, legal_actions):
        # Attempt to advance the root node based on current game state and legal actions.
        if self.root is None or not self.root.children:
            return None

        if self.root.board == game_state.board.chips:
            self._refresh_untried_actions(self.root, legal_actions)
            return self.root

        for child in self.root.children:
            if child.board == game_state.board.chips:       # only check board state
                child.parent = None
                self._refresh_untried_actions(child, legal_actions)

                return child
        return None

    

    def _refresh_untried_actions(self, node, actions): 
        # Update the node's untried_actions based on current legal actions.
        tried_actions = [child.action for child in node.children]  
        node.untried_actions = [a for a in actions if a['type'] in ['place','discard'] and a not in tried_actions] 
        # can dev the re-sort by heuristic if desired


    def _is_terminal(self, node, state):
        #Check if the last move leading to node resulted in a terminal win condition.
        # If no action (root) or no parent, not terminal by definition here
        if node.action is None or node.parent is None:
            return False
        prev_player = node.parent.player  # the player who made the move
        if node.action['type'] == 'place':
            r, c = node.action['coords']
            if all(node.parent.board[x][y] == state.agents[prev_player].colour for (x,y) in CENTER_COORDS):
                return True
            result, _ = self.rule.checkSeq(node.parent.board, state.agents[prev_player], (r, c))
            if result is not None and result['num_seq'] >= 1:
                return True
        return False

    def _simulate_action(self, node, action, state, available_cards):
        """Apply an action to the given node's state and return the new child node."""
        # Determine next player (alternate turn)
        next_player = 1 - node.player
        # Copy the board and other state components from current node
        new_board = [row[:] for row in node.board]
        new_hand = node.hand.copy() if node.hand is not None else None
        new_draft = node.draft.copy()
        # Apply the action on new_board and update hands/draft
        if action['type'] == 'place':
            (r, c) = action['coords']
            # Place chip of current player on board
            new_board[r][c] = state.agents[node.player].colour

        # Update hands and draft for both place and discard similarly:
        if node.player == self.id:
            # Our agent's turn
            if action['play_card'] in new_hand:
                new_hand.remove(action['play_card'])
            drawn_card = action['draft_card']
            new_hand.append(drawn_card)
        # For opponent, we don't explicitly track their full hand. So we assume they had the play_card and used it.
        # Remove play_card from available deck
        if node.player != self.id and action['type'] == 'place' and action['play_card'] is not None:
            if action['play_card'] in available_cards:
                available_cards.remove(action['play_card'])
        if action['draft_card'] in new_draft:
            # Remove the taken draft card from pool
            new_draft.remove(action['draft_card'])
        # Draw a replacement from deck for the draft pool
        if available_cards:
            new_card = random.choice(available_cards)
            available_cards.remove(new_card)  # remove from deck
            new_draft.append(new_card)
        next_hand = new_hand
        # Create the new child node with updated state
        if next_player == self.id:
            child_node = self.Node(parent=node, action=action, player=next_player,
                               board=new_board, hand=next_hand, draft=new_draft)
            child_actions = self.GenerateActionsForHand(child_node.board, next_hand, new_draft, state)
            # Prioritize by heuristic
            child_actions.sort(key=lambda a: self.HeuristicBoard(child_node.board, a.get('coords'), state, self.id), reverse=True)
            child_node.untried_actions = child_actions
        else:
            child_node = self.Node(parent=node, action=action, player=next_player,
                               board=new_board, hand=next_hand, draft=new_draft)    #still save our hand card, but do not use it if it is opponent's turn
            # Opponent
            child_node.untried_actions = self._opponent_actions(new_board, state)
        return child_node

    def _opponent_actions(self, board, state):
        acts = []
        for r in range(10):
            for c in range(10):
                if board[r][c] == EMPTY:
                    acts.append({'type': 'place',
                                'coords': (r, c),
                                'play_card': None,  
                                'draft_card': None})
        acts.sort(key=lambda a: self.HeuristicBoard(board, a['coords'], state, opp_id),
                reverse=True)
        return acts

    def _rollout(self, node, state, available_cards):
        """Perform a simulated playout (with depth limit) from the given node, returning outcome for root (self.id)."""
        MAX_DEPTH = 6  # limit rollout length to 6 moves (3 per player) for efficiency
        current_player = node.player
        # Make fresh copies of state to simulate on
        sim_board = [row[:] for row in node.board]
        # Copy our agent's hand if known, and a placeholder for opponent's hand size
        sim_hand = node.hand.copy() if current_player == self.id and node.hand is not None else (node.hand.copy() if node.hand is not None else [])
        sim_draft = node.draft.copy()
        depth = 0
        outcome = None
        # Simple structures to track available deck for simulation (copy current available_cards list)
        sim_deck = available_cards.copy()
        # Rollout loop
        while depth < MAX_DEPTH:
            # Check if the last move (by previous player) ended the game
            if node.parent is not None and self._is_terminal(node, state):
                outcome = 1.0 if node.parent.player == self.id else 0.0
                break
            # Determine current player's action using heuristic policy
            if current_player == self.id:
                # Our agent's turn in simulation: choose best heuristic move from available actions
                possible_actions = self.GenerateActionsForHand(sim_board, sim_hand, sim_draft, state)
                # If no possible action, break
                if not possible_actions:
                    break
                # Pick the action with highest heuristic score
                possible_actions.sort(key=lambda a: self.HeuristicBoard(sim_board, a.get('coords'), state, self.id), reverse=True)
                sim_action = possible_actions[0]
            else:
                # Generate a plausible move for opponent. We'll maximizes their heuristic.
                best_score = -float('inf')
                best_move_coords = None
                # Evaluate all empty positions
                for r in range(10):
                    for c in range(10):
                        if sim_board[r][c] == EMPTY:
                            score = self.HeuristicBoard(sim_board, (r, c), state, current_player)
                            if score > best_score:
                                best_score = score
                                best_move_coords = (r, c)
                if best_move_coords is None:
                    # No empty spot – end rollout
                    break
                # Construct a hypothetical opponent action for placement
                play_card = None
                for card, coords_list in COORDS.items():
                    if best_move_coords in coords_list:
                        play_card = card
                        break
                # If no specific card found, just use a wild card
                if play_card is None:
                    play_card = 'jd'
                # Choose a random draft card (opponent likely picks randomly or any, we assume random. can be improved)
                draft_choice = random.choice(sim_draft) if sim_draft else None
                sim_action = {'type': 'place', 'coords': best_move_coords, 'play_card': play_card, 'draft_card': draft_choice}
            # Apply the chosen sim_action to sim_board and update sim_hand/sim_draft similarly to _simulate_action
            if sim_action['type'] == 'place' and sim_action['coords']:
                r, c = sim_action['coords']
                sim_board[r][c] = state.agents[current_player].colour
            # Update hands and draft
            if current_player == self.id:
                if sim_action['play_card'] in sim_hand:
                    sim_hand.remove(sim_action['play_card'])
                if sim_action['draft_card']:
                    sim_hand.append(sim_action['draft_card'])
            else:
                if sim_action['play_card'] and sim_action['play_card'] in sim_deck:
                    sim_deck.remove(sim_action['play_card'])
            if sim_action.get('draft_card') in sim_draft:
                sim_draft.remove(sim_action['draft_card'])
            if sim_deck:
                new_card = random.choice(sim_deck)
                sim_deck.remove(new_card)
                sim_draft.append(new_card)
            # Check if this move ends the game
            if sim_action['type'] == 'place' and sim_action['coords']:
                if all(sim_board[x][y] == state.agents[current_player].colour for (x,y) in CENTER_COORDS):
                    outcome = 1.0 if current_player == self.id else 0.0
                    break
                res, _ = self.rule.checkSeq(sim_board, state.agents[current_player], sim_action['coords'])
                if res is not None and res['num_seq'] >= 1:
                    outcome = 1.0 if current_player == self.id else 0.0
                    break
            # Switch player turn
            current_player = 1 - current_player
            depth += 1

        # If we exited without setting outcome, evaluate heuristic final state
        if outcome is None:
            my_score = 0
            opp_score = 0
            #estimate h by scanning board for our advantage vs opponent.
            for r in range(10):
                for c in range(10):
                    if sim_board[r][c] == EMPTY:
                        # evaluate this empty spot for both players
                        my_score += self.HeuristicBoard(sim_board, (r, c), state, self.id)
                        opp_score += self.HeuristicBoard(sim_board, (r, c), state, 1 - self.id)
            # Normalize between 0 and 1
            total = my_score + opp_score
            outcome = 0.5
            if total != 0:
                outcome = my_score / total
        return outcome

    def GenerateActionsForHand(self, board, hand, draft, state):
        """Generate all valid actions (place or discard) for our agent given current board, hand, and draft."""
        actions = []
        # Compute all place actions
        card_targets = set()
        for card in hand:
            if card not in ['jd', 'jc']:
                for (r,c) in COORDS.get(card, []):
                    if board[r][c] == EMPTY:
                        card_targets.add((r,c))
        for card in hand:
            if card in ['jd','jc']:
                # Two-eyed jack can place on any empty spot not already reachable by a normal card in hand (to avoid duplicates)
                jack_moves = [(r,c) for r in range(10) for c in range(10) if board[r][c] == EMPTY and (r,c) not in card_targets]
                # Sort jack moves by proximity to center (as a heuristic for better positions)
                jack_moves.sort(key=lambda pos: abs(pos[0]-4.5) + abs(pos[1]-4.5))
                for (r,c) in jack_moves[:12]:  # limit jack moves to 12 best to reduce branching
                    for draft_card in draft:
                        actions.append({'type': 'place', 'coords': (r,c), 'play_card': card, 'draft_card': draft_card})
            elif card not in ['jh','js']:
                # Normal card: each unique empty position for that card
                added = set()
                for (r,c) in COORDS.get(card, []):
                    if board[r][c] == EMPTY and (r,c) not in added:
                        added.add((r,c))
                        for draft_card in draft:
                            actions.append({'type': 'place', 'coords': (r,c), 'play_card': card, 'draft_card': draft_card})
        # Compute possible discard actions:
        # If any dead card in hand, we can discard one (the first dead) for any draft card
        dead_cards = [card for card in hand if self.is_dead_card(card, board)]
        if dead_cards:
            dead_card = dead_cards[0]
            for draft_card in draft:
                actions.append({'type': 'discard', 'play_card': dead_card, 'draft_card': draft_card})
        else:
            min_score = float('inf'); worst_card = None
            for card in hand:
                if card in ['jd','jc','jh','js']:
                    continue
                max_val = -1
                for (r,c) in COORDS.get(card, []):
                    if board[r][c] == EMPTY:
                        val = self.HeuristicBoard(board, (r,c), state, self.id)
                        if val > max_val:
                            max_val = val
                if max_val < min_score:
                    min_score = max_val
                    worst_card = card
            if worst_card and min_score < 10:  # if a card is particularly low-value
                for draft_card in draft:
                    actions.append({'type': 'discard', 'play_card': worst_card, 'draft_card': draft_card})
        return actions

    def HeuristicBoard(self, board, coords, state, agent_id):
        # If no specific coords (e.g., for discard), return 0 heuristic
        if coords is None:
            return 0
        r, c = coords
        colour = state.agents[agent_id].colour
        score = max(0, 5 - abs(r - 4.5) - abs(c - 4.5)) * 1.5  # center proximity bonus
        directions = [(0,1), (1,0), (1,1), (1,-1)]
        fork_dirs = 0
        for (d_r, d_c) in directions:
            aligned, open_ends = self.CountAlignedChips(board, r, c, d_r, d_c, colour)
            # Score patterns: reward longer alignments with open ends
            if aligned >= 5:
                score += 200  
            elif aligned == 4 and open_ends >= 1:
                score += 90
            elif aligned == 3 and open_ends == 2:
                score += 50
            elif aligned == 2 and open_ends == 2:
                score += 20
            else:
                score += 2 
            if aligned >= 3 and open_ends >= 1:
                fork_dirs += 1
        if fork_dirs >= 2:
            score += (fork_dirs - 1) * 10
        return score

    def CountAlignedChips(self, board, row, col, d_row, d_col, player_colour):
        open_ends = 0
        fwd_count = bwd_count = 0
        # forward direction
        for step in range(1, 5):
            r, c = row + d_row * step, col + d_col * step
            if r < 0 or r > 9 or c < 0 or c > 9:
                break
            chip = board[r][c]
            if chip == player_colour:
                fwd_count += 1
            elif chip == EMPTY:
                open_ends += 1
                break
            else:
                break
        # backward direction
        for step in range(1, 5):
            r, c = row - d_row * step, col - d_col * step
            if r < 0 or r > 9 or c < 0 or c > 9:
                break
            chip = board[r][c]
            if chip == player_colour:
                bwd_count += 1
            elif chip == EMPTY:
                open_ends += 1
                break
            else:
                break

        return 1 + fwd_count + bwd_count, open_ends


