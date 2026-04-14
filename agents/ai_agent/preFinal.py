from template import Agent
from Sequence.sequence_model import SequenceGameRule as GameRule
import time, random
from copy import deepcopy
from Sequence.sequence_model import COORDS
from Sequence.sequence_utils import EMPTY

MAX_THINK_TIME = 0.95
SAFETY_BUFFER = 0.02
CENTER_COORDS = [(4, 4), (4, 5), (5, 4), (5, 5)]

class myAgent(Agent):
    def __init__(self, _id):
        super().__init__(_id)
        self.id = _id
        self.rule = GameRule(2)

    def isWinningMove(self, state, action, agent_id):
        """Check if placing a chip results in immediate victory"""
        if action['type'] != 'place' or action['coords'] is None:
            return False

        # Simulation
        r, c = action['coords']
        temp_state = deepcopy(state)
        board = temp_state.board.chips
        agent = temp_state.agents[agent_id]
        board[r][c] = state.agents[agent_id].colour

        # Check if occupy the central four squares
        if all(board[x][y] == agent.colour for x, y in CENTER_COORDS):
            return True

        # Check if score a sequence
        result, _ = self.rule.checkSeq(board, agent, (r, c))
        return result and result['num_seq'] >= 1

    def FindImmediateWin(self, actions, state, agent_id):
        """Find a winning move if available."""
        for action in actions:
            if self.isWinningMove(state, action, agent_id):
                return action

        return None

    def SelectAction(self, actions, game_state):
        """Main entry point — pick a winning move or fall back to heuristic-based strategy."""
        winning_move = self.FindImmediateWin(actions, game_state, self.id)

        if winning_move:
            return winning_move

        return self.TwoStepLookaheadSearch(actions, game_state)

    # Final Chosen Method - Two-Step GBFS
    def SimulatedBoard(self, state, action, agent_id):
        """Simulate the board after the given action."""
        board = deepcopy(state.board.chips)

        if action['coords'] is None:
            return board
        r, c = action['coords']
        colour = state.agents[agent_id].colour

        if action['type'] == 'place':
            board[r][c] = colour

        return board

    def CountAlignedChips(self, board, row, col, d_row, d_col, player_colour):
        """Count the number of existing aligned chips (sequence)."""
        open_ends = 0

        # Check forward direction
        fwd_count = 0
        for step in range(1, 5):
            r, c = row + d_row * step, col + d_col * step
            if 0 <= r < 10 and 0 <= c < 10:
                chip = board[r][c]
                if chip == player_colour:
                    fwd_count += 1
                elif chip == EMPTY:
                    open_ends += 1
                    break
                else:
                    break
            else:
                break

        # Check backward direction
        bwd_count = 0
        for step in range(1, 5):
            r, c = row - d_row * step, col - d_col * step
            if 0 <= r < 10 and 0 <= c < 10:
                chip = board[r][c]
                if chip == player_colour:
                    bwd_count += 1
                elif chip == EMPTY:
                    open_ends += 1
                    break
                else:
                    break
            else:
                break

        total_aligned = 1 + fwd_count + bwd_count
        return total_aligned, open_ends

    def HeuristicBoard(self, board, coords, state, agent_id):
        """ Heuristic evaluation of placing a chip at the given coordinate."""
        if coords is None:
            return 0

        r, c = coords
        colour = state.agents[agent_id].colour

        # Reward placements closer to central four squares
        score = max(0, 5 - abs(r - 4.5) - abs(c - 4.5)) * 1.5

        # Evaluate alignment in all four directions
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        for d_row, d_col in directions:
            aligned, open_ends = self.CountAlignedChips(board, r, c, d_row, d_col, colour)

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

        return score

    def GeneratePlacingActions(self, board, hand, draft):
        """Generate all feasible placing actions for the current board state, hand, and draft."""
        actions = []
        fallback_limit = 12
        CENTER_COORDS = [(4, 4), (4, 5), (5, 4), (5, 5)]

        # Step 1: Determine all coordinates playable by regular cards
        card_targets = set()
        for card in hand:
            if card not in ['jd', 'jc']:  # Ignore jacks
                for pos in COORDS.get(card, []):
                    if board[pos[0]][pos[1]] == EMPTY:
                        card_targets.add(pos)

        # Step 2: Build actions
        for card in hand:
            if card in ['jd', 'jc']:
                jack_moves = []

                # First priority: clear central squares
                for pos in CENTER_COORDS:
                    if board[pos[0]][pos[1]] == EMPTY and pos not in card_targets:
                        jack_moves.append(pos)

                # Fallback: Any other available tile, sorted by center proximity
                if not jack_moves:
                    for r in range(10):
                        for c in range(10):
                            if board[r][c] == EMPTY and (r, c) not in card_targets:
                                jack_moves.append((r, c))

                    jack_moves.sort(key=lambda pos: abs(pos[0] - 4.5) + abs(pos[1] - 4.5))

                # Add actions for Double-eye jack
                for (r, c) in jack_moves[:fallback_limit]:
                    for d in draft:
                        actions.append({
                            'type': 'place',
                            'coords': (r, c),
                            'play_card': card,
                            'draft_card': d
                        })

            else:
                # Other normal cards
                added = set()
                for r, c in COORDS.get(card, []):
                    if board[r][c] == EMPTY and (r, c) not in added:
                        added.add((r, c))
                        for d in draft:
                            actions.append({
                                'type': 'place',
                                'coords': (r, c),
                                'play_card': card,
                                'draft_card': d
                            })

        return actions

    def TwoStepLookaheadSearch(self, actions, state):
        """Performs a Two-Step Greedy Best-First Search to select the best action."""
        start_time = time.perf_counter()
        best_score, best_action = float('-inf'), None
        agent_id = self.id
        original_hand = state.agents[agent_id].hand
        original_draft = state.board.draft

        full_deck = [r + s for r in '23456789tjqka' for s in 'dchs'] * 2

        for a1 in actions:
            if time.perf_counter() - start_time > MAX_THINK_TIME - SAFETY_BUFFER:
                return random.choice(actions)

            if a1['type'] != 'place' or a1['coords'] is None:
                continue

            # Step 1: Simulate first move board
            board1 = self.SimulatedBoard(state, a1, agent_id)
            score1 = self.HeuristicBoard(board1, a1['coords'], state, agent_id)

            # Step 2: Simulate hand and draft after first move
            new_hand = original_hand.copy()
            if a1['play_card'] in new_hand:
                new_hand.remove(a1['play_card'])
            new_hand.append(a1['draft_card'])

            new_draft = original_draft.copy()
            if a1['draft_card'] in new_draft:
                new_draft.remove(a1['draft_card'])

            # Simulate a new random draft card
            seen_cards = set(original_hand + original_draft)
            available = [c for c in full_deck if c not in seen_cards]
            if available:
                new_draft.append(random.choice(available))

            # Step 3: Generate second actions based on simulated future
            second_actions = self.GeneratePlacingActions(board1, new_hand, new_draft)

            best_future = 0
            for a2 in second_actions:
                if time.perf_counter() - start_time > MAX_THINK_TIME - SAFETY_BUFFER:
                    break
                score2 = self.HeuristicBoard(board1, a2['coords'], state, agent_id)
                best_future = max(best_future, score2)

            total_score = score1 + best_future
            if total_score > best_score:
                best_score = total_score
                best_action = a1

        return best_action if best_action else random.choice(actions)