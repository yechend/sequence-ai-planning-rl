from Sequence.sequence_utils import *
from copy import deepcopy
import time, random
from Sequence.sequence_model import SequenceGameRule, COORDS

PLAYERS = 2

class myAgent():
    def __init__(self, _id):
        self.id = _id
        self.game_rule = SequenceGameRule(PLAYERS)
        self.total_time = 0.0
        self.max_total_time = 3.0
        self.timeout_count = 0
        self.max_timeouts = 3
        self.corner_coords = [(0, 0), (0, 9), (9, 0), (9, 9)]
        self.center_coords = [(4, 4), (4, 5), (5, 4), (5, 5)]
        self.card_value_cache = {}

    def register_initial_state(self, state):
        board = state.board.chips
        self_color = state.agents[self.id].colour
        opp_color = state.agents[1 - self.id].colour
        self.card_value_cache = self.precompute_card_values(board, self_color, opp_color)

    def precompute_card_values(self, board, self_color, opp_color):
        card_scores = {}
        for card, positions in COORDS.items():
            score = 0
            for r, c in positions:
                if board[r][c] != EMPTY:
                    continue
                if (r, c) in self.center_coords:
                    score += 50
                if (r, c) in self.corner_coords:
                    score += 30
                if self.near_sequence_check(board, (r, c), self_color, 3):
                    score += 60
                if self.near_sequence_check(board, (r, c), opp_color, 3):
                    score += 80
            card_scores[card] = score
        return card_scores

    def evaluate_card_heuristic(self, card, hand, board, self_color, opp_color):
        if self.card_value_cache and card in self.card_value_cache:
            return self.card_value_cache[card]
        if card in ['jc', 'jd']:
            return self._score_two_eye_joker(board, self_color, opp_color)
        elif card in ['js', 'jh']:
            return self._score_one_eye_joker(board, opp_color)
        return self._score_regular_card(card, hand, board, self_color, opp_color)

    def _score_two_eye_joker(self, board, self_color, opp_color):
        score = 100
        for r in range(10):
            for c in range(10):
                if board[r][c] == EMPTY:
                    if (r, c) in self.center_coords:
                        score += 20
                    if (r, c) in self.corner_coords:
                        score += 30
                    if self.near_sequence_check(board, (r, c), self_color):
                        score += 40
                    if self.near_sequence_check(board, (r, c), opp_color):
                        score += 50
        return score

    def _score_one_eye_joker(self, board, opp_color):
        score = 60
        for r in range(10):
            for c in range(10):
                if board[r][c] == opp_color and self.near_sequence_check(board, (r, c), opp_color, min_len=4):
                    score += 120
        return score

    def _score_regular_card(self, card, hand, board, self_color, opp_color):
        score = hand.count(card) * 20
        for r, c in COORDS[card]:
            if board[r][c] == EMPTY:
                score += self._score_position_features((r, c), board, self_color, opp_color)
        return score

    def _score_position_features(self, pos, board, self_color, opp_color):
        r, c = pos
        score = 0
        if pos in self.corner_coords:
            score += 70
        if pos in self.center_coords:
            score += 60
        if self.near_sequence_check(board, pos, self_color, 3):
            score += 90
        if self.near_sequence_check(board, pos, self_color, 2):
            score += 50
        if self.near_sequence_check(board, pos, opp_color, 4):
            score += 150
        if self.near_sequence_check(board, pos, opp_color, 3):
            score += 70
        score += self.estimate_multi_seq_value(board, pos, self_color)
        return score

    def near_sequence_check(self, board, pos, color, min_len=4):
        r, c = pos
        directions = [(0,1),(1,0),(1,1),(1,-1)]
        for dr, dc in directions:
            count = 0
            for i in range(-4, 5):
                nr, nc = r + dr*i, c + dc*i
                if 0 <= nr < 10 and 0 <= nc < 10:
                    if board[nr][nc] == color or board[nr][nc] == EMPTY:
                        count += 1
                        if count >= min_len:
                            return True
                    else:
                        count = 0
        return False

    def estimate_multi_seq_value(self, board, pos, color):
        r, c = pos
        directions = [(0,1),(1,0),(1,1),(1,-1)]
        value = 0
        for dr, dc in directions:
            count, empty = 0, 0
            for i in range(-4,5):
                nr, nc = r + dr*i, c + dc*i
                if 0 <= nr < 10 and 0 <= nc < 10:
                    if board[nr][nc] == color:
                        count += 1
                    elif board[nr][nc] == EMPTY:
                        empty += 1
                    else:
                        count, empty = 0, 0
                if count + empty >= 4 and count > 0:
                    value += 20 * count
        return value

    def simulate_and_score(self, action, state):
        next_state = deepcopy(state)
        self.apply_action_to_state(next_state, action, self.id)
        return next_state.agents[self.id].score

    def SelectAction(self, actions, rootstate):
        start = time.time()
        if self.timeout_count >= self.max_timeouts or self.total_time >= self.max_total_time:
            return random.choice(actions)

        hand_cards = rootstate.agents[self.id].hand
        available_draft = rootstate.board.draft
        board = rootstate.board.chips
        self_color = rootstate.agents[self.id].colour
        opp_color = rootstate.agents[1 - self.id].colour

        if all(len(self.get_possible_positions(card, board)) == 0 for card in hand_cards):
            trade_actions = [a for a in actions if a['type'] == 'trade']
            return random.choice(trade_actions) if trade_actions else random.choice(actions)

        best_draft = max(available_draft, key=lambda c: self.evaluate_card_heuristic(c, hand_cards, board, self_color, opp_color))
        filtered = [a for a in actions if a['draft_card'] == best_draft] or actions

        chosen = max(filtered, key=lambda a: self.evaluate_card_heuristic(a['draft_card'], hand_cards, board, self_color, opp_color) +
                                   self.simulate_and_score(a, rootstate) * 100)
        self.total_time += time.time() - start
        return chosen

    def get_possible_positions(self, card, board):
        return [(r, c) for r, c in COORDS.get(card, []) if board[r][c] == EMPTY]

    def apply_action_to_state(self, game_state, action, pid):
        game_state.board.new_seq = False
        player = game_state.agents[pid]
        player.last_action = action
        score_gain = 0
        played_card = action['play_card']
        drafted_card = action['draft_card']
        if played_card:
            player.hand.remove(played_card)
            player.discard = played_card
            game_state.deck.discards.append(played_card)
            game_state.board.draft.remove(drafted_card)
            player.hand.append(drafted_card)
        row, col = action['coords']
        if action['type'] == 'place':
            game_state.board.chips[row][col] = player.colour
            game_state.board.empty_coords.remove((row, col))
            game_state.board.plr_coords[player.colour].append((row, col))
        elif action['type'] == 'remove':
            game_state.board.chips[row][col] = EMPTY
            game_state.board.empty_coords.append((row, col))
            game_state.board.plr_coords[player.opp_colour].remove((row, col))
        if action['type'] == 'place':
            result, seq_type = self.game_rule.checkSeq(game_state.board.chips, player, (row, col))
            if result:
                score_gain += result['num_seq']
                game_state.board.new_seq = seq_type
                for seq_coords in result['coords']:
                    for r_, c_ in seq_coords:
                        if game_state.board.chips[r_][c_] != JOKER:
                            game_state.board.chips[r_][c_] = player.seq_colour
                            try:
                                game_state.board.plr_coords[player.colour].remove((r_, c_))
                            except:
                                pass
                player.completed_seqs += result['num_seq']
                player.seq_orientations.extend(result['orientation'])
        player.trade = False
        player.agent_trace.action_reward.append((action, score_gain))
        player.score += score_gain
        return game_state
