import time
import random
from Sequence.sequence_model import SequenceGameRule as GameRule

TIME_LIMIT = 0.95

class myAgent(Agent):
    def __init__(self, _id):
        super().__init__(_id)
        self.id = _id
        self.rule = GameRule(2)

    def SelectAction(self, self, actions, game_state):
        self.start_time = time.time()
        depth_limit = 2

        best_action = random.choice(actions)
        action = self.minimax_decision(game_state, depth_limit)
        if action is not None:
            best_action = action
        return best_action

    def minimax_decision(self, root_state, depth):
        best_action = None
        best_score = float("-inf")
        actions = self.rule.getLegalActions(root_state, self.id)

        for action in actions:
            if time.time() - self.start_time > TIME_LIMIT:
                break

            self.apply_action(root_state, action, self.id)
            score = self.min_value(root_state, depth - 1, float("-inf"), float("inf"))
            self.undo_action(root_state, action, self.id)

            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    def max_value(self, state, depth, alpha, beta):
        if depth == 0 or self.is_terminal(state):
            return self.evaluate(state)

        max_eval = float("-inf")
        actions = self.rule.getLegalActions(state, self.id)

        for action in actions:
            if time.time() - self.start_time > TIME_LIMIT:
                break

            self.apply_action(state, action, self.id)
            eval_score = self.min_value(state, depth - 1, alpha, beta)
            self.undo_action(state, action, self.id)

            max_eval = max(max_eval, eval_score)
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break

        return max_eval

    def min_value(self, state, depth, alpha, beta):
        if depth == 0 or self.is_terminal(state):
            return self.evaluate(state)

        min_eval = float("inf")
        opponent_id = 1 - self.id
        self.ensure_opponent_hand(state, opponent_id)
        actions = self.rule.getLegalActions(state, opponent_id)

        for action in actions:
            if time.time() - self.start_time > TIME_LIMIT:
                break

            self.apply_action(state, action, opponent_id)
            eval_score = self.max_value(state, depth - 1, alpha, beta)
            self.undo_action(state, action, opponent_id)

            min_eval = min(min_eval, eval_score)
            beta = min(beta, eval_score)
            if beta <= alpha:
                break

        return min_eval

    def apply_action(self, state, action, agent_id):
        agent = state.agents[agent_id]
        board = state.board.chips

        if action['type'] == 'trade':
            return

        r, c = action['coords']
        if action['type'] == 'place':
            board[r][c] = agent.colour
        elif action['type'] == 'remove':
            board[r][c] = ' '  # mark as empty

    def undo_action(self, state, action, agent_id):
        agent = state.agents[agent_id]
        board = state.board.chips

        if action['type'] == 'trade':
            return

        r, c = action['coords']
        if action['type'] == 'place':
            board[r][c] = ' '
        elif action['type'] == 'remove':
            board[r][c] = agent.opp_colour  # revert to opponent's colour

    def is_terminal(self, state):
        return self.rule.gameEnds(state)

    def evaluate(self, state):
        my_seq = state.agents[self.id].completed_seqs
        opponent_seq = state.agents[1 - self.id].completed_seqs
        return (my_seq - opponent_seq) * 1000  # strong win margin

    def ensure_opponent_hand(self, state, opponent_id):
        agent = state.agents[opponent_id]
        if not hasattr(agent, "hand") or agent.hand is None:
            full_deck = [
                '2d', '3d', '4d', '5d', '6d', '7d', '8d', '9d', 'td', 'jd', 'qd', 'kd', 'ad',
                '2c', '3c', '4c', '5c', '6c', '7c', '8c', '9c', 'tc', 'jc', 'qc', 'kc', 'ac',
                '2h', '3h', '4h', '5h', '6h', '7h', '8h', '9h', 'th', 'jh', 'qh', 'kh', 'ah',
                '2s', '3s', '4s', '5s', '6s', '7s', '8s', '9s', 'ts', 'js', 'qs', 'ks', 'as'
            ]
            used_cards = []

            board = state.board.chips
            for r in range(10):
                for c in range(10):
                    if board[r][c] not in (' ', 'jk'):
                        used_cards.append(board[r][c])

            used_cards += state.deck.discards
            used_cards += state.board.draft

            remaining_cards = [card for card in full_deck if card not in used_cards]
            if len(remaining_cards) < 6:
                remaining_cards += state.board.draft

            agent.hand = random.sample(remaining_cards, min(6, len(remaining_cards)))
