# pattern_q_agent.py - Q-learning with tactical pattern features
from template import Agent
from Sequence.sequence_model import SequenceGameRule as GameRule, COORDS
from Sequence.sequence_utils import *
from copy import deepcopy
import random, json, os, time

ALPHA = 0.2
GAMMA = 0.9
EPSILON = 0.1
LEARNING = False
WEIGHT_FILE = "./agents/ai_agent/weights1.json"
THINK_TIME_LIMIT = 0.95

FEATURES = [
    "live_one", "sleep_two", "live_two", "sleep_three", "live_three",
    "chong_four", "live_four", "live_five",
    "opp_live_three", "opp_live_four", "opp_live_five"
]

INIT_WEIGHTS = {f: 0.0 for f in FEATURES}

class myAgent(Agent):
    def __init__(self, _id):
        super().__init__(_id)
        self.id = _id
        self.rule = GameRule(2)
        self.weights = INIT_WEIGHTS.copy()
        self.last_state = None
        self.last_action = None
        self.load_weights()

    def SelectAction(self, actions, state):
        start = time.time()

        if self.last_state and self.last_action:
            reward = state.agents[self.id].completed_seqs
            self.update(self.last_state, self.last_action, reward, state)

        best_action = self.choose_action(actions, state, start)
        self.last_state = deepcopy(state)
        self.last_action = best_action
        return best_action

    def choose_action(self, actions, state, start_time):
        if random.random() < EPSILON:
            return random.choice(actions)
        best_score, best_action = float('-inf'), random.choice(actions)
        for a in actions:
            if time.time() - start_time > THINK_TIME_LIMIT:
                break
            q = self.q_value(state, a)
            if q > best_score:
                best_score, best_action = q, a
        return best_action

    def q_value(self, state, action):
        feats = self.extract_features(state, action)
        return sum(self.weights[f] * feats[f] for f in FEATURES)

    def update(self, prev_state, action, reward, next_state):
        if not LEARNING:
            return
        feats = self.extract_features(prev_state, action)
        pred_q = sum(self.weights[f] * feats[f] for f in FEATURES)
        next_qs = [self.q_value(next_state, a) for a in self.rule.getLegalActions(next_state, self.id)]
        max_next_q = max(next_qs) if next_qs else 0.0
        target = reward + GAMMA * max_next_q
        for f in FEATURES:
            self.weights[f] += ALPHA * (target - pred_q) * feats[f]
        self.save_weights()

    def extract_features(self, state, action):
        board = [row[:] for row in state.board.chips]
        me, opp = state.agents[self.id], state.agents[1 - self.id]
        r, c = action['coords']
        if action['type'] == 'place':
            board[r][c] = me.colour
        elif action['type'] == 'remove':
            board[r][c] = EMPTY
        return {
            "live_one": self.count_pattern(board, me.colour, 1, 2),
            "sleep_two": self.count_pattern(board, me.colour, 2, 1),
            "live_two": self.count_pattern(board, me.colour, 2, 2),
            "sleep_three": self.count_pattern(board, me.colour, 3, 1),
            "live_three": self.count_pattern(board, me.colour, 3, 2),
            "chong_four": self.count_pattern(board, me.colour, 4, 1),
            "live_four": self.count_pattern(board, me.colour, 4, 2),
            "live_five": self.count_pattern(board, me.colour, 5, 2),
            "opp_live_three": self.count_pattern(board, opp.colour, 3, 2),
            "opp_live_four": self.count_pattern(board, opp.colour, 4, 2),
            "opp_live_five": self.count_pattern(board, opp.colour, 5, 2)
        }

    def count_pattern(self, board, colour, target_len, open_ends):
        count = 0
        for r in range(10):
            for c in range(10):
                for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                    aligned = 0
                    left_open = right_open = 0
                    for i in range(target_len):
                        nr, nc = r + dr * i, c + dc * i
                        if 0 <= nr < 10 and 0 <= nc < 10 and board[nr][nc] == colour:
                            aligned += 1
                        else:
                            break
                    if aligned == target_len:
                        lr, lc = r - dr, c - dc
                        rr, rc = r + dr * target_len, c + dc * target_len
                        if 0 <= lr < 10 and 0 <= lc < 10 and board[lr][lc] == EMPTY:
                            left_open = 1
                        if 0 <= rr < 10 and 0 <= rc < 10 and board[rr][rc] == EMPTY:
                            right_open = 1
                        if left_open + right_open >= open_ends:
                            count += 1
        return count

    def save_weights(self):
        with open(WEIGHT_FILE, 'w') as f:
            json.dump(self.weights, f)

    def load_weights(self):
        if os.path.exists(WEIGHT_FILE):
            with open(WEIGHT_FILE, 'r') as f:
                self.weights = json.load(f)
        else:
            self.weights = INIT_WEIGHTS.copy()
            self.save_weights()