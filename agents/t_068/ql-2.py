import random
import json
import os
from Sequence.sequence_model import SequenceGameRule as GameRule
from copy import deepcopy

# --- Constants ---
ALPHA = 0.5
GAMMA = 0.9
EPSILON = 0.1
LEARNING = False
WEIGHTS_FILE = "./agents/t_000/req1_sequence_weights.json"

# --- Feature names ---
FEATURE_NAMES = [
    "own_chips_connected",
    "opp_chips_connected",
    "distance_to_center",
    "create_new_sequence",
    "block_opp_sequences",
    "own_total_sequences",
    "opp_total_sequences",
    "available_moves",
    "adjacent_to_joker"
]
class myAgent(Agent):
    def __init__(self, _id):
        super().__init__(_id)
        self.id = _id
        self.rule = GameRule(2)
        self.weights = {}  # feature_name -> weight
        self.load_weights()
        self.last_state = None
        self.last_action = None

    def SelectAction(self, actions, game_state):
        # --- TD update from previous state-action ---
        if self.last_state is not None and self.last_action is not None:
            reward = self.get_reward(game_state)
            self.update(self.last_state, self.last_action, reward, game_state)

        # --- Check immediate win ---
        win_action = self.find_winning_move(actions, game_state, self.id)
        if win_action:
            selected_action = win_action
        else:
            selected_action = self.choose_action(actions, game_state)

        # --- Store current state/action for next step's update ---
        self.last_state = self.copy_state(game_state)
        self.last_action = selected_action

        return selected_action

    def update(self, prev_state, action, reward, next_state):
        if not LEARNING:
            return

        # --- Feature extraction ---
        prev_features = self.extract_features(prev_state, action, self.id)
        prediction = sum(self.weights.get(f, 0.0) * prev_features[f] for f in FEATURE_NAMES)

        # --- Estimate target using next state's best action ---
        next_actions = self.rule.getLegalActions(next_state, self.id)
        if not next_actions:
            target = reward
        else:
            next_scores = []
            for next_action in next_actions:
                if next_action is None:
                    continue
                features = self.extract_features(next_state, next_action, self.id)
                score = sum(self.weights.get(f, 0.0) * features[f] for f in FEATURE_NAMES)
                next_scores.append(score)

            target = reward + GAMMA * max(next_scores) if next_scores else reward

        # --- Compute and clip TD error ---
        error = ALPHA*(target - prediction)
        error = max(-5.0, min(5.0, error))

        # --- Gradient descent weight update with clipping ---
        for f in FEATURE_NAMES:
            self.weights[f] = self.weights.get(f, 0.0) + ALPHA * error * prev_features.get(f, 0.0)
            self.weights[f] = max(-10.0, min(10.0, self.weights[f]))

        self.save_weights()

    def extract_features(self, state, action, agent_id):
        features = {f: 0.0 for f in FEATURE_NAMES}

        if action is None:
            return features

        from copy import deepcopy
        board = deepcopy(state.board.chips)
        agent = state.agents[agent_id]
        opp_id = 1 - agent_id
        opponent = state.agents[opp_id]

        if action['type'] == 'place':
            r, c = action['coords']
            board[r][c] = agent.colour

            directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

            # 1. own_chips_connected
            own_conn = 0
            for dx, dy in directions:
                for step in range(1, 5):
                    nx, ny = r + dx * step, c + dy * step
                    if 0 <= nx < 10 and 0 <= ny < 10 and board[nx][ny] == agent.colour:
                        own_conn += 1
                    else:
                        break
            features["own_chips_connected"] = own_conn / 4.0

            # 2. opp_chips_connected
            opp_conn = 0
            for dx, dy in directions:
                for step in range(1, 5):
                    nx, ny = r + dx * step, c + dy * step
                    if 0 <= nx < 10 and 0 <= ny < 10 and board[nx][ny] == opponent.colour:
                        opp_conn += 1
                    else:
                        break
            features["opp_chips_connected"] = opp_conn / 4.0

            # 3. distance_to_center
            features["distance_to_center"] = (abs(r - 4.5) + abs(c - 4.5)) / 9.0

            # 4. create_new_sequence
            result, _ = self.rule.checkSeq(board, agent, (r, c))
            features["create_new_sequence"] = 1.0 if result and result.get("num_seq", 0) > 0 else 0.0

            # 5. block_opp_sequence
            if opp_conn >= 2:
                features["block_opp_sequence"] = 1.0

            # 6. own_total_sequences
            features["own_total_sequences"] = agent.completed_seqs / 2.0

            # 7. opp_total_sequences
            features["opp_total_sequences"] = opponent.completed_seqs / 2.0

            # 8. available_moves
            features["available_moves"] = len(state.board.empty_coords) / 100.0  # normalised

            # 9. adjacent_to_joker
            jokers = [(0, 0), (0, 9), (9, 0), (9, 9)]
            if action['type'] == 'place':
                r, c = action['coords']
                for jr, jc in jokers:
                    if abs(r - jr) + abs(c - jc) == 1:
                        features["adjacent_to_joker"] = 1.0
                        break

        return features

    def choose_action(self, actions, game_state):
        if random.random() < EPSILON:
            return random.choice(actions)
        else:
            scores = []
            for action in actions:
                features = self.extract_features(game_state, action, self.id)
                score = sum(self.weights.get(f, 0.0) * features[f] for f in features)
                scores.append(score)

            max_score = max(scores)
            best_actions = [a for a, s in zip(actions, scores) if s == max_score]
            return random.choice(best_actions)
    def load_weights(self):
        if os.path.exists(WEIGHTS_FILE):
            with open(WEIGHTS_FILE, 'r') as f:
                self.weights = json.load(f)
        else:
            self.weights = {f: 0.0 for f in FEATURE_NAMES}
            self.save_weights()

    def save_weights(self):
        with open(WEIGHTS_FILE, 'w') as f:
            json.dump(self.weights, f)

    def get_reward(self, state):
        return state.agents[self.id].completed_seqs

    def copy_state(self, state):
        from copy import deepcopy
        return deepcopy(state)

    def check_winning_move(self, state, action, agent_id):
        from copy import deepcopy
        temp_state = deepcopy(state)
        agent = temp_state.agents[agent_id]
        board = temp_state.board.chips

        if action['type'] != 'place':
            return False

        r, c = action['coords']
        board[r][c] = agent.colour

        result, _ = self.rule.checkSeq(board, agent, (r, c))
        return bool(result and result.get("num_seq", 0) >= 1)
