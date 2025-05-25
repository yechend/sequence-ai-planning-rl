from template import Agent
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
Q_TABLE_FILE = "./agents/t_000/q_learning_weights.json"

class myAgent(Agent):
    def __init__(self, _id):
        super().__init__(_id)
        self.id = _id
        self.rule = GameRule(2)
        self.q_table = {}
        self.load_q_table()
        self.last_state = None
        self.last_action = None

    def SelectAction(self, actions, game_state):
        # --- Q-value Update from Last Action ---
        if self.last_state is not None and self.last_action is not None:
            reward = self.get_reward(game_state)
            self.update(self.last_state, self.last_action, reward, game_state)

        # --- Check for Immediate Win ---
        win_action = self.find_winning_move(actions, game_state, self.id)
        if win_action:
            selected_action = win_action
        else:
            selected_action = self.choose_action(actions, game_state)

        # --- Track State/Action for Future Updates ---
        self.last_state = self.copy_state(game_state)
        self.last_action = selected_action

        return selected_action

    def choose_action(self, actions, game_state):
        state_rep = self.get_state_representation(game_state)

        if random.random() < EPSILON:
            return random.choice(actions)
        else:
            q_values = [self.get_q_value(state_rep, action) for action in actions]
            max_q = max(q_values)
            best_actions = [a for a, q in zip(actions, q_values) if q == max_q]
            return random.choice(best_actions)

    def get_state_representation(self, state):
        return tuple(tuple(row) for row in state.board.chips)

    def get_q_value(self, state_rep, action):
        return self.q_table.get((state_rep, str(action)), 0.0)

    def update(self, prev_state, action, reward, next_state):
        if not LEARNING:
            return

        prev_rep = self.get_state_representation(prev_state) # old state
        next_rep = self.get_state_representation(next_state) # new state
        prev_q = self.get_q_value(prev_rep, action)

        next_actions = self.rule.getLegalActions(next_state, self.id)
        future_q = max(
            [self.get_q_value(next_rep, a) for a in next_actions] or [0.0]
        )

        new_q = (1 - ALPHA) * prev_q + ALPHA * (reward + GAMMA * future_q)
        self.q_table[(prev_rep, str(action))] = new_q

        self.save_q_table()

    def save_q_table(self):
        to_save = {}
        for (state_rep, action_str), value in self.q_table.items():
            key = state_rep + "||" + action_str
            to_save[key] = value

        with open(Q_TABLE_FILE, 'w') as f:
            json.dump(to_save, f, indent=2)

    def load_q_table(self):
        if os.path.exists(Q_TABLE_FILE):
            with open(Q_TABLE_FILE, 'r') as f:
                loaded = json.load(f)
            self.q_table = {}
            for key, value in loaded.items():
                state_rep, action_str = key.split("||", 1)
                self.q_table[(state_rep, action_str)] = value
        else:
            with open(Q_TABLE_FILE, 'w') as f:
                json.dump({}, f)
            self.q_table = {}

    def get_reward(self, state):
        # IMPROVE
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

        result, seq_type = self.rule.checkSeq(board, agent, (r, c))
        if result and result['num_seq'] >= 1:
            return True

        return False
