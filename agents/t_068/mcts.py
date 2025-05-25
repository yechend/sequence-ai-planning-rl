import math
import time
from copy import deepcopy

class myAgent(Agent):
    def __init__(self, _id):
        super().__init__(_id)
        self.id = _id
        self.rule = GameRule(NUM_PLAYERS)

    def SelectAction(self, actions, game_state):
        for action in actions:
            if self.is_winning_action(game_state, action, self.id):
                return action

        root = Node(game_state, self.id)
        start_time = time.time()
        iterations = 0

        while time.time() - start_time < THINKTIME:
            node = root

            # Selection
            while node.is_fully_expanded() and not node.is_terminal():
                node = node.select_child()

            # Expansion
            if not node.is_terminal():
                node = node.expand()

            # Simulation
            reward = node.rollout()

            # Backpropagation
            node.backpropagate(reward)

            iterations += 1

        # Choose the best move
        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.move

    def is_winning_action(self, state, action, agent_id):
        from copy import deepcopy

        # Simulate the state after taking the action
        temp_state = deepcopy(state)
        agent = temp_state.agents[agent_id]
        board = temp_state.board.chips

        # Only 'place' actions can create sequences
        if action['type'] != 'place':
            return False

        r, c = action['coords']
        board[r][c] = agent.colour

        # Check if a new sequence is formed
        result, seq_type = self.rule.checkSeq(board, agent, (r, c))
        if result and result.get('num_seq', 0) >= 1:
            return True

        return False




class Node:
    def __init__(self, state, player_id, parent=None, move=None):
        self.state = state
        self.player_id = player_id
        self.parent = parent
        self.move = move
        self.children = []
        self.visits = 0
        self.reward = 0

        if not hasattr(state.agents[player_id], 'hand'):
            self.fake_opponent_hand(state, player_id)

        self.untried_moves = GameRule(NUM_PLAYERS).getLegalActions(state, player_id)
        self.rule = GameRule(NUM_PLAYERS)

    def is_terminal(self):
        return self.rule.gameEnds()

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    def expand(self):
        move = self.untried_moves.pop()
        next_state = self.simulate_action(self.state, move, self.player_id)
        next_player = (self.player_id + 1) % NUM_PLAYERS
        child = Node(next_state, next_player, parent=self, move=move)
        self.children.append(child)
        return child

    def select_child(self):
        log_N_parent = math.log(self.visits)
        best_score = -float('inf')
        best_child = None

        for child in self.children:
            if child.visits == 0:
                return child
            exploit = child.reward / child.visits
            explore = math.sqrt(log_N_parent / child.visits)
            score = exploit + 1.41 * explore  # 1.41 ~ sqrt(2), common in UCB
            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def rollout(self):
        rollout_state = self.copy_state(self.state)
        rollout_player = self.player_id
        rollout_rule = GameRule(NUM_PLAYERS)
        steps = 0
        # estiamte the opponent's acions
        while not rollout_rule.gameEnds() and steps < 5:
            actions = rollout_rule.getLegalActions(rollout_state, rollout_player)
            if not actions:
                break
            action = random.choice(actions)
            rollout_state = self.simulate_action(rollout_state, action, rollout_player)
            rollout_player = (rollout_player + 1) % NUM_PLAYERS
            steps += 1

        return self.evaluate(rollout_state)

    def backpropagate(self, reward):
        self.visits += 1
        self.reward += reward
        if self.parent:
            if self.parent.player_id == self.player_id:
                self.parent.backpropagate(reward)
            else:
                self.parent.backpropagate(-reward)

    def simulate_action(self, state, action, agent_id):
        # Manually apply the action; do not use generateSuccessor
        agent = state.agents[agent_id]
        board = state.board.chips

        if action['type'] == 'trade':
            return state  # +记录已经出过的手牌

        r, c = action['coords']
        if action['type'] == 'place':
            board[r][c] = agent.colour
        elif action['type'] == 'remove':
            board[r][c] = ' '  # EMPTY

        return state
    def evaluate(self, state):
        # Simple evaluation: difference in completed sequences (increase)
        my_seq = state.agents[self.parent.player_id].completed_seqs
        opp_seq = state.agents[(self.parent.player_id + 1) % NUM_PLAYERS].completed_seqs
        return my_seq - opp_seq

    def copy_state(self, state):
        from copy import deepcopy
        return deepcopy(state)

    def fake_opponent_hand(self, state, player_id):
        import random

        agent = state.agents[player_id]

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
                if board[r][c] != ' ' and board[r][c] != 'jk':
                    used_cards.append(board[r][c])

        used_cards += state.deck.discards
        used_cards += state.board.draft

        remaining_cards = [card for card in full_deck if card not in used_cards]

        if len(remaining_cards) < 6:
            remaining_cards += state.board.draft

        agent.hand = random.sample(remaining_cards, min(6, len(remaining_cards))) #opponen's hand

