
from template import Agent
from Sequence.sequence_model import SequenceGameRule as GameRule
import time, random
from copy import deepcopy
from Sequence.sequence_model import COORDS
from Sequence.sequence_utils import EMPTY

MAX_THINK_TIME = 0.97

CENTER_COORDS = [(4, 4), (4, 5), (5, 4), (5, 5)]

class myAgent(Agent):
    def __init__(self, _id):
        super().__init__(_id)
        self.id = _id
        self.rule = GameRule(2)
        self.opp_hand = []
        self.seen_cards_notInDraft = []

        self.shape_score = {
            'LIVE_1'   :     1,
            'BLO_2'  :     20,
            'LIVE_2'   :    300,
            'BLO_3'  :    400,
            'LIVE_3'   :  6000,
            'DUP_4'  :  9000,
            'LIVE_4'   : 10000,
            'LIVE_5'   : 15000,
            'OPP_LIVE_3'  :  1000,
            'OPP_LIVE_4'  : 12000,
            'OPP_LIVE_5'  : 13000
        }

    def _shape_key(self, segment, colour):
        # Classify a 5-cell segment into a shape key (LIVE_3, BLO_3) for the given colour.
        cnt = segment.count(colour)
        if cnt == 0:
            return 'NONE'
        left_blocked  = (segment[0] != EMPTY and segment[0] != colour)
        right_blocked = (segment[-1] != EMPTY and segment[-1] != colour)
        blocked_ends  = left_blocked + right_blocked

        if cnt == 5:
            return 'LIVE_5'   
        if cnt == 4:
            return 'DUP_4' if blocked_ends == 1 else 'LIVE_4'
        if cnt == 3:
            return 'BLO_3' if blocked_ends >= 1 else 'LIVE_3'
        if cnt == 2:
            return 'BLO_2' if blocked_ends >= 1 else 'LIVE_2'
        if cnt == 1:
            return 'LIVE_1'
        return 'NONE'


    def _direction_threat(self, row, col, dr, dc, board, colour):
        # Check if placing at (row,col) in direction (dr,dc) creates a large than 3-in-a-row threat with at least one open end.
        aligned, open_ends = self.CountAlignedChips(
                                 board, row, col, dr, dc, colour)
        return aligned >= 3 and open_ends >= 1

    def HeuristicBoard(self, board, coords, state, agent_id):
        # Compute the move’s heuristic
        if coords is None:          
            return 0

        r, c = coords
        my_col  = state.agents[agent_id].colour
        opp_col = state.agents[1 - agent_id].colour
        score   = 0
        directions = [(0,1), (1,0), (1,1), (1,-1)]
        for d_row, d_col in directions:

            aligned, opens = self.CountAlignedChips(
                                board, r, c, d_row, d_col, my_col)
            if aligned >= 5:
                key_self = 'LIVE_5'
            elif aligned == 4:
                key_self = 'LIVE_4'  if opens == 2 else 'DUP_4'
            elif aligned == 3:
                key_self = 'LIVE_3'  if opens == 2 else 'BLO_3'
            elif aligned == 2:
                key_self = 'LIVE_2'  if opens == 2 else 'BLO_2'
            else:
                key_self = 'LIVE_1'
            score += self.shape_score.get(key_self, 0)

            # opponent
            opp_aligned, opp_opens = self.CountAlignedChips(
                                        board, r, c, d_row, d_col, opp_col)
            if opp_aligned >= 5:
                key_opp = 'LIVE_5'
            elif opp_aligned == 4:
                key_opp = 'LIVE_4' if opp_opens == 2 else 'DUP_4'
            elif opp_aligned == 3:
                key_opp = 'LIVE_3' if opp_opens == 2 else 'BLO_3'
            else:
                key_opp = None
            if key_opp and key_opp.startswith('LIVE_'):
                score += self.shape_score[f'OPP_{key_opp}']
        
        fork_dirs = 0
        for dr, dc in [(0,1), (1,0), (1,1), (1,-1)]:
            if self._direction_threat(r, c, dr, dc, board, my_col):
                fork_dirs += 1
        if fork_dirs >= 2:
            score += (fork_dirs - 1) * 100
        score += max(0, 5 - abs(r - 4.5) - abs(c - 4.5)) * 10
        return score


    def isWinningMove(self, state, action, agent_id):
        # Simulate the action and return True if it generates at least one valid five-in-a-row sequence.
        if action['type'] != 'place' or action['coords'] is None:
            return False
        r, c = action['coords']
        temp_state = deepcopy(state)
        board = temp_state.board.chips
        agent = temp_state.agents[agent_id]
        board[r][c] = state.agents[agent_id].colour
        if all(board[x][y] == agent.colour for x, y in CENTER_COORDS):
            return True
        result, _ = self.rule.checkSeq(board, agent, (r, c))
        return result and result['num_seq'] >= 1

    def FindImmediateWin(self, actions, state, agent_id):
        # Scan all actions and return one that results in an immediate win, or None.
        for action in actions:
            if self.isWinningMove(state, action, agent_id):
                return action
        return None

    def SelectAction(self, actions, game_state):
        start_time = time.perf_counter()
        # update opponent hand by last action
        self.updateOppDraftSeen(game_state, 1 - self.id)
        
        dead_cards = [card for card in game_state.agents[self.id].hand
                      if self.is_dead_card(card, game_state.board.chips,
                      game_state.agents[self.id].colour)]
        
        plan_draft = None
        draft_cards = game_state.board.draft 
        for draft in draft_cards:
            if draft in ['jd', 'jc','jh','js']:
                plan_draft = draft

        # trade dead cards
        if len(dead_cards) >= 1:
            if plan_draft is None:                
                plan_draft = self.eva_draft(game_state, None, self.id)
                
            action = {
                'type'      : 'trade',
                'coords'    : None,       
                'play_card' : dead_cards[0],
                'draft_card': plan_draft    
            }
            action['draft_card'] = plan_draft

            return action

        winning_move = self.FindImmediateWin(actions, game_state, self.id)
        if winning_move:
            return winning_move
        action = self.TwoStepLookaheadSearch(actions, game_state, start_time)

        if plan_draft is not None:
                action['draft_card'] = plan_draft
        self.seen_cards_notInDraft.append(action['draft_card'])
        self.seen_cards_notInDraft.append(action['play_card'])
        
        return action

    def eva_draft(self, game_state, action, id):
        # Evaluate draft card by resulting heuristic score and select the highest-scoring one.
        eva_draft = {}
        for card in game_state.board.draft:
            for (r, c) in COORDS[card]:
                if game_state.board.chips[r][c] == EMPTY:
                    if action is not None and action['type'] == 'place' and action['coords'] == (r, c):
                        board = self.SimulatedBoard(game_state, action, id)
                    board = game_state.board.chips
                    eva_draft[card] = self.HeuristicBoard(board, (r, c), game_state, self.id)
        return max(eva_draft, key=eva_draft.get)
    

    def SimulatedBoard(self, state, action, agent_id):
        # Return a deep-copied board with the given action applied.
        board = deepcopy(state.board.chips)
        if action['type'] == 'place' and action['coords']:
            r, c = action['coords']
            board[r][c] = state.agents[agent_id].colour
        return board

    def CountAlignedChips(self, board, row, col, d_row, d_col, player_colour):
        # Count contiguous chips and open ends from (row,col) along (d_row,d_col) for a colour.
        open_ends = 0
        fwd_count = bwd_count = 0
        for step in range(1, 5):
            r, c = row + d_row * step, col + d_col * step
            if 0 <= r < 10 and 0 <= c < 10:
                chip = board[r][c]
                if chip == player_colour:
                    fwd_count += 1
                elif chip == EMPTY:
                    open_ends += 1; break
                else: break
        for step in range(1, 5):
            r, c = row - d_row * step, col - d_col * step
            if 0 <= r < 10 and 0 <= c < 10:
                chip = board[r][c]
                if chip == player_colour:
                    bwd_count += 1
                elif chip == EMPTY:
                    open_ends += 1; break
                else: break
        return 1 + fwd_count + bwd_count, open_ends

    def is_dead_card(self, card, board, colour):
        # Return True if a non-jack card has no empty placement positions on the board.
        if card in ['jd', 'jc', 'jh', 'js']:
            return False
        positions = COORDS.get(card, [])
        return all(board[r][c] != EMPTY for r, c in positions)


    def updateOppDraftSeen(self, game_state,opp_id):
        # update seen card from opponent's last action
        action = game_state.agents[opp_id].last_action
        draft = action["draft_card"]
        
        self.seen_cards_notInDraft.append(draft)
        self.seen_cards_notInDraft.append(action['play_card'])



    def GeneratePlacingActions(self, board, hand, draft, state):
        # Generate all legal place or discard actions based on current hand and draft pile.
        actions = []
        fallback_limit = 12
        card_targets = set()
        for card in hand:
            if card not in ['jd', 'jc']:
                for pos in COORDS.get(card, []):
                    if board[pos[0]][pos[1]] == EMPTY:
                        card_targets.add(pos)

        for card in hand:
            if card in ['jd', 'jc']:
                jack_moves = [(r, c) for r in range(10) for c in range(10)
                              if board[r][c] == EMPTY and (r, c) not in card_targets]
                jack_moves.sort(key=lambda pos: abs(pos[0] - 4.5) + abs(pos[1] - 4.5))
                for (r, c) in jack_moves[:fallback_limit]:
                    for d in draft:
                        actions.append({'type': 'place', 'coords': (r, c), 'play_card': card, 'draft_card': d})
            else:
                added = set()
                for r, c in COORDS.get(card, []):
                    if board[r][c] == EMPTY and (r, c) not in added:
                        added.add((r, c))
                        for d in draft:
                            actions.append({'type': 'place', 'coords': (r, c), 'play_card': card, 'draft_card': d})

        colour = state.agents[self.id].colour
        dead_cards = [card for card in hand if self.is_dead_card(card, board, colour)]
        if dead_cards:
            for d in draft:
                actions.append({'type': 'discard', 'play_card': dead_cards[0], 'draft_card': d})
        else:
            min_score, worst_card = float('inf'), None
            for card in hand:
                if card in ['jd', 'jc', 'jh', 'js']:
                    continue
                max_card_score = max(
                    self.HeuristicBoard(board, (r, c), state, self.id)
                    for r, c in COORDS.get(card, [])
                    if board[r][c] == EMPTY
                ) if any(board[r][c] == EMPTY for r, c in COORDS.get(card, [])) else -1
                if max_card_score < min_score:
                    min_score = max_card_score
                    worst_card = card
            if min_score < 10 and worst_card:
                for d in draft:
                    actions.append({'type': 'discard', 'play_card': worst_card, 'draft_card': d})

        return actions

    def TwoStepLookaheadSearch(self, actions, state, start_time):
        # Perform a two-step greedy lookahead, scoring each first move plus best reply, and return the best.
        best_score, best_action = float('-inf'), None
        agent_id = self.id
        original_hand = state.agents[agent_id].hand
        original_draft = state.board.draft
        full_deck = [r + s for r in '23456789tjqka' for s in 'dchs'] * 2
        seen = self.seen_cards_notInDraft+original_hand
        for item in full_deck:
            if item in seen:
                full_deck.remove(item)
        
        for a1 in actions:
            if time.perf_counter() - start_time > MAX_THINK_TIME:
                break
            if a1['type'] not in ['place', 'discard','remove']:
                continue
            board1 = self.SimulatedBoard(state, a1, agent_id)
            score1 = self.HeuristicBoard(board1, a1.get('coords'), state, agent_id)
            new_hand = original_hand.copy()
            if a1['play_card'] in new_hand:
                new_hand.remove(a1['play_card'])
            new_hand.append(a1['draft_card'])
            new_draft = original_draft.copy()
            if a1['draft_card'] in new_draft:
                new_draft.remove(a1['draft_card'])
            available = full_deck.copy()
            available = [c for c in available if c not in original_draft]
            if available:
                new_draft.append(random.choice(available))
            second_actions = self.GeneratePlacingActions(board1, new_hand, new_draft, state)
            best_future = 0
            for a2 in second_actions:
                if time.perf_counter() - start_time > MAX_THINK_TIME:
                    break
                score2 = self.HeuristicBoard(board1, a2.get('coords'), state, agent_id)
                best_future = max(best_future, score2)
            total_score = score1 + best_future
            if total_score > best_score:
                best_score = total_score
                best_action = a1
        return best_action if best_action else random.choice(actions)

    