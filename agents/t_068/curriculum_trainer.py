# curriculum_trainer.py
# Curriculum learning with soft-label policy head, trained on 4000 games

import random
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sequence_state import SequenceState
import importlib
from tqdm import tqdm

# === Load external agents dynamically ===
def load_opponent(agent_path):
    module_path, class_name = agent_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    agent_class = getattr(module, class_name)
    return agent_class(1)  # opponent always has id=1

# Define curriculum weights
def weighted_opponent_choice(opponent_list, n_total):
    pool = []
    for path, weight in opponent_list:
        count = int(weight * n_total)
        pool.extend([path] * count)
    while len(pool) < n_total:
        pool.append(random.choice([p for p, _ in opponent_list]))
    random.shuffle(pool)
    return pool

# Self-play logic with soft-label collection
def self_play_game(opponent):
    states, actions = [], []
    state = SequenceState()
    agent_turn = 0

    while not state.game_over():
        legal = state.get_legal_actions()
        if not legal:
            break

        if agent_turn == 0:
            action = random.choice(legal)
        else:
            action = opponent.SelectAction(legal, state.state)
            if action not in legal:
                break

        try:
            legal = state.get_legal_actions()
            if action not in legal:
                break
            state.apply_action(action)
        except Exception:
            break

        states.append(state.encode())
        actions.append(state.action_to_index(action))
        agent_turn = 1 - agent_turn

    winner = state.get_winner()
    return states, actions, winner

# Training data with soft labels
def generate_curriculum_data(opponents, n_games=4000):
    X_states, y_policy, y_value = [], [], []
    for _ in tqdm(range(n_games), desc="Generating Self-Play Data"):
        path = random.choices([p for p, _ in opponents], weights=[w for _, w in opponents])[0]
        opponent = load_opponent(path)
        states, actions, winner = self_play_game(opponent)

        for i, state_features in enumerate(states):
            if i % 2 == 0:
                X_states.append(state_features)
                soft_label = np.zeros(SequenceState.get_action_space_size())
                main_idx = actions[i]

                legal = SequenceState().get_legal_actions()
                legal_indices = [SequenceState().action_to_index(a) for a in legal if a['type'] == 'place']
                alt_indices = [idx for idx in legal_indices if idx != main_idx]
                random.shuffle(alt_indices)
                top_k = [main_idx] + alt_indices[:2]
                weights = [0.5, 0.3, 0.2][:len(top_k)]

                for j, idx in enumerate(top_k):
                    soft_label[idx] += weights[j]

                y_policy.append(soft_label)
                y_value.append(1 if winner == 0 else 0)

    return np.array(X_states), np.array(y_policy), np.array(y_value)

# Model architecture
def build_model(input_dim, output_dim):
    inputs = keras.Input(shape=(input_dim,))
    x = layers.Dense(64, activation='relu')(inputs)
    x = layers.Dense(32, activation='relu')(x)
    policy_head = layers.Dense(output_dim, activation='softmax', name='policy')(x)
    value_head = layers.Dense(1, activation='sigmoid', name='value')(x)
    model = keras.Model(inputs=inputs, outputs=[policy_head, value_head])
    model.compile(optimizer='adam',
                  loss=['categorical_crossentropy', 'binary_crossentropy'],
                  metrics={'policy': 'accuracy', 'value': 'mae'})
    return model

# Main training routine
def train_curriculum_model():
    opponents = [
        ("agents.generic.random.Agent", 0.48),
        ("agents.t_068.test05.myAgent", 0.48),
        ("agents.generic.blockerAgent.myAgent", 0.04)
    ]
    print("Starting curriculum training...")
    X, y_policy, y_value = generate_curriculum_data(opponents, n_games=4000)
    print("Data shapes:", X.shape, y_policy.shape, y_value.shape)
    model = build_model(X.shape[1], SequenceState.get_action_space_size())
    model.fit(X, [y_policy, y_value], epochs=10, batch_size=32, validation_split=0.1, verbose=2)
    model.save("policy_value_model_curriculum.keras")
    print("✅ Saved curriculum-trained model with soft labels.")

if __name__ == '__main__':
    train_curriculum_model()