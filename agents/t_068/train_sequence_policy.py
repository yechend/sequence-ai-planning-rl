# train_sequence_policy.py
# Trains a lightweight policy-value network for Sequence using self-play.

import random
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sequence_state import SequenceState
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# === Self-play function ===
def self_play_game(policy_model=None):
    states = []
    actions = []
    state = SequenceState()
    agent_turn = 0

    while not state.game_over():
        legal_actions = state.get_legal_actions()
        if not legal_actions:
            break

        if policy_model and agent_turn == 0:
            state_input = state.encode()
            policy_pred, _ = policy_model.predict(np.array([state_input]), verbose=0)
            action_index = np.argmax(policy_pred[0])
            action = state.index_to_action(action_index)
            if action not in legal_actions:
                action = random.choice(legal_actions)
        else:
            action = random.choice(legal_actions)

        states.append(state.encode())
        actions.append(state.action_to_index(action))
        state.apply_action(action)
        agent_turn = 1 - agent_turn

    winner = state.get_winner()
    return states, actions, winner

# === Generate training data ===
def generate_self_play_data(n_games=10):
    X_states, y_policy, y_value = [], [], []
    for g in range(n_games):
        states, actions, winner = self_play_game()
        for i, state_features in enumerate(states):
            if i % 2 == 0:
                X_states.append(state_features)
                y_policy.append(actions[i])
                y_value.append(1 if winner == 0 else 0)
    return np.array(X_states), np.array(y_policy), np.array(y_value)

# === Define and train model ===
def build_model(input_dim, output_dim):
    inputs = keras.Input(shape=(input_dim,))
    x = layers.Dense(64, activation='relu')(inputs)
    x = layers.Dense(32, activation='relu')(x)
    policy_head = layers.Dense(output_dim, activation='softmax', name='policy')(x)
    value_head = layers.Dense(1, activation='sigmoid', name='value')(x)
    model = keras.Model(inputs=inputs, outputs=[policy_head, value_head])
    model.compile(optimizer='adam',
                  loss=['sparse_categorical_crossentropy', 'binary_crossentropy'],
                  metrics={'policy': 'accuracy', 'value': 'mae'})
    return model

# === Main training procedure ===
def train_and_save_model():
    print("Generating self-play data...")
    X, y_policy, y_value = generate_self_play_data(n_games=10)
    print("Data shape:", X.shape, y_policy.shape, y_value.shape)

    print("Training model...")
    model = build_model(input_dim=X.shape[1], output_dim=SequenceState.get_action_space_size())
    model.fit(X, [y_policy, y_value], epochs=10, batch_size=32, validation_split=0.1, verbose=2)

    print("Saving model to policy_value_model.h5")
    model.save("policy_value_model.h5")
    print("Training complete.")

if __name__ == '__main__':
    train_and_save_model()
