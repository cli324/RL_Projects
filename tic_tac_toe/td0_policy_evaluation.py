# TD0 evaluation of a uniform random policy playing against a uniform random policy in tic tac toe.
# The value function representation is a table.

import numpy as np
from simulator.tic_tac_toe import TicTacToe
import matplotlib.pyplot as plt
import threading
import time
import pickle


def random_policy(state):
    valid_actions = np.where(state == 0)[0]
    try:
        action = np.random.choice(valid_actions)
    except ValueError:
        raise Exception("The random policy could not select a valid action.")
    return action


def interactive_policy(state, simulator, value_function, is_player_one):
    print(f"Current state: {state}")

    # Printing value of the state
    if not is_player_one:
         adjusted_state = -state
    else:
        adjusted_state = state
    print(f"Value of current state: {value_function[adjusted_state.tobytes()]}")

    received_input = []
    def request_user_input(output):
        valid_inputs = np.where(state == 0)[0]

        while True:
            user_input = input(f"Which square will you play next? Your options are: {valid_inputs}\n")
            try:
                user_input = int(user_input)
            except ValueError:
                print("Invalid input received! Please enter an integer corresponding to the index of a grid cell.")
                continue

            if user_input in valid_inputs:
                output.append(user_input)
                plt.close("all")
                return
            else:
                print("Invalid input received! Please enter an integer corresponding to the index of a grid cell.")

    receive_input_thread = threading.Thread(target=request_user_input, args=(received_input,), daemon=True)
    receive_input_thread.start()
    simulator.show_board()
    receive_input_thread.join()

    return received_input[-1]

def episode_returns(game_outcome):
    # Returns are returned in the format (player_one_return, player_two_return)
    if game_outcome == 0:
        return (0, 0)
    elif game_outcome == 1:
        return (1, -1)
    elif game_outcome == 2:
        return (-1, 1)
    else:
        raise Exception(f"Invalid value of game_outcome passed to episode_returns. Received value: {game_outcome}")

# Value function is of the form {state : value} - this is different from the monte carlo script
value_function = {}
epochs = 50000
game = TicTacToe()
t0 = time.time()

for epoch in range(epochs):
    if epoch % 1000 == 0:
        print(f"Epoch: {epoch}, elapsed time: {time.time() - t0} seconds")

    game.reset()
    # Using an episodic rather than incremental approach - makes the code cleaner
    game.simulate_episode(player_one_policy=random_policy, player_two_policy=random_policy)

    player_one_return, player_two_return = episode_returns(game.game_outcome)

    alpha = 1 / (0.001 * epoch + 1)

    # Iterating backwards over the state history to update the value function
    for t in range(len(game.player_one_state_history)-1, -1, -1):
        state = game.player_one_state_history[t].tobytes()
        if t == len(game.player_one_state_history) - 1:
            value_of_next_state = player_one_return
        else:
            value_of_next_state = value_function[game.player_one_state_history[t+1].tobytes()]

        if state in value_function:
            value_function[state] = value_function[state] + alpha * (value_of_next_state - value_function[state])
        else:
            value_function[state] = alpha * value_of_next_state

    for t in range(len(game.player_two_state_history) - 1, -1, -1):
        state = (-game.player_two_state_history[t]).tobytes()
        if t == len(game.player_two_state_history) - 1:
            value_of_next_state = player_two_return
        else:
            value_of_next_state = value_function[(-game.player_two_state_history[t + 1]).tobytes()]

        if state in value_function:
            value_function[state] = value_function[state] + alpha * (value_of_next_state - value_function[state])
        else:
            value_function[state] = alpha * value_of_next_state


# Interactive games of tic tac toe to view value function
player_one_policy = lambda state : interactive_policy(state, game, value_function, True)
player_two_policy = lambda state : interactive_policy(state, game, value_function, False)
while True:
    game.reset()
    game.simulate_episode(player_one_policy, player_two_policy)
