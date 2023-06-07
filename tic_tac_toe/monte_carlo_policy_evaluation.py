# Monte Carlo evaluation of a uniform random policy playing against a uniform random policy in tic tac toe.
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
    print(f"Value of current state: {value_function[adjusted_state.tobytes()][0]}, "
          f"Number of times that this state was visited: {value_function[adjusted_state.tobytes()][1]}")

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

# Value function dictionary is of the form: {state : [value, N]}, where N is the number of times
# that state has been visited
value_function = {}
epochs = 5000000
game = TicTacToe()
t0 = time.time()
for epoch in range(epochs):
    if epoch % 1000 == 0:
        print(f"Epoch: {epoch}, elapsed time: {time.time() - t0} seconds")
    game.reset()
    game.simulate_episode(player_one_policy=random_policy, player_two_policy=random_policy)

    # Value function updates
    player_one_return, player_two_return = episode_returns(game.game_outcome)

    for state in game.player_one_state_history:
        if state.tobytes() in value_function:
            value_function[state.tobytes()][1] = value_function[state.tobytes()][1] + 1
            n = value_function[state.tobytes()][1]
            prev_value = value_function[state.tobytes()][0]
            value_function[state.tobytes()][0] = prev_value + (player_one_return - prev_value) / n
        else:
            value_function[state.tobytes()] = [player_one_return, 1]

    for state in game.player_two_state_history:
        # Multiplying state by -1 to represent cells in which your own markers are placed as +1
        state = -state
        if state.tobytes() in value_function:
            value_function[state.tobytes()][1] = value_function[state.tobytes()][1] + 1
            n = value_function[state.tobytes()][1]
            prev_value = value_function[state.tobytes()][0]
            value_function[state.tobytes()][0] = prev_value + (player_two_return - prev_value) / n
        else:
            value_function[state.tobytes()] = [player_two_return, 1]

with open("outputs/monte_carlo_policy_evaluation_value_function.pkl", "wb") as f:
    pickle.dump(value_function, f)

# Interactive games of tic tac toe to view value function
player_one_policy = lambda state : interactive_policy(state, game, value_function, True)
player_two_policy = lambda state : interactive_policy(state, game, value_function, False)
while True:
    game.reset()
    game.simulate_episode(player_one_policy, player_two_policy)
