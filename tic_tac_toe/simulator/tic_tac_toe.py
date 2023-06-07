import numpy as np
import matplotlib.pyplot as plt
import threading


class TicTacToe:
    def __init__(self):
        self._initial_state = np.zeros((3, 3), dtype=int)
        self._state = self._initial_state.copy()

        self.player_one_state_history = []
        self.player_one_action_history = []
        self.player_two_state_history = []
        self.player_two_action_history = []

        self.game_over = False
        self.game_outcome = -1  # 0 is a tie, 1 means player 1 wins, 2 means player 2 wins

    def simulate_episode(self, player_one_policy, player_two_policy):
        while not self.game_over:
            self.simulate_one_timestep(player_one_policy, player_two_policy)

    def simulate_one_timestep(self, player_one_policy, player_two_policy):
        self._make_move(player_one_policy, is_player_one=True)
        self._make_move(player_two_policy, is_player_one=False)

    def _make_move(self, policy, is_player_one):
        if self.game_over:
            return

        # An action is defined to be the index of a free square on the tic tac toe board
        # Indices of the board correspond to the indices of the flattened numpy array representing the state
        action = policy(self._state.flatten())
        row = action // 3
        column = action % 3

        if action > 8 or action < 0:
            raise Exception(f"Invalid action by player one. Attempted action: {action}")
        elif self._state[row, column] != 0:
            raise Exception(f"Invalid action by player one. Attempted to move into an occupied square.")

        if is_player_one:
            self.player_one_state_history.append(self._state.flatten())
            self.player_one_action_history.append(action)
            self._state[row, column] = 1
        else:
            self.player_two_state_history.append(self._state.flatten())
            self.player_two_action_history.append(action)
            self._state[row, column] = -1

        self._check_if_game_over((row, column), is_player_one)

    def _check_if_game_over(self, last_move, player_ones_move):
        if self.game_over:
            return

        row, column = last_move
        # Checking for tic tac toe via a row or column
        if np.abs(np.sum(self._state[row, :])) == 3 or np.abs(np.sum(self._state[:, column])) == 3:
            self.game_over = True

        # Checking for tic tac toe along a diagonal
        if np.abs(np.trace(self._state)) == 3 or np.abs(np.trace(np.fliplr(self._state))) == 3:
            self.game_over = True

        if self.game_over:
            if player_ones_move:
                self.game_outcome = 1
            else:
                self.game_outcome = 2

        # Checking if the board is full
        if np.sum(np.abs(self._state)) == 9:
            self.game_over = True
            self.game_outcome = 0

    def show_board(self):
        fig = plt.figure(figsize=(8, 8))
        fig.subplots_adjust(hspace=0, wspace=0)
        fig.suptitle("Player 1 = X, Player 2 = O")
        for i in range(9):
            plt.subplot(3, 3, i + 1)
            ax = plt.gca()
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xticks([])
            ax.set_yticks([])
            if self._state.flatten()[i] == 1:
                plt.text(0.5, 0.5, "X", fontsize="xx-large")
            elif self._state.flatten()[i] == -1:
                plt.text(0.5, 0.5, "O", fontsize="xx-large")

        plt.show()

    def reset(self):
        self._state = self._initial_state.copy()

        self.player_one_state_history = []
        self.player_one_action_history = []
        self.player_two_state_history = []
        self.player_two_action_history = []

        self.game_over = False
        self.game_outcome = -1


# Policy which allows for user inputs
def interactive_policy(state, simulator):
    print(f"Current state: {state}")
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


if __name__ == "__main__":
    game = TicTacToe()

    player_one_policy = lambda state: interactive_policy(state, game)
    player_two_policy = lambda state: interactive_policy(state, game)

    game.simulate_episode(player_one_policy, player_two_policy)
    print(f"Game outcome: {game.game_outcome}")