import numpy as np
import torch.optim

from simulator.tic_tac_toe import TicTacToe
import matplotlib.pyplot as plt
import threading
import time
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from collections import namedtuple

import pickle

def random_policy(state):
    valid_actions = np.where(state == 0)[0]
    try:
        action = np.random.choice(valid_actions)
    except ValueError:
        raise Exception("The random policy could not select a valid action.")
    return action


def interactive_policy(state, simulator, is_player_one):
    print(f"Current state: {state}")

    # Printing value of the state
    if not is_player_one:
        adjusted_state = -state
    else:
        adjusted_state = state
    with torch.no_grad():
        print(f"Predicted value of current state: {model(torch.from_numpy(adjusted_state).float())}")

    with open("outputs/monte_carlo_policy_evaluation_value_function_ground_truth.pkl", "rb") as f:
        ground_truth_value_function = pickle.load(f)
        print(f"Actual value of current state: {ground_truth_value_function[adjusted_state.tobytes()][0]}")

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


class ValueFunctionApproximator(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(in_features=9, out_features=9),
            nn.ReLU(),
            nn.Linear(in_features=9, out_features=1),
        )

    def forward(self, x):
        return self.linear_relu_stack(x.view(-1, 9))

# Dataset for the simulated experience
class TicTacToeDataset(Dataset):
    def __init__(self, list_of_episodes):
        self.experience = []

        Experience = namedtuple("Experience", ["initial_state", "next_state"])

        for episode in list_of_episodes:
            for t, state in enumerate(episode.state_history):
                if t == len(episode.state_history) - 1:
                    # At the end of the episode, 1s are returned in the "next state" field for a win,
                    # -1s are returned for a loss,
                    # and 2s are returned for a tie
                    if episode.episode_return == 1:
                        self.experience.append(Experience(state, np.ones(state.shape)))
                    elif episode.episode_return == -1:
                        self.experience.append(Experience(state, -np.ones(state.shape)))
                    else:
                        self.experience.append(Experience(state, 2*np.ones(state.shape)))
                else:
                    next_state = episode.state_history[t+1]
                    self.experience.append(Experience(state, next_state))

    def __len__(self):
        return len(self.experience)

    def __getitem__(self, index):
        item = self.experience[index]
        return torch.from_numpy(item.initial_state).float(), torch.from_numpy(item.next_state).float()


def train_loop(model, optimizer, dataloader, loss_func):
    model.train()
    for batch, (X, Y) in enumerate(dataloader):
        # Computing value of next state
        with torch.no_grad():
            next_state_vals = model(Y)
        wins_indices = torch.where(Y==torch.ones(Y.shape[1]),1,0).all(axis=1)
        losses_indices = torch.where(Y==-torch.ones(Y.shape[1]),1,0).all(axis=1)
        ties_indices = torch.where(Y==2*torch.ones(Y.shape[1]),1,0).all(axis=1)

        next_state_vals[wins_indices] = 1
        next_state_vals[losses_indices] = -1
        next_state_vals[ties_indices] = 0


        preds = model(X)
        # No discounting
        loss = loss_func(preds, next_state_vals)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def test_loop(model, dataloader, loss_func):
    model.eval()

    test_dataset_size = len(dataloader.dataset)
    total_loss = 0

    with torch.no_grad():
        for batch, (X,Y) in enumerate(dataloader):
            next_state_vals = model(Y)
            wins_indices = torch.where(Y == torch.ones(Y.shape[1]), 1, 0).all(axis=1)
            losses_indices = torch.where(Y == -torch.ones(Y.shape[1]), 1, 0).all(axis=1)
            ties_indices = torch.where(Y == 2 * torch.ones(Y.shape[1]), 1, 0).all(axis=1)

            next_state_vals[wins_indices] = 1
            next_state_vals[losses_indices] = -1
            next_state_vals[ties_indices] = 0

            preds = model(X)
            loss = loss_func(preds, next_state_vals)
            total_loss += loss * X.shape[0]

    print(f"Average loss on {test_dataset_size} samples in the test dataset: {total_loss/test_dataset_size}")
    return total_loss/test_dataset_size



Episode = namedtuple("Episode", ["state_history", "episode_return"])

training_episode_count = 10000
test_episode_count = 2000
game = TicTacToe()

training_episodes = []
test_episodes = []

# Simulating games and storing the results
print("Acquiring simulation experience...")
for episode in range(training_episode_count + test_episode_count):
    if episode % 1000 == 0:
        print(f"Simulating game number {episode}.")
    game.reset()
    game.simulate_episode(player_one_policy=random_policy, player_two_policy=random_policy)
    player_1_return, player_2_return = episode_returns(game.game_outcome)

    # Applying negative to player 2 state history
    player_two_state_history = game.player_two_state_history.copy()
    for index in range(len(player_two_state_history)):
        player_two_state_history[index] = -player_two_state_history[index]

    if episode < training_episode_count:
        training_episodes.append(Episode(game.player_one_state_history, player_1_return))
        training_episodes.append(Episode(player_two_state_history, player_2_return))
    else:
        test_episodes.append(Episode(game.player_one_state_history, player_1_return))
        test_episodes.append(Episode(player_two_state_history, player_2_return))

print(f"Finished simulating {training_episode_count+test_episode_count} games.")


training_data = TicTacToeDataset(training_episodes)
test_data = TicTacToeDataset(test_episodes)

training_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=128, shuffle=False)

loss = nn.MSELoss()

device = "cpu"
model = ValueFunctionApproximator().to(device)

learning_rate = 0.005
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

epochs = 50

losses = []
t0 = time.time()
for epoch in range(epochs):
    print(f"Starting epoch {epoch} at time {time.time()-t0} seconds.")
    train_loop(model, optimizer, training_dataloader, loss)
    average_loss = test_loop(model, test_dataloader, loss)
    losses.append(average_loss)

plt.figure()
plt.plot(np.linspace(1, epochs, num=epochs), losses)
plt.title("Mean squared error loss vs Epochs")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.show()


#Interactive games of tic tac toe to view value function
player_one_policy = lambda state : interactive_policy(state, game, True)
player_two_policy = lambda state : interactive_policy(state, game, False)
while True:
    game.reset()
    game.simulate_episode(player_one_policy, player_two_policy)