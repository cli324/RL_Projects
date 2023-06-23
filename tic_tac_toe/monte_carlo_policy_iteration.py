from simulator.tic_tac_toe import TicTacToe
import numpy as np
import torch
import matplotlib.pyplot as plt
import threading
from collections import namedtuple
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
import time


class TicTacToeDataset(Dataset):
    def __init__(self, list_of_episodes):
        self.experience = []

        for episode in list_of_episodes:
            for index in range(len(episode.action_history)):
                state = episode.state_history[index]
                action = episode.action_history[index]
                # One hot encoding of action
                action_encoding = np.zeros(state.shape[0])
                action_encoding[action] = 1
                state_action_pair = np.concatenate((state, action_encoding))

                self.experience.append((state_action_pair, episode.episode_return))

    def __len__(self):
        return len(self.experience)

    def __getitem__(self, index):
        item = self.experience[index]
        return torch.from_numpy(item[0]).float(), torch.Tensor([item[1]])


class ActionValueFunctionApproximator(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(in_features=18, out_features=18),
            nn.ReLU(),
            nn.Linear(in_features=18, out_features=18),
            nn.ReLU(),
            nn.Linear(in_features=18, out_features=1)
        )

    def forward(self, x):
        return self.linear_relu_stack(x.view(-1, 18))


def epsilon_greedy(state, model, epsilon, is_player_one):
    # Adjusting representation of the state so that the player's markers are represented by 1s and the opponent's by -1s
    if is_player_one:
        adjusted_state = state.copy()
    else:
        adjusted_state = -state.copy()

    # Finding the best predicted available action
    available_actions = np.where(adjusted_state == 0)[0]
    greedy_action = None
    max_action_value = None
    model.eval()
    with torch.no_grad():
        for action in available_actions:
            # One hot encoding the action
            action_encoding = np.zeros((state.shape[0]))
            action_encoding[action] = 1
            state_action_pair = np.concatenate((adjusted_state, action_encoding))
            action_value_prediction = model(torch.from_numpy(state_action_pair).float()).flatten()[0]
            if greedy_action is None:
                greedy_action = action
                max_action_value = action_value_prediction
            elif action_value_prediction > max_action_value:
                greedy_action = action
                max_action_value = action_value_prediction

    # Epsilon greedy
    rand = np.random.uniform()
    if rand >= 1 - epsilon:
        selected_action = np.random.choice(available_actions)
    else:
        selected_action = greedy_action

    return selected_action

# Simulator is only needed for display purposes
# Model and is_player_one are only needed to print out action values
def interactive_policy(state, simulator, model, is_player_one):
    # Printing value of the state
    if not is_player_one:
        adjusted_state = -state
    else:
        adjusted_state = state

    available_actions = np.where(state == 0)[0]
    action_values_predictions = []
    for i in range(state.shape[0]):
        if i in available_actions:
            # One hot encoding of the action
            action_encoding = np.zeros((state.shape[0]))
            action_encoding[i] = 1
            state_action_pair = np.concatenate((adjusted_state, action_encoding))
            with torch.no_grad():
                action_value_prediction = model(torch.from_numpy(state_action_pair).float()).flatten()[0]
            action_values_predictions.append(action_value_prediction)
        else:
            action_values_predictions.append(np.nan)

    action_values_predictions = np.array(action_values_predictions).reshape((3, 3))

    print("\n")
    print("Predicted action values:")
    print(action_values_predictions)

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


def generate_experience(train_episode_count, test_episode_count, player_one_policy, player_two_policy):
    Episode = namedtuple("Episode", ["state_history", "action_history", "episode_return"])

    training_episodes = []
    test_episodes = []

    for episode in range(train_episode_count + test_episode_count):
        game.reset()
        game.simulate_episode(player_one_policy=player_one_policy, player_two_policy=player_two_policy)
        player_1_return, player_2_return = episode_returns(game.game_outcome)

        # Applying negative to player 2 state history
        player_two_state_history = game.player_two_state_history.copy()
        for index in range(len(player_two_state_history)):
            player_two_state_history[index] = -player_two_state_history[index]

        if episode < train_episode_count:
            training_episodes.append(
                Episode(game.player_one_state_history, game.player_one_action_history, player_1_return))
            training_episodes.append(Episode(player_two_state_history, game.player_two_action_history, player_2_return))
        else:
            test_episodes.append(
                Episode(game.player_one_state_history, game.player_one_action_history, player_1_return))
            test_episodes.append(Episode(player_two_state_history, game.player_two_action_history, player_2_return))

    return training_episodes, test_episodes


def train_loop(model, loss_func, optimizer, dataloader):
    model.train()

    for batch, (X, Y) in enumerate(dataloader):
        preds = model(X)
        loss = loss_func(preds, Y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def test_loop(model, loss_func, dataloader):
    model.eval()

    total_loss = 0
    with torch.no_grad():
        for X, Y in dataloader:
            preds = model(X)
            loss = loss_func(preds, Y)
            total_loss += loss * X.shape[0]

    average_loss = total_loss / len(dataloader.dataset)
    return average_loss


# -----------------------------------------------Model Training-----------------------------------------------------
game = TicTacToe()
model = ActionValueFunctionApproximator()
loss_function = nn.MSELoss()
learning_rate = 0.005
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
model_save_path = "outputs/monte_carlo_policy_iteration_nn.pth"

training_episode_count = 5000
test_episode_count = 2000

iterations = 100
epochs_per_iteration = 50

t0 = time.time()
losses_across_iterations = []
# Each iteration generates a new batch of data under an updated policy and trains the model on this data
for iter in range(iterations):
    print(f"Beginning iteration {iter} at time {time.time()-t0}.")
    # Creating the dataset for the iteration
    print(f"Acquiring the dataset (time = {time.time()-t0})")
    epsilon = 1 / (1 + 0.2*iter)
    player_one_policy = lambda state: epsilon_greedy(state, model, epsilon, True)
    player_two_policy = lambda state: epsilon_greedy(state, model, epsilon, False)
    training_episodes, test_episodes = generate_experience(training_episode_count, test_episode_count,
                                                           player_one_policy, player_two_policy)
    training_dataset = TicTacToeDataset(training_episodes);
    test_dataset = TicTacToeDataset(test_episodes)
    training_dataloader = DataLoader(training_dataset, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # Training the neural network
    print(f"Training the network (time = {time.time()-t0})")
    losses = []
    for epoch in range(epochs_per_iteration):
        train_loop(model, loss_function, optimizer, training_dataloader)
        average_loss = test_loop(model, loss_function, test_dataloader)
        print(f"Average loss on epoch {epoch}: {average_loss}")
        losses.append(average_loss)
    losses_across_iterations.append(losses[-1])
    print("\n")

    # Plotting the losses over the iteration
    '''plt.figure()
    plt.plot(np.linspace(0, epochs_per_iteration-1, num=epochs_per_iteration), losses)
    plt.title(f"MSE Losses for Iteration {iter} Across {epochs_per_iteration} Epochs of Training")
    plt.xlabel("Epochs")
    plt.ylabel("MSE")
    plt.show()'''

torch.save(model, model_save_path)

# Plotting the losses across iterations
plt.figure()
plt.plot(np.linspace(0, iterations-1, num=iterations), losses_across_iterations)
plt.title(f"MSE Losses Across {iterations} Policy Iterations")
plt.xlabel("Iterations")
plt.ylabel("MSE")
plt.show()

# Interactive games of tic tac toe to view value function
player_one_policy = lambda state: interactive_policy(state, game, model, True)
player_two_policy = lambda state: interactive_policy(state, game, model, False)
while True:
    game.reset()
    game.simulate_episode(player_one_policy, player_two_policy)
