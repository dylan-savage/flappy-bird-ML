
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from collections import deque
import random
from environment import FlappyBirdEnv

# Hyperparameters
GAMMA = 0.99
LEARNING_RATE = 0.0001
EPSILON_START = .5
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
BATCH_SIZE = 128
MEMORY_SIZE = 50000
EPISODES = 2000
SHOW_GAME = False
PLOT_INTERVAL = 10  

# Replay Buffer
class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def size(self):
        return len(self.buffer)

def build_model(input_size, output_size):
    """Build a simple feedforward neural network with dropout."""
    model = nn.Sequential(
        nn.Linear(input_size, 256),
        nn.ReLU(),
        nn.Dropout(0.2),  # Add dropout for regularization
        nn.Linear(256, output_size)
    )
    return model

def plot_in_game_scores(all_scores):
    # Plot in-game scores for each episode.

    plt.figure("In-Game Scores")  
    plt.clf()  
    plt.plot(all_scores, label="Game Score (per episode)", marker='o')
    plt.title("In-Game Scores Over Episodes")
    plt.xlabel("Episodes")
    plt.ylabel("Score")
    plt.legend()
    plt.draw() 
    plt.pause(0.001) 

def plot_training_performance(scores, mean_scores, best_scores):  
    # Plot training performance over batches of episodes.

    plt.figure("Training Performance")  
    plt.clf()  
    plt.plot(scores, label="Game Score (every 10 games)", marker='o')
    plt.plot(mean_scores, label="Mean Game Score (average)", linestyle="--")
    plt.plot(best_scores, label="Best Score (up to this point)", linestyle="-.", color='green')
    plt.title("Training Performance Over Episodes")
    plt.xlabel("Game Batches (Every 10 Episodes)")
    plt.ylabel("Score")
    plt.legend()
    plt.draw() 
    plt.pause(0.001) 

def train():
    # Initialize the environment, model, and replay buffer
    env = FlappyBirdEnv(screen_width=800, screen_height=600)
    state_shape = env.reset().shape
    action_space = 2  # [0: Do Nothing, 1: Jump]

    # Build the model and optimizer
    model = build_model(input_size=state_shape[0], output_size=action_space)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()
    replay_buffer = ReplayBuffer(max_size=MEMORY_SIZE)

    scores = [] 
    mean_scores = [] 
    best_scores = []  
    all_scores = [] 
    pipes_passed_scores = []  
    total_score = 0
    epsilon = EPSILON_START
    record = 0

    for episode in range(EPISODES):
        state = env.reset()
        total_reward = 0
        pipes_passed = 0  
        done = False

        while not done:
            if SHOW_GAME:
                env.render()  

            # Choose action: epsilon-greedy strategy
            if np.random.rand() < epsilon:
                action = np.random.randint(0, action_space)  # Explore
            else:
                q_values = model(torch.tensor(state, dtype=torch.float32))
                action = torch.argmax(q_values).item()  # Exploit

            # Take a step in the environment
            next_state, reward, done, info = env.step(action)
            total_reward += reward

            # Extract pipes passed from the `info` dictionary
            pipes_passed = info.get("pipes_passed", 0)

            # Store transition in replay buffer
            replay_buffer.add((state, action, reward, next_state, done))
            state = next_state

            # Train the model if the replay buffer is large enough
            if replay_buffer.size() >= BATCH_SIZE:
                batch = replay_buffer.sample(BATCH_SIZE)
                states, actions, rewards, next_states, dones = map(np.array, zip(*batch))

                # Convert to tensors
                states = torch.tensor(states, dtype=torch.float32)
                actions = torch.tensor(actions, dtype=torch.long)
                rewards = torch.tensor(rewards, dtype=torch.float32)
                next_states = torch.tensor(next_states, dtype=torch.float32)
                dones = torch.tensor(dones, dtype=torch.float32)

                # Compute Q-values and targets
                q_values = model(states)
                q_next = model(next_states).detach()
                targets = q_values.clone()
                for i in range(BATCH_SIZE):
                    Q_new = rewards[i]
                    if not dones[i]:
                        Q_new += GAMMA * torch.max(q_next[i])
                    targets[i, actions[i]] = Q_new

                # Compute loss and optimize
                optimizer.zero_grad()
                loss = loss_fn(q_values, targets)
                loss.backward()
                optimizer.step()

        # Record game score
        all_scores.append(total_reward) 
        pipes_passed_scores.append(pipes_passed)  
        total_score += total_reward
        mean_score = total_score / (episode + 1)
        record = max(record, total_reward)

        print(f"Episode {episode + 1}/{EPISODES} - Pipes Passed: {pipes_passed}, Mean Score: {mean_score:.2f},")

        # Update batch scores every 10 games
        if (episode + 1) % PLOT_INTERVAL == 0:
            batch_mean = np.mean(all_scores[-PLOT_INTERVAL:])  # Mean of last 10 scores
            scores.append(batch_mean)
            mean_scores.append(mean_score)
            best_scores.append(record)

         
            plot_in_game_scores(pipes_passed_scores)  
            plot_training_performance(scores, mean_scores, best_scores)  

        # Decay epsilon
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

    env.close()
    torch.save(model.state_dict(), "flappy_bird_dqn.pth")
    print("Training complete!")


if __name__ == "__main__":
    train()