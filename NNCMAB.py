import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

# Bandit Environment
class BanditEnvironment:
    def __init__(self, num_arms, reward_means):
        self.num_arms = num_arms
        self.reward_means = reward_means

    def pull_arm(self, arm):
        return np.random.binomial(1, self.reward_means[arm])  # Bernoulli rewards


# ==========================
# Fixed Exploration Then Exploitation
# ==========================
class FixedExplorationThenGreedy:
    def __init__(self, num_arms, exploration_steps):
        self.num_arms = num_arms
        self.exploration_steps = exploration_steps
        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)
        self.t = 1

    def select_arm(self, state=None):
        # fixed exploration
        if self.t <= self.exploration_steps:
            return np.random.randint(self.num_arms)
        # greedy
        else:
            return np.argmax(self.values)

    def update(self, arm, reward):
        self.counts[arm] += 1
        self.values[arm] += (reward - self.values[arm]) / self.counts[arm]
        self.t += 1


# ==========================
# Epsilon-Greedy Algorithm
# ==========================
class EpsilonGreedy:
    def __init__(self, num_arms, epsilon):
        self.num_arms = num_arms
        self.epsilon = epsilon
        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)

    def select_arm(self, state=None):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_arms)
        else:
            return np.argmax(self.values)

    def update(self, arm, reward):
        self.counts[arm] += 1
        self.values[arm] += (reward - self.values[arm]) / self.counts[arm]


# ==========================
# Epsilon-Greedy with Decaying Exploration
# ==========================
class EpsilonGreedyDecaying:
    def __init__(self, num_arms, epsilon_schedule):
        self.num_arms = num_arms
        self.epsilon_schedule = epsilon_schedule  # Function for epsilon_t
        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)
        self.t = 1

    def select_arm(self, state=None):
        epsilon = self.epsilon_schedule(self.t)
        if np.random.rand() < epsilon:
            return np.random.randint(self.num_arms)
        else:
            return np.argmax(self.values)

    def update(self, arm, reward):
        self.counts[arm] += 1
        self.values[arm] += (reward - self.values[arm]) / self.counts[arm]
        self.t += 1


# ==========================
# UCB Algorithm
# ==========================
class UCB:
    def __init__(self, num_arms, c):
        self.num_arms = num_arms
        self.c = c
        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)
        self.t = 1

    def select_arm(self, state=None):
        # 1. check arm
        for arm in range(self.num_arms):
            if self.counts[arm] == 0:
                return arm

        # 2.  every arm try once time at least and calculate UCB
        ucb_values = np.zeros(self.num_arms)
        for arm in range(self.num_arms):
            bonus = self.c * np.sqrt(np.log(self.t) / self.counts[arm])
            ucb_values[arm] = self.values[arm] + bonus

        return np.argmax(ucb_values)

    def update(self, arm, reward):
        self.counts[arm] += 1
        self.values[arm] += (reward - self.values[arm]) / self.counts[arm]
        self.t += 1


# ==========================
# Thompson Sampling Algorithm
# ==========================
class ThompsonSampling:
    def __init__(self, num_arms):
        self.num_arms = num_arms
        self.successes = np.zeros(num_arms)
        self.failures = np.zeros(num_arms)

    def select_arm(self, state=None):
        samples = []
        for i in range(self.num_arms):
            sample = np.random.beta(1 + self.successes[i], 1 + self.failures[i])
            samples.append(sample)
        return np.argmax(samples)

    def update(self, arm, reward):
        if reward == 1:
            self.successes[arm] += 1
        else:
            self.failures[arm] += 1



# ==========================
# Experiment Runner
# ==========================
def run_experiment(bandit_class, bandit_params, env, num_steps):
    bandit = bandit_class(**bandit_params)
    regrets = []
    optimal_reward = max(env.reward_means)

    for t in range(num_steps):
        arm = bandit.select_arm()
        reward = env.pull_arm(arm)
        bandit.update(arm, reward)
        regret = optimal_reward - reward
        regrets.append(regret)

    return np.cumsum(regrets)

# ==========================
# Running Experiments
# ==========================
num_arms = 10
reward_means = np.linspace(0, 1, num_arms)  # Linearly spaced rewards
env = BanditEnvironment(num_arms, reward_means)
num_steps = 10000

# Define epsilon schedule
def epsilon_schedule(t):
    return 1 / (t + 1)

# Plot setup
plt.figure(figsize=(10,5))

# Run and plot Fixed Exploration Then Exploitation
fixed_exploration_regret = run_experiment(FixedExplorationThenGreedy, {'num_arms': num_arms, 'exploration_steps': 100}, env, num_steps)
plt.plot(fixed_exploration_regret, label='Fixed Exploration')

# Run and plot ε-Greedy
epsilon_greedy_regret = run_experiment(EpsilonGreedy, {'num_arms': num_arms, 'epsilon': 0.1}, env, num_steps)
plt.plot(epsilon_greedy_regret, label='Epsilon-Greedy')

# Run and plot Decaying ε-Greedy
decaying_epsilon_greedy_regret = run_experiment(EpsilonGreedyDecaying, {'num_arms': num_arms, 'epsilon_schedule': epsilon_schedule}, env, num_steps)
plt.plot(decaying_epsilon_greedy_regret, label='Decaying Epsilon-Greedy')

# Run and plot UCB
ucb_regret = run_experiment(UCB, {'num_arms': num_arms, 'c': 4}, env, num_steps)
plt.plot(ucb_regret, label='UCB')

# Run and plot Thompson Sampling
thompson_regret = run_experiment(ThompsonSampling, {'num_arms': num_arms}, env, num_steps)
plt.plot(thompson_regret, label='Thompson Sampling')

plt.xlabel("Time Steps")
plt.ylabel("Cumulative Regret")
plt.legend()
plt.title("Bandit Algorithm Performance")
plt.show()

# ==========================
# Instructions for Students
# ==========================
print("TODO: Complete the missing functions for Fixed-Exploration-Greedy, Epsilon-Greedy, Decaying Epsilon-Greedy, UCB, and Thompson Sampling.")
print("TODO: Implement and compare different epsilon schedules (e.g., 1/t, 1/sqrt(t), log(t)/t). Discuss the impact on exploration and cumulative regret in your report.")
print("TODO: Answer the questions in the assignment and conduct the necessary experiments to answer them.")


# ==========================
# Neural Network based Contextual Bandit Extension
# ==========================


import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import matplotlib.pyplot as plt

# 1. define a Context net
class ContextNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(ContextNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.model(x)

# 2. Neural CMAB
class NeuralContextualBandit:
    def __init__(self, state_dim, num_arms, lr=5e-4):
        self.state_dim = state_dim
        self.num_arms = num_arms
        self.net = ContextNet(state_dim, num_arms)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.memory = []
        self.batch_size = 64
        self.global_step = 0

        # Decaying ε-greedy related parameters
        self.epsilon_start = 0.3
        self.epsilon_end = 0.01
        self.epsilon_decay_steps = 10000

    def _get_epsilon(self):
        """
        Calculate the current ε value based on the current global_step to implement linear decay.
        At the beginning, ε is relatively high to ensure sufficient exploration.
        As training progresses, ε decays to a lower level,
        leaning more toward exploiting the Q-values estimated by the network.
        """

        # Linear Decaying: gradually from 0.3 to 0.01
        progress = min(self.global_step / self.epsilon_decay_steps, 1.0)
        eps = self.epsilon_start + (self.epsilon_end - self.epsilon_start) * progress
        return eps

    def select_arm(self, state):
        """
        If a random number is less than ε, randomly select an arm (exploration);
        Otherwise, convert the current state into a tensor and input it
        into the network to obtain the Q-values of each action,
        and select the action with the highest Q-value (exploitation).
        """

        epsilon = self._get_epsilon()
        self.global_step += 1

        if np.random.rand() < epsilon:
            return np.random.randint(self.num_arms)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                q_values = self.net(state_tensor).squeeze().numpy()
            return np.argmax(q_values)

    def store_transition(self, state, arm, reward):
        """
        The state, arm, reward generated by each decision
        is stored in the experience playback pool memory,
        and the oldest data is discarded when the memory exceeds 5000.
        """
        self.memory.append((state, arm, reward))
        if len(self.memory) > 5000:
            self.memory.pop(0)

    def update(self):
        """

        1.If there are fewer than one batch (64) of samples in memory, don’t update.

        2.Randomly sample a batch (experience replay) from memory
        and convert state, action, and reward into tensors.

        3.Use the network to get all Q-values (q_pred),
        then use gather to get the predicted Q-values for the chosen actions (q_selected).

        4.Use Mean Squared Error (MSE) to compute the difference
        between predicted Q-values and actual rewards as the loss.

        5.Use an optimizer to perform backpropagation and
        update the network so its output gets closer to real rewards

        """
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, arms, rewards = zip(*batch)

        states = torch.tensor(np.array(states), dtype=torch.float32)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32)
        arms = torch.tensor(np.array(arms), dtype=torch.int64)

        q_pred = self.net(states)
        q_selected = q_pred.gather(1, arms.unsqueeze(1)).squeeze()
        loss = self.loss_fn(q_selected, rewards)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# Partial Reward Environment
class PartialRewardEnv:
    """
    Designed a softer environment:
    Even if you pick the wrong one,
    as long as it’s “not too far off,” you still get a partial reward.
    This helps the model gradually learn the best strategy
    — not just “right = 1, wrong = 0.

    """
    def __init__(self, state_dim, num_arms):
        """
        best_arm remains unchanged in the same episode, giving "partial reward"

        reward = max(0, 1 - (|arm - best_arm| * 0.1))
        """
        self.state_dim = state_dim
        self.num_arms = num_arms
        self.best_arm = None
        self.state = None

    def reset(self):
        """
        At the start of each episode (training round):
        Randomly pick a “best arm” (best_arm), e.g., one of the arms from 0 to 9.
        Example: best_arm = 7
        Use this best arm’s position as the state for the agent:
        To normalize the value (avoid large inputs), divide it by the max arm number.
        Example: state = 7 / 9 = 0.777...
        Final state is [0.777] (a 1D vector)
        Let the agent know where the best arm is in this episode.
        (Make the environment a Contextual Bandit (CMAB) —
        meaning rewards depend not only on the chosen arm but also on the current state.)

        """
        self.best_arm = np.random.randint(self.num_arms)
        s0 = self.best_arm / max(1, (self.num_arms - 1))
        self.state = np.array([s0], dtype=np.float32)
        return self.state

    def step(self, arm):
        dist = abs(arm - self.best_arm)
        reward = 1.0 - 0.1 * dist
        if reward < 0:
            reward = 0.0
        return self.state, reward


def train_partial_reward_bandit():
    state_dim = 1
    num_arms_nn = 10
    env_context = PartialRewardEnv(state_dim, num_arms_nn)
    nn_bandit = NeuralContextualBandit(state_dim, num_arms_nn, lr=1e-4) #Initialize the neural network

    num_episodes = 800
    steps_per_episode = 30
    episode_rewards = []

    for ep in range(num_episodes):
        state = env_context.reset()
        total_reward = 0
        for t in range(steps_per_episode):
            # Give the current state to the network to estimate the Q-value of each arm.
            # Use ε-Greedy strategy to choose one arm (explore or exploit).
            arm = nn_bandit.select_arm(state)
            next_state, reward = env_context.step(arm)
            #Store the current state, selected arm,
            #and received reward in the replay buffer for later training.
            nn_bandit.store_transition(state, arm, reward)
            #Randomly sample a batch from the replay buffer.
            #Use the batch to compute predicted Q-values and update the network (optimizer + loss).
            nn_bandit.update()
            state = next_state
            total_reward += reward
        episode_rewards.append(total_reward)



 # Training Begin
train_partial_reward_bandit()
print("PartialRewardEnv training (improved) complete.")


# ==========================
# Calculate Cumulative Regret
# ==========================

def train_and_get_model():
    """
    Train the model and return the trained NeuralContextualBandit .
    Train using the previous PartialRewardEnv and NeuralContextualBandit structures.
    """
    state_dim = 1
    num_arms_nn = 10
    env_context = PartialRewardEnv(state_dim, num_arms_nn)
    nn_bandit = NeuralContextualBandit(state_dim, num_arms_nn, lr=5e-5)

    num_episodes = 800
    steps_per_episode = 30

    for ep in range(num_episodes):
        state = env_context.reset()
        for t in range(steps_per_episode):
            arm = nn_bandit.select_arm(state)
            next_state, reward = env_context.step(arm)
            nn_bandit.store_transition(state, arm, reward)
            nn_bandit.update()
            state = next_state
    return nn_bandit


def test_nn_bandit_cumulative_regret(nn_bandit, num_test_episodes=100, steps_per_episode=30):
    """

    (regret = optimal_reward (1) - obtained reward)，
    Run num_test_episodes in the new test environment using the trained model,
    Calculate the cumulative regret for each episode

    """
    state_dim = 1
    num_arms_nn = 10
    env = PartialRewardEnv(state_dim, num_arms_nn)
    cumulative_regrets = []

    for ep in range(num_test_episodes):
        state = env.reset()
        episode_regret = 0
        for t in range(steps_per_episode):
            arm = nn_bandit.select_arm(state)
            next_state, reward = env.step(arm)
            # In this context, the optimal reward is 1.0, so regret = 1.0-reward
            regret = 1.0 - reward
            episode_regret += regret
            state = next_state
        cumulative_regrets.append(episode_regret)



# Train the model and test cumulative regrets
trained_model = train_and_get_model()
test_nn_bandit_cumulative_regret(trained_model)

print("Neural Contextual Bandit cumulative regret test complete.")


# ==========================
# Comparison different model in the same environment:
# Traditional MAB vs neural network CMAB
# ==========================

import math


def run_bandit_on_partial_env(bandit, env, total_steps):
    """
    In the given PartialRewardEnv environment, run the total_steps single-step arm,
    Each step resets the environment (so best_arm changes randomly),
    Calculate and return the cumulative result of each step's immediate regret
    """
    regrets = []
    cumulative = 0.0

    for t in range(total_steps):
        state = env.reset()
        arm = bandit.select_arm(state)
        _, reward = env.step(arm)
        bandit.update(arm, reward)
        regret = 1.0 - reward
        cumulative += regret
        regrets.append(cumulative)
    return regrets


def compare_all_algos_in_partial_env():

    total_steps = 10000
    num_arms = 10
    env = PartialRewardEnv(state_dim=1, num_arms=num_arms)

    # 1) FixedExplorationThenGreedy
    # The first 100 steps are pure exploration,
    # followed by the arm with the highest average reward at present.
    fe_bandit = FixedExplorationThenGreedy(num_arms, exploration_steps=100)
    fe_regret = run_bandit_on_partial_env(fe_bandit, env, total_steps)

    # 2) Epsilon-Greedy
    #There is a 10% probability of random exploration,
    # and a 90% probability of selecting the current best.
    eg_bandit = EpsilonGreedy(num_arms, epsilon=0.1)
    eg_regret = run_bandit_on_partial_env(eg_bandit, env, total_steps)

    # 3) Decaying Epsilon-Greedy
    #The exploration rate gradually decreases,
    # the later the greedy, the initial exploration is stronger.

    def eps_schedule(t):
        return 1 / (t + 1)

    deg_bandit = EpsilonGreedyDecaying(num_arms, eps_schedule)
    deg_regret = run_bandit_on_partial_env(deg_bandit, env, total_steps)

    # 4) UCB
    # Using confidence upper bounds, balance known optimal and explore uncertain options.
    # Parameter c controls the degree of exploration.
    ucb_bandit = UCB(num_arms, c=4)
    ucb_regret = run_bandit_on_partial_env(ucb_bandit, env, total_steps)

    # 5) Thompson Sampling
    ts_bandit = ThompsonSampling(num_arms)
    ts_regret = run_bandit_on_partial_env(ts_bandit, env, total_steps)

    # 6) NeuralContextualBandit
    # modify NNBanditWrapper，in select_arm memory state，when update use last state
    class NNBanditWrapper:
        def __init__(self, state_dim, num_arms):
            self.bandit = NeuralContextualBandit(state_dim, num_arms, lr=5e-4)
            self.last_state = None

        def select_arm(self, state):
            self.last_state = state
            return self.bandit.select_arm(state)

        def update(self, arm, reward):
            # Update with the last recorded state
            self.bandit.store_transition(self.last_state, arm, reward)
            self.bandit.update()

    nn_bandit = NNBanditWrapper(state_dim=1, num_arms=num_arms)
    nn_regret = run_bandit_on_partial_env(nn_bandit, env, total_steps)


    plt.figure(figsize=(10, 6))
    plt.plot(fe_regret, label='FixedExploration')
    plt.plot(eg_regret, label='EpsilonGreedy')
    plt.plot(deg_regret, label='DecayingEpsGreedy')
    plt.plot(ucb_regret, label='UCB')
    plt.plot(ts_regret, label='ThompsonSampling')
    plt.plot(nn_regret, label='NeuralContextualBandit', linewidth=2, color='black')
    plt.xlabel("Time Steps")
    plt.ylabel("Cumulative Regret")
    plt.title("Comparison")
    plt.legend()
    plt.show()



compare_all_algos_in_partial_env()
print("Comparison done: all algos vs partial env with cumulative regret.")
