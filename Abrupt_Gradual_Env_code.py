import numpy as np
import matplotlib.pyplot as plt

# --------- Abrupt Changes Env ---------
class DynamicPricingEnv:
    def __init__(self, prices, change_points, reward_probabilities):
        self.prices = prices
        self.change_points = change_points
        self.reward_probabilities = reward_probabilities
        self.current_phase = 0
        self.t = 0

    def step(self, price_index):
        if self.t in self.change_points:
            self.current_phase = min(self.current_phase + 1, len(self.reward_probabilities) - 1)
        self.t += 1
        return np.random.rand() < self.reward_probabilities[self.current_phase][price_index]

    def get_optimal_reward(self):
        return max(self.reward_probabilities[self.current_phase])

    def reset(self):
        self.current_phase = 0
        self.t = 0

# --------- Gradual Changes Env ---------
class GradualDynamicPricingEnv:
    def __init__(self, prices, T):
        self.prices = prices
        self.T = T
        self.t = 0

    def get_probabilities(self):
        return [
            0.3 + 0.2 * self.t / self.T,
            0.6 - 0.1 * self.t / self.T,
            0.2 + 0.1 * self.t / self.T,
            0.1 + 0.3 * self.t / self.T
            
        ]

    def step(self, price_index):
        probs = self.get_probabilities()
        reward = np.random.rand() < probs[price_index]
        self.t += 1
        return reward

    def get_optimal_reward(self):
        return max(self.get_probabilities())

    def reset(self):
        self.t = 0

# --------- Discounted UCB ---------
class DiscountedUCB:
    def __init__(self, n_arms, gamma=0.99):
        self.n_arms = n_arms
        self.gamma = gamma
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)

    def select_arm(self):
        total_counts = np.sum(self.counts)
        for i in range(self.n_arms):
            if self.counts[i] < 1e-5:
                return i
        ucb_values = np.zeros(self.n_arms)
        for i in range(self.n_arms):
            avg_reward = self.values[i] / self.counts[i]
            bonus = 2 * np.sqrt((0.5 * np.log(total_counts)) / self.counts[i])
            ucb_values[i] = avg_reward + bonus
        return np.argmax(ucb_values)

    def update(self, arm, reward):
        self.counts *= self.gamma
        self.values *= self.gamma
        self.counts[arm] += 1
        self.values[arm] += reward

# --------- Sliding Window UCB ---------
class SlidingWindowUCB:
    def __init__(self, n_arms, window_size=50):
        self.n_arms = n_arms
        self.window_size = window_size
        self.rewards = [[] for _ in range(n_arms)]
        self.t = 0

    def select_arm(self):
        current_window = min(self.t, self.window_size)
        for i in range(self.n_arms):
            if len(self.rewards[i]) == 0:
                return i
        ucb_values = np.zeros(self.n_arms)
        for i in range(self.n_arms):
            n_i = len(self.rewards[i])
            avg_reward = np.mean(self.rewards[i])
            bonus = np.sqrt((0.5 * np.log(current_window)) / n_i)
            ucb_values[i] = avg_reward + bonus
        return np.argmax(ucb_values)

    def update(self, arm, reward):
        self.t += 1
        self.rewards[arm].append(reward)
        if len(self.rewards[arm]) > self.window_size:
            self.rewards[arm].pop(0)

# --------- Experiment Setup ---------
T = 10000
prices = [5, 10, 15, 20]
change_points = [2000, 4000, 6000, 8000]
reward_probabilities = [
    [0.3, 0.5, 0.2, 0.1],
    [0.2, 0.6, 0.3, 0.15],
    [0.1, 0.4, 0.5, 0.3],
    [0.25, 0.35, 0.3, 0.2],
    [0.15, 0.5, 0.25, 0.4]
]

# --------- Abrupt Experiments ---------
env_abrupt = DynamicPricingEnv(prices, change_points, reward_probabilities)
d_ucb_abrupt = DiscountedUCB(len(prices))
sw_ucb_abrupt = SlidingWindowUCB(len(prices), window_size=50)

regrets_d_abrupt = []
regrets_sw_abrupt = []

for t in range(T):
    # D-UCB
    arm_d = d_ucb_abrupt.select_arm()
    reward_d = env_abrupt.step(arm_d)
    d_ucb_abrupt.update(arm_d, reward_d)
    regrets_d_abrupt.append(env_abrupt.get_optimal_reward() - reward_d)

    # SW-UCB
    arm_sw = sw_ucb_abrupt.select_arm()
    reward_sw = env_abrupt.step(arm_sw)
    sw_ucb_abrupt.update(arm_sw, reward_sw)
    regrets_sw_abrupt.append(env_abrupt.get_optimal_reward() - reward_sw)

# --------- Gradual Experiments ---------
env_gradual = GradualDynamicPricingEnv(prices, T)
d_ucb_gradual = DiscountedUCB(len(prices))
sw_ucb_gradual = SlidingWindowUCB(len(prices), window_size=50)

regrets_d_gradual = []
regrets_sw_gradual = []

for t in range(T):
    probs = env_gradual.get_probabilities()
    optimal_reward = max(probs)

    # D-UCB
    arm_d = d_ucb_gradual.select_arm()
    reward_d = env_gradual.step(arm_d)
    d_ucb_gradual.update(arm_d, reward_d)
    regrets_d_gradual.append(optimal_reward - reward_d)

    # SW-UCB
    arm_sw = sw_ucb_gradual.select_arm()
    reward_sw = env_gradual.step(arm_sw)
    sw_ucb_gradual.update(arm_sw, reward_sw)
    regrets_sw_gradual.append(optimal_reward - reward_sw)

# --------- Plot All Curves ---------
plt.figure(figsize=(10, 6))
plt.plot(np.cumsum(regrets_d_abrupt), label="D-UCB (abrupt)")
plt.plot(np.cumsum(regrets_sw_abrupt), label="SW-UCB (abrupt)")
plt.plot(np.cumsum(regrets_d_gradual), label="D-UCB (gradual)")
plt.plot(np.cumsum(regrets_sw_gradual), label="SW-UCB (gradual)")
plt.xlabel("Time")
plt.ylabel("Cumulative Regret")
plt.title("D-UCB vs SW-UCB in Abrupt and Gradual Environments")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()