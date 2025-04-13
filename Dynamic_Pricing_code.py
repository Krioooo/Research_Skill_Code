import numpy as np
import matplotlib.pyplot as plt

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

    def reset(self):
        self.current_phase = 0
        self.t = 0

class DiscountedUCB:
    def __init__(self, n_arms, gamma=0.99):
        self.n_arms = n_arms
        self.gamma = gamma
    
        # Initialize discounted counts and sum of rewards for each arm
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)

    def select_arm(self):
        total_counts = np.sum(self.counts)
        # Force exploration: if any arm has near-zero count, select it.
        for i in range(self.n_arms):
            if self.counts[i] < 1e-5:
                return i
        ucb_values = np.zeros(self.n_arms)
        for i in range(self.n_arms):
            avg_reward = self.values[i] / self.counts[i]
            bonus = 2* np.sqrt((0.5 * np.log(total_counts)) / self.counts[i])
            ucb_values[i] = avg_reward + bonus
        return np.argmax(ucb_values)

    def update(self, arm, reward):
        # Apply discounting to all arms
        self.counts *= self.gamma
        self.values *= self.gamma
        # Update selected arm with new observation
        self.counts[arm] += 1
        self.values[arm] += reward

class SlidingWindowUCB:
    def __init__(self, n_arms, window_size=50):
        self.n_arms = n_arms
        self.window_size = window_size
        
        # For each arm, store the recent rewards in a sliding window
        self.rewards = [[] for _ in range(n_arms)]
        self.t = 0

    def select_arm(self):
        current_window = min(self.t, self.window_size)
        # Ensure exploration: select an arm if it hasn't been played in the current window.
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
        # Maintain the window: discard the oldest reward if the window is exceeded
        if len(self.rewards[arm]) > self.window_size:
            self.rewards[arm].pop(0)

# Experiment Setup
prices = [5, 10, 15, 20]
change_points = [2000, 4000, 6000, 8000]
reward_probabilities = [
    [0.3, 0.5, 0.2, 0.1],  # Phase 1
    [0.2, 0.6, 0.3, 0.15], # Phase 2
    [0.1, 0.4, 0.5, 0.3],  # Phase 3
    [0.25, 0.35, 0.3, 0.2], # Phase 4
    [0.15, 0.5, 0.25, 0.4]  # Phase 5
]

env = DynamicPricingEnv(prices, change_points, reward_probabilities)
d_ucb = DiscountedUCB(len(prices), gamma=0.99)
sw_ucb = SlidingWindowUCB(len(prices), window_size=50)

T = 10000  # Total time steps
regrets_d = []
regrets_sw = []

for t in range(T):
    # Discounted UCB
    arm_d = d_ucb.select_arm()
    reward_d = env.step(arm_d)
    d_ucb.update(arm_d, reward_d)
    optimal_reward = max(reward_probabilities[env.current_phase])
    regrets_d.append(optimal_reward - reward_d)

    # Sliding Window UCB
    arm_sw = sw_ucb.select_arm()
    reward_sw = env.step(arm_sw)
    sw_ucb.update(arm_sw, reward_sw)
    regrets_sw.append(optimal_reward - reward_sw)

# Plot Regrets
plt.plot(np.cumsum(regrets_d), label='Discounted UCB')
plt.plot(np.cumsum(regrets_sw), label='Sliding Window UCB')
plt.xlabel("Time")
plt.ylabel("Cumulative Regret")
plt.legend()
plt.title("Comparison of Discounted UCB and Sliding Window UCB in Dynamic Pricing")
plt.show()