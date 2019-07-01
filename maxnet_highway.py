import gym
import numpy as np
# import value_iteration as vi
import math
import pickle
import environments.highway
from collections import defaultdict
import random


def value_iteration_function(env, getReward, theta=0.001, discount_factor=0.9):
    number_of_actions = env.action_space.n

    V = defaultdict(lambda: 0.0)
    while True:
        delta = 0
        for s in env.possible_states:
            A = one_step_lookahead_function(s, V, env, discount_factor, number_of_actions, getReward)
            best_action_value = np.max(A)
            delta = np.max([delta, np.abs(best_action_value - V[str(s)])])
            V[str(s)] = best_action_value
        if delta < theta:
            break

    policy = defaultdict(lambda: np.ones(number_of_actions) * -1)
    for s in env.possible_states:
        A = one_step_lookahead_function(s, V, env, discount_factor, number_of_actions, getReward)
        # best_action = np.argmax(A)
        # policy[str(s)] = best_action
        A_norm = np.exp(A) / sum(np.exp(A))
        for n in A_norm:
            if math.isnan(n):
                print('help 2')
        policy[str(s)] = A_norm

    return policy, V


def one_step_lookahead_function(state, V, env, discount_factor, number_of_actions, getReward):
    A = np.zeros(number_of_actions)
    for a in range(number_of_actions):
        next_states = env.getNextPossibleStates(state, a)
        prob = 1.0 / float(len(next_states))
        for next_state in next_states:
            reward = getReward(next_state)
            # print(1.0 / len(next_states), prob, reward, V[str(next_state)])
            A[a] += prob * (reward + discount_factor * V[str(next_state)])
    return A


def argmax(actions):
    max = np.max(actions)

    index_array = []
    for index, value in enumerate(actions):
        if max == value:
            index_array.append(index)
    action = np.random.choice(index_array)
    return action


def compute_likelihood(policy, trajectories):
    likelihoods = np.zeros(len(trajectories))
    count = 0
    for trajectory in trajectories:
        likelihood = 0.0
        for state, action, _, _, _ in trajectory:
            prob = policy[str(state)][action]
            log_prob = math.log(prob)
            likelihood = likelihood + log_prob
        likelihoods[count] = likelihood
        count = count + 1
    return np.mean(likelihoods), np.std(likelihoods)


def compute_average_reward(env_name, timesteps, policy):
    total_reward = 0.0
    # seed = 42
    env = gym.make(env_name)
    # env.seed(seed)

    state = env.reset()

    for t in range(timesteps):
        action = argmax(policy[str(state)])
        next_state, reward, _, _ = env.step(action)
        total_reward = total_reward + reward
        state = next_state

    return total_reward / float(timesteps)


def generate_trajectories(env, policy, number_of_trajectories=100, length_of_trajectory=127):
    trajectories = []
    for i in range(number_of_trajectories):
        episode = []
        state = env.reset()
        for _ in range(length_of_trajectory):
            if policy is None:
                action = np.random.choice(env.action_space.n, 1)[0]
            else:
                action = argmax(policy[str(state)])
            next_state, reward, done, _ = env.step(action)
            episode.append((str(state), action, str(next_state), reward, done))
            state = next_state
        trajectories.append(episode)
    return trajectories


def get_expected_state_visitation_frequencies(env, trajectories, policy, state_map):
    random.shuffle(trajectories)
    number_of_states = env.observation_space.n

    T = len(trajectories)
    # mu[s, t] is the prob of visiting state s at time t
    mu = np.zeros([number_of_states, T])

    for trajectory in trajectories:
        for state, _, _, _, _ in trajectory:
            mu[state_map[state], 0] += 1
    mu[:, 0] = mu[:, 0] / float(len(trajectories))

    for state in env.possible_states:
        state_index = state_map[str(state)]
        action = argmax(policy[str(state)])
        possible_next_states = env.getNextPossibleStates(state, action)
        probs = (1.0 / float(len(possible_next_states)))
        for t in range(T - 1):
            for next_state in possible_next_states:
                next_state_index = state_map[str(next_state)]
                mu[next_state_index, t + 1] = mu[state_index, t] * probs
                # mu[s, t + 1] = sum(
                #     [mu[pre_s, t] * env.getTransitionProbs(pre_s, np.argmax(policy[str(pre_s)]), s) for pre_s in
                #      env.possible_states])
    p = np.sum(mu, 1)
    return p


def getRewardFunction(env, rewards):
    state_map = env.stat_to_int

    def getReward(state):
        return rewards[state_map[str(state)]]

    return getReward


def irl(env, number_of_iterations, trajectories, learning_rate=0.1):
    state_map = env.stat_to_int

    feature_map = np.eye(env.observation_space.n)

    # Initialise weights
    theta = np.random.uniform(size=(feature_map.shape[0],)) * 0.1

    # Calculate the feature expectations (expert).
    feature_expectations = np.zeros(feature_map.shape[1])
    for episode in trajectories:
        for state, action, next_state, _, _ in episode:
            feature_expectations += feature_map[state_map[state]]
    feature_expectations = feature_expectations / float(len(trajectories))

    # Gradient descent on theta.
    for iteration in range(number_of_iterations):

        if iteration % (number_of_iterations / 20) == 0:
            print('iteration: {}/{}'.format(iteration, number_of_iterations))

        # compute reward function
        rewards = np.dot(feature_map, theta)
        getReward = getRewardFunction(env, rewards)
        recovered_policy, recovered_value = value_iteration_function(env, getReward, theta=0.001, discount_factor=0.9)

        # compute state visitation frequencies
        svf = get_expected_state_visitation_frequencies(env, trajectories, recovered_policy, state_map)

        # compute gradients
        grad = feature_expectations - feature_map.T.dot(svf)

        # update params
        theta = theta + (learning_rate * grad)
        # # Clip theta
        # for j in range(len(theta)):
        #     if theta[j] < 0:
        #         theta[j] = 0

    recovered_rewards = np.dot(feature_map, theta)
    recovered_rewards = recovered_rewards / sum(recovered_rewards) * 100
    return recovered_rewards


def main(env_name, save=False):
    seed = 42
    env = gym.make(env_name)
    env.seed(seed)

    optimal_policy, optimal_values = value_iteration_function(env, env.getReward)
    optimal_policy = dict(optimal_policy)
    optimal_trajectories = generate_trajectories(env, optimal_policy, 16, 127)

    print(optimal_policy)

    recovered_rewards = irl(env, 10, optimal_trajectories)
    reward_function = getRewardFunction(env, recovered_rewards)
    recovered_policy, recovered_values = value_iteration_function(env, reward_function)
    recovered_policy = dict(recovered_policy)

    print(recovered_policy)

    # env.seed(42)
    random_trajectories = generate_trajectories(env, None, 16, 127)

    if save:
        pickle_out = open(env_name + "_policies_trajectories.pickle", "wb")
        pickle.dump((optimal_policy, optimal_trajectories, recovered_rewards, recovered_policy, random_trajectories),
                    pickle_out)
        pickle_out.close()

    return optimal_policy, optimal_trajectories, recovered_rewards, recovered_policy, random_trajectories


def getArrow(action):
    if action == 0:
        return '←'
    if action == 2:
        return '→'
    if action == 1:
        return '↑'


if __name__ == '__main__':
    env_name = 'highway-nasty-v0'
    optimal_policy, optimal_trajectories, recovered_rewards, recovered_policy, random_trajectories = main(env_name,
                                                                                                          False)
    pickle_in = open("highway-nice-v0_policies_trajectories_1.pickle", "rb")
    optimal_policy, optimal_trajectories, recovered_rewards, recovered_policy, random_trajectories = pickle.load(
        pickle_in)

    print('')
    print('average reward with optimal policy', compute_average_reward(env_name, 1000, optimal_policy))
    print('average reward with recovered policy', compute_average_reward(env_name, 1000, recovered_policy))
    print('')

    print('for optimal trajectories (optimal)', compute_likelihood(optimal_policy, optimal_trajectories))
    print('for optimal trajectories (recovered)', compute_likelihood(recovered_policy, optimal_trajectories))
    print('')
    print('for random trajectories (optimal)', compute_likelihood(optimal_policy, random_trajectories))
    print('for random trajectories (recovered)', compute_likelihood(recovered_policy, random_trajectories))

    op_policy = np.ndarray((len(optimal_policy.items()), 3))
    count = 0
    for state, actions_probs in optimal_policy.items():
        op_policy[count] = actions_probs[:]
        count = count + 1
    print(op_policy)
    print('')
    rc_policy = np.ndarray((len(recovered_policy.items()), 3))
    count = 0
    for state, actions_probs in recovered_policy.items():
        rc_policy[count] = actions_probs[:]
        count = count + 1
    print(rc_policy)
    print('wait')
