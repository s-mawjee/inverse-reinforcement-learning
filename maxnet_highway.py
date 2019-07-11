import gym
import environments.highway
import numpy as np
from collections import defaultdict


def argmax(actions):
    max = np.max(actions)

    index_array = []
    for index, value in enumerate(actions):
        if max == value:
            index_array.append(index)
    action = np.random.choice(index_array)
    return action


def value_iteration_function(env, getReward, theta=0.001, discount_factor=0.9):
    V = defaultdict(lambda: 0.0)
    while True:
        delta = 0
        for s in env.possible_states:
            A = one_step_lookahead_function(s, V, env, discount_factor, getReward)
            best_action_value = np.max(A)
            delta = np.max([delta, np.abs(best_action_value - V[str(s)])])
            V[str(s)] = best_action_value
        if delta < theta:
            break

    policy = defaultdict(lambda: np.ones(env.action_space.n) * -1)
    for s in env.possible_states:
        A = one_step_lookahead_function(s, V, env, discount_factor, getReward)
        A_max = np.max(A)
        A_less_max = np.exp(A - A_max)
        A_norm = A_less_max / np.sum(A_less_max)
        policy[str(s)] = A_norm

    return policy, V


def one_step_lookahead_function(state, V, env, discount_factor, getReward):
    A = np.zeros(env.action_space.n)
    for a in range(env.action_space.n):
        next_states = env.getNextPossibleStates(state, a)
        prob = 1.0 / float(len(next_states))
        for next_state in next_states:
            reward = getReward(state)
            A[a] += prob * (reward + (discount_factor * V[str(next_state)]))
    return A


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
            episode.append((str(state), int(action), str(next_state), reward, done))
            state = next_state
        trajectories.append(episode)
    return trajectories


def get_expert_feature_expectations(stateEncoder, trajectories, number_of_feature):
    feature_expectations = np.zeros(number_of_feature)
    for episode in trajectories:
        for state, _, _, _, _ in episode:
            feature_expectations += stateEncoder(eval(state))[:]
    feature_expectations /= (float(len(trajectories)))  # * float(len(trajectories[0])))

    return feature_expectations


def get_expected_state_visitation_frequencies(env, trajectories, policy):
    state_map = env.state_to_int
    number_of_states = env.observation_space.n

    time_steps = len(trajectories[0])
    # mu[s, t] is the prob of visiting state s at time t
    mu = np.zeros([number_of_states, time_steps])

    for trajectory in trajectories:
        state, _, _, _, _ = trajectory[0]
        state_idx = state_map[state]
        mu[state_idx, 0] += 1.0
    mu[:, 0] = mu[:, 0] / float(len(trajectories))

    for next_state in env.possible_states:
        next_state_index = state_map[str(next_state)]
        actions = policy[str(next_state)]
        for action, action_prob in enumerate(actions):
            possible_prev_states = env.getPredecessorPossibleState(next_state, action)
            if len(possible_prev_states) > 0:
                probs = (1.0 / float(len(possible_prev_states)))
                for t in range(time_steps - 1):
                    for prev_state in possible_prev_states:
                        prev_state_index = state_map[str(prev_state)]
                        mu[next_state_index, t + 1] += (mu[prev_state_index, t] * probs * action_prob)
    d = np.sum(mu, axis=1)
    return d


def get_reward_function(stateEncoder, rewards):
    def get_reward(state):
        encoded_state = stateEncoder(state)
        reward = encoded_state.dot(rewards)
        return reward

    return get_reward


def irl(env, number_of_features, optimal_trajectories, number_of_iterations, learning_rate, print_out=False):
    stateEncoder = env.getStateFeatureFunction()

    theta = (np.ones(number_of_features) / float(number_of_features)) * 0.1
    expert = get_expert_feature_expectations(stateEncoder, optimal_trajectories, number_of_features)

    for iteration in range(number_of_iterations):
        if print_out and iteration % (number_of_iterations / 4) == 0:
            print('iteration: {}/{}'.format(iteration, number_of_iterations))

        get_reward_fun = get_reward_function(stateEncoder, theta)
        recovered_policy, recovered_value = value_iteration_function(env, get_reward_fun)

        svf = get_expected_state_visitation_frequencies(env, optimal_trajectories, recovered_policy)
        # compute gradients
        learner = env.state_to_feature.T.dot(svf)
        grad = expert - learner
        theta += learning_rate * grad

    recovered_rewards = np.matmul(env.state_to_feature, theta)

    return recovered_rewards, theta


def main():
    number_of_trajectories = 16
    trajectory_length = 127

    number_of_iterations = 5
    learning_rate = 0.1
    env_name = 'highway-nice-v0'
    print(env_name)
    env = gym.make(env_name)

    optimal_policy, optimal_values = value_iteration_function(env, env.getReward)
    optimal_trajectories = generate_trajectories(env, optimal_policy, number_of_trajectories, trajectory_length)

    # print('[0, 1, 1, 1, 0]', optimal_policy['[0, 1, 1, 1, 0]'], optimal_values['[0, 1, 1, 1, 0]'])
    # print('[0, 1, 1, 1, 1]', optimal_policy['[0, 1, 1, 1, 1]'], optimal_values['[0, 1, 1, 1, 1]'])

    stateEncoder = env.getStateFeatureFunction()
    example_state = env.reset()
    example_encoding = stateEncoder(example_state)

    # print(example_state, '->', example_encoding)
    number_of_features = len(example_encoding)

    recovered_rewards, theta = irl(env, number_of_features, optimal_trajectories, number_of_iterations, learning_rate,
                                   print_out=True)

    get_reward_fun = get_reward_function(stateEncoder, theta)
    recovered_policy, recovered_value = value_iteration_function(env, get_reward_fun)

    print('theta', theta)
    # print('')
    # print('[0, 1, 1, 1, 0]', recovered_policy['[0, 1, 1, 1, 0]'], recovered_value['[0, 1, 1, 1, 0]'])
    # print('[1, 1, 1, 1, 1]', recovered_policy['[1, 1, 1, 1, 1]'], recovered_value['[1, 1, 1, 1, 1]'])
    # print('')
    # print("Policy")
    # for state_str, _ in optimal_policy.items():
    #     print(state_str, "  ", optimal_policy[state_str], "  ", recovered_policy[state_str])
    # print("")
    # count = 0
    # for state_str, _ in optimal_values.items():
    #     state = eval(state_str)
    #     print(state, ": ",
    #           "%10.4f %10.4f %10.4f %10.4f %10.4f" % (
    #               optimal_values[state_str], recovered_value[state_str], env.getReward(state), get_reward_fun(state),
    #               recovered_rewards[env.state_to_int[state_str]]), stateEncoder(state))
    #     count = count + 1


if __name__ == '__main__':
    main()
