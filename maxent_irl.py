import value_iteration as vi
import numpy as np
from environments.gridworld import GridworldEnv
from environments.cliff_walking import CliffWalkingEnv
import matplotlib.pyplot as plt


def normalise(values):
    min_val = np.min(values)
    max_val = np.max(values)
    return (values - min_val) / (max_val - min_val)


def generate_trajectories(env, policy, number_of_trajectories=100, max_length_of_trajectory=20):
    trajectories = []
    for i in range(number_of_trajectories):
        episode = []
        state = env.reset()
        for _ in range(max_length_of_trajectory):
            action = np.argmax(policy[state])
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, next_state, reward, done))
            state = next_state
            if done:
                action = np.argmax(policy[state])
                next_state, reward, done, _ = env.step(action)
                episode.append((state, action, next_state, reward, done))
                break

        trajectories.append(episode)
    return trajectories


def get_feature_expectations(feature_map, trajectories):
    # compute feature_expectations
    feature_expectations = np.zeros(feature_map.shape[1])
    for episode in trajectories:
        for step in episode:
            feature_expectations += feature_map[step[0]]
    feature_expectations = feature_expectations / len(trajectories)
    return feature_expectations


def get_expected_state_visitation_frequencies(transition_probs, trajectories, rewards, policy=None):
    """compute the expected states visitation frequency p(s| theta, T)
    using dynamic programming
    inputs:
      transition_probs     nSxnAXnS matrix - transition dynamics
      trajectories   list of list of Steps - collected from expert
      policy  nSx1 vector

    returns:
      p       nSx1 vector - state visitation frequencies
    """

    if policy == None:
        # compute policy
        policy, _ = vi.value_iteration_cython(transition_probs, rewards, theta=0.001, discount_factor=0.9)

    number_of_states, number_of_actions, _ = np.shape(transition_probs)

    T = len(trajectories[0])
    # mu[s, t] is the prob of visiting state s at time t
    mu = np.zeros([number_of_states, T])

    for trajectory in trajectories:
        mu[trajectory[0][0], 0] += 1
    mu[:, 0] = mu[:, 0] / len(trajectories)

    for s in range(number_of_states):
        for t in range(T - 1):
            mu[s, t + 1] = sum(
                [mu[pre_s, t] * transition_probs[pre_s][np.argmax(policy[s])][s] for pre_s in
                 range(number_of_states)])
            p = np.sum(mu, 1)
    return p


def maxent_irl(feature_map, trajectories, transition_probs, learning_rate, number_of_iterations, print_out=True):
    # Initialise weights
    theta = np.random.uniform(size=(feature_map.shape[1],)) * 0.1

    # Calculate the feature expectations \tilde{phi}.
    feature_expectations = get_feature_expectations(feature_map, trajectories)

    # Gradient descent on theta.
    for iteration in range(number_of_iterations):

        if print_out and iteration % (number_of_iterations / 20) == 0:
            print('iteration: {}/{}'.format(iteration, number_of_iterations))

        # compute reward function
        rewards = np.dot(feature_map, theta)

        # compute state visitation frequencies
        svf = get_expected_state_visitation_frequencies(transition_probs, trajectories, rewards, policy=None)

        # compute gradients
        grad = feature_expectations - feature_map.T.dot(svf)

        # update params
        theta += learning_rate * grad

    rewards = np.dot(feature_map, theta)
    return normalise(rewards)


def irl(env, env_name='Grid World', number_irl_iterations=10, learning_rate=0.09, number_of_trajectories=10,
        max_length_of_trajectory=30):
    transition_probs = np.zeros((env.observation_space.n, env.action_space.n, env.observation_space.n))
    rewards = np.zeros(env.observation_space.n)

    for state, value in env.P.items():
        for action, value_2 in value.items():
            prob, next_state, reward, done = value_2[0]
            transition_probs[state, action, next_state] = prob
            rewards[state] = reward

    policy, values = vi.value_iteration_cython(transition_probs, rewards, theta=0.001, discount_factor=0.9)

    # use identity matrix as feature
    feature_map = np.eye(env.observation_space.n)

    trajectories = generate_trajectories(env=env,
                                         policy=policy,
                                         number_of_trajectories=number_of_trajectories,
                                         max_length_of_trajectory=max_length_of_trajectory)

    recovered_rewards = maxent_irl(feature_map=feature_map,
                                   trajectories=trajectories,
                                   transition_probs=transition_probs,
                                   learning_rate=learning_rate,
                                   number_of_iterations=number_irl_iterations)

    print(np.array(recovered_rewards).reshape(env.shape))
    recovered_policy, recovered_values = vi.value_iteration_cython(transition_probs, recovered_rewards, theta=0.001, discount_factor=0.9)

    fig = plt.figure()
    # TODO: Add spacing between title and plots
    fig.suptitle('MaxEnt IRL on ' + env_name, fontsize=14)

    axs1 = fig.add_subplot(2, 2, 1, aspect='equal')
    fig.gca().invert_yaxis()
    fig.gca().xaxis.tick_top()
    c1 = axs1.pcolor(rewards.reshape(env.shape))
    axs1.set_title("Reward")
    fig.colorbar(c1, ax=axs1)

    axs2 = fig.add_subplot(2, 2, 2, aspect='equal')
    fig.gca().invert_yaxis()
    fig.gca().xaxis.tick_top()
    c2 = axs2.pcolor(values.reshape(env.shape))
    axs2.set_title("Value")
    fig.colorbar(c2, ax=axs2)

    axs3 = fig.add_subplot(2, 2, 3, aspect='equal')
    fig.gca().invert_yaxis()
    fig.gca().xaxis.tick_top()
    c3 = axs3.pcolor(np.array(recovered_rewards).reshape(env.shape))
    axs3.set_title("Recovered Reward")
    fig.colorbar(c3, ax=axs3)

    axs4 = fig.add_subplot(2, 2, 4, aspect='equal')
    fig.gca().invert_yaxis()
    fig.gca().xaxis.tick_top()
    c4 = axs4.pcolor(recovered_values.reshape(env.shape))
    axs4.set_title("Recovered Value")
    fig.colorbar(c4, ax=axs4)

    fig.show()

    return recovered_policy


if __name__ == '__main__':
    np.random.seed(1)
    print('Grid World')
    env = GridworldEnv(shape=[10, 10], targets=[99])
    env.seed(42)
    recovered_policy_grid_world = irl(env, env_name='Grid World')
    print()
    print('Cliff Walking')
    env2 = CliffWalkingEnv()
    env2.seed(42)
    recovered_policy_cliff = irl(env2, env_name='Cliff Walking')
