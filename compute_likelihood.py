import value_iteration as vi
import numpy as np
import math
from environments.gridworld import GridworldEnv
from environments.windy_gridworld import WindyGridworldEnv

from maxent_irl import irl

def argmax(actions):
    max = np.max(actions)
    index_array= []
    for index, value in enumerate(actions):
        if max == value:
            index_array.append(index)

    action = np.random.choice(index_array)
    return  action

def generate_trajectories(env, policy, number_of_trajectories, max_length_of_trajectory=30):
    trajectories = []
    for i in range(number_of_trajectories):
        episode = []
        state = env.reset()
        for _ in range(max_length_of_trajectory):
            if policy is None:
                action = env.action_space.sample()
            else:
                action = np.argmax(policy[state])
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, next_state, reward, done))
            state = next_state
            if done:
                break

        trajectories.append(episode)
    return trajectories

def compute_likelihood(policy, trajectories):
    total_likelihood = 0.0
    for trajectory in trajectories:
        likelihood = 0
        for step in trajectory:
            state = int(step[0])
            action = int(step[1])
            prob = policy[state][action]
            log_prob =  math.log(prob)
            likelihood = likelihood + log_prob
        total_likelihood = total_likelihood + likelihood
    return total_likelihood

def check_policy(env, policy):
    state = env.reset()
    env.render()
    for i in range(40):
        action = np.argmax(policy[state])
        next_state, reward, done, _ = env.step(action)
        print(i,(state, action, next_state, reward, done))
        env.render()
        if done:
            break
        state = next_state

def main():
    name = 'Gridworld'
    np.random.seed(1)
    print(name)
    env = GridworldEnv(shape=[10, 10], targets=[99])
    #env = WindyGridworldEnv()
    env.seed(42)

    recovered_policy = irl(env, env_name=name, number_irl_iterations=10, learning_rate=0.1, number_of_trajectories=10,
        max_length_of_trajectory=30)
    #np.save('recovered_policy_'+name, recovered_policy)



    transition_probs = np.zeros((env.observation_space.n, env.action_space.n, env.observation_space.n))
    rewards = np.zeros(env.observation_space.n)

    for state, value in env.P.items():
        for action, value_2 in value.items():
            prob, next_state, reward, done = value_2[0]
            transition_probs[state, action, next_state] = prob
            rewards[next_state] = reward

    policy, _ = vi.value_iteration_cython(transition_probs, rewards, theta=0.001, discount_factor=0.9)
    trajectories = generate_trajectories(env=env,
                                         policy=policy,
                                         number_of_trajectories=10)
    #np.save('trajectories_' + name, trajectories)


    ##check_policy(env, recovered_policy)

    #recovered_policy =  np.load('recovered_policy_'+name+'.npy')
    #trajectories = np.load('trajectories_'+name+'.npy', allow_pickle=True)

    random_trajectories = generate_trajectories(env=env,policy=None,number_of_trajectories=10)

    print('policy')
    print(policy)
    print()

    print('recovered_policy')
    print(recovered_policy)
    print()

    print('for random trajectories',compute_likelihood(recovered_policy, random_trajectories))
    print('for optimal trajectories', compute_likelihood(recovered_policy, trajectories))


if __name__ == '__main__':
    main()