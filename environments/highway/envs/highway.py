import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from collections import defaultdict

NUMBER_OF_ACTIONS = 3  # 0 left, 1 stay, 2 right
LEFT = 0
RIGHT = 2
# STAY = 1
ROAD_LENGTH = 2


class Highway(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, number_lanes=3, road_length=ROAD_LENGTH, reward=1, collided_reward=100, grass_reward=-1):
        self.number_lanes = 3  # number_lanes
        self.road_length = road_length

        # Gym spaces for observation and action space
        self.observation_space = spaces.Discrete(
            (self.road_length ** self.number_lanes) * (self.number_lanes + 2) * 2)  # grass on each side?
        self.action_space = spaces.Discrete(NUMBER_OF_ACTIONS)

        self.reward = reward
        self.collided_reward = collided_reward
        self.grass_reward = grass_reward

        # print('reward', self.reward, 'collided', self.collided_reward, 'grass', self.grass_reward)

        self.possible_states = []
        self.stat_to_int = defaultdict(lambda: -1)
        index = 0
        for i in range(self.number_lanes + 2):
            for j in range(1, self.road_length + 1):
                for k in range(1, self.road_length + 1):
                    for l in range(1, self.road_length + 1):
                        for m in range(2):
                            self.possible_states.append([i, j, k, l, m])
                            self.stat_to_int[str([i, j, k, l, m])] = index
                            index = index + 1

        index = np.random.choice(range(len(self.possible_states)), 1)[0]
        self.current_state = self.possible_states[index][:]
        # print('#states', len(self.possible_states))

    def seed(self, seed=None):
        seeding.np_random(seed)
        # self.road.seed(seed)

    def step(self, action):
        next_states = self.getNextPossibleStates(self.current_state, action)
        index = np.random.choice(range(len(next_states)), 1)[0]
        self.current_state = next_states[index][:]
        reward = self.getReward(self.current_state)

        return self.current_state, reward, False, None

    def getReward(self, state):
        if state[0] == 0 or state[0] == self.number_lanes + 1:
            return self.grass_reward
        elif state[-1] == 1:
            return self.collided_reward
        return self.reward

    def getNextPossibleStates(self, state_, action):
        state = state_[:]
        succ_states = []
        if action == RIGHT:
            # go right
            if state[0] < (self.number_lanes) + 1:
                state[0] = state[0] + 1
        elif action == LEFT:
            # go left
            if state[0] > 0:
                state[0] = state[0] - 1

        state[1] = state[1] - 1
        state[2] = state[2] - 1
        state[3] = state[3] - 1
        if (state[1] == 0):
            i = range(1, self.road_length + 1);
        else:
            i = range(state[1], state[1] + 1);

        if (state[2] == 0):
            j = range(1, self.road_length + 1);
        else:
            j = range(state[2], state[2] + 1);

        if (state[3] == 0):
            k = range(1, self.road_length + 1);
        else:
            k = range(state[3], state[3] + 1);

        if state[0] == 0 or state[0] == self.number_lanes + 1:
            state[-1] = 0
        else:
            if state[state[0]] == 0:
                state[-1] = 1
            else:
                state[-1] = 0

        for newi in i:
            for newj in j:
                for newk in k:
                    new_state = state[:]
                    new_state[1] = newi
                    new_state[2] = newj
                    new_state[3] = newk
                    succ_states.append(new_state)

        return succ_states

    def reset(self):
        index = np.random.choice(range(len(self.possible_states)), 1)[0]
        self.current_state = self.possible_states[index][:]
        return self.current_state

    def render(self, mode='human'):
        pass

    def close(self):
        pass
