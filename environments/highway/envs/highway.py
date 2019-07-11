import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from collections import defaultdict

NUMBER_OF_ACTIONS = 3  # 0 left, 1 stay, 2 right
LEFT = 0
RIGHT = 2
# STAY = 1
ROAD_LENGTH = 3


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

        self.possible_states = []  # np.zeros((self.observation_space.n, self.number_lanes + 2), dtype=np.int32)
        self.state_to_int = defaultdict(lambda: -1)
        self.state_to_feature = np.zeros(
            (self.observation_space.n, len(self.getStateFeature([0, 0, 0, 0, 0]))))
        index = 0
        for i in range(self.number_lanes + 2):
            for j in range(1, self.road_length + 1):
                for k in range(1, self.road_length + 1):
                    for l in range(1, self.road_length + 1):
                        for m in range(2):
                            s = [i, j, k, l, m]
                            self.possible_states.append(s)
                            self.state_to_int[str(s)] = index
                            self.state_to_feature[index] = self.getStateFeature(s)
                            index = index + 1

        index = np.random.choice(range(len(self.possible_states)), 1)[0]
        self.current_state = self.possible_states[index][:]
    
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
            i = range(1, self.road_length + 1)
        else:
            i = range(state[1], state[1] + 1)

        if (state[2] == 0):
            j = range(1, self.road_length + 1)
        else:
            j = range(state[2], state[2] + 1)

        if (state[3] == 0):
            k = range(1, self.road_length + 1)
        else:
            k = range(state[3], state[3] + 1)

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

    def getPredecessorPossibleState(self, next_state, action):
        predecessor_states = []
        state = next_state[:]
        predecessor_current_lane = None
        if action == RIGHT:
            # moved into right wall
            if next_state[0] == self.number_lanes + 1:
                predecessor_current_lane = self.number_lanes + 1

            # move one lane left if not in 0 lane
            elif next_state[0] != 0:
                predecessor_current_lane = next_state[0] - 1

        elif action == LEFT:
            # move into left wall
            if next_state[0] == 0:
                predecessor_current_lane = 0

            # move one lane right if not at last lane
            elif next_state[0] != self.number_lanes + 1:
                predecessor_current_lane = next_state[0] + 1

        else:  # straight
            predecessor_current_lane = next_state[0]

        if predecessor_current_lane != None:
            state[0] = predecessor_current_lane
            state[1] = state[1] + 1
            state[2] = state[2] + 1
            state[3] = state[3] + 1

            i = [1]
            j = [1]
            k = [1]
            if (state[1] <= self.road_length):
                i.append(state[1])
            if (state[2] <= self.road_length):
                j.append(state[2])
            if (state[3] <= self.road_length):
                k.append(state[3])

            for newi in i:
                for newj in j:
                    for newk in k:
                        new_state = [predecessor_current_lane, newi, newj, newk, 0]
                        predecessor_states.append(new_state)
        return predecessor_states

    def reset(self):
        index = np.random.choice(range(len(self.possible_states)), 1)[0]
        self.current_state = self.possible_states[index][:]
        return self.current_state

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def getStateFeatureFunction(self):
        return self.getStateFeature

    def getStateFeature(self, state):
        currentLane = np.zeros(self.number_lanes + 2, dtype=np.int8)
        currentLane[state[0]] = 1

        lane1 = np.zeros(self.road_length, dtype=np.int8)
        lane2 = np.zeros(self.road_length, dtype=np.int8)
        lane3 = np.zeros(self.road_length, dtype=np.int8)

        lane1[state[1] - 1] = 1  # zero index is a waste
        lane2[state[2] - 1] = 1
        lane3[state[3] - 1] = 1

        collided = np.zeros(2, dtype=np.int8)
        collided[state[4]] = 1

        return np.concatenate((currentLane, lane1, lane2, lane3, collided), axis=0)
