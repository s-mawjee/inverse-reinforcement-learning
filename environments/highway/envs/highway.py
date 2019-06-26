import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from collections import defaultdict

NUMBER_OF_ACTIONS = 3  # 0 left, 1 stay, 2 right
LEFT = 0
RIGHT = 2
# STAY = 1
ROAD_LENGTH = 6


class Highway(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, number_lanes=3, road_length=ROAD_LENGTH, reward=10, collided_reward=-50, grass_reward=-10000):
        self.number_lanes = number_lanes
        self.road_length = road_length
        self.road = Road(number_lanes=self.number_lanes, road_length=self.road_length)

        # Gym spaces for observation and action space
        self.observation_space = spaces.Discrete(
            (self.road_length ** self.number_lanes) * (self.number_lanes + 2) * 2)  # grass on each side?
        self.action_space = spaces.Discrete(NUMBER_OF_ACTIONS)

        self.is_collided = 0
        self.collisions = 0
        self.current_lane = 1
        self.car_collided = None

        self.current_state = self._getState()

        self.reward = reward
        self.collided_reward = collided_reward
        self.grass_reward = grass_reward

        self.possibleStates = []
        index = 0
        for i in range(self.number_lanes + 2):
            for j in range(1, self.road_length + 1):
                for k in range(1, self.road_length + 1):
                    for l in range(1, self.road_length + 1):
                        for m in range(2):
                            self.possibleStates.append([i, j, k, l, m])
                            index = index + 1

    def seed(self, seed=None):
        seeding.np_random(seed)
        self.road.seed(seed)

    def step(self, action):
        if action == RIGHT:
            # go right
            if self.current_lane < (self.road.number_lanes + 1):
                self.current_lane = self.current_lane + 1
        elif action == LEFT:
            # go left
            if self.current_lane > 0:
                self.current_lane = self.current_lane - 1

        ##self.road.update()

        next_states = self.getNextPossibleStates(self.current_state, action)
        index = np.random.choice(range(len(next_states)), 1)[0]
        self.current_state = next_states[index]
        done = False
        reward = self.getReward(self.current_state)
        # if self.is_collided:
        #     done = True
        return self.current_state, reward, done, None

    def getReward(self, state):
        reward = self.reward
        if state[0] == 0 or state[0] == self.number_lanes + 1:
            reward = self.grass_reward
        elif state[-1] == 1:
            reward = self.collided_reward
        return reward

    def getTransitionProbs(self, state, action, next_state):
        possible_next_state = self.getNextPossibleStates(state, action)
        probs = 1 / (len(possible_next_state))
        for s in possible_next_state:
            if next_state == s:
                return probs
        return 0.0

    def getNextPossibleStates(self, state_, action):
        state = state_[:]
        succ_states = []
        if action == RIGHT:
            # go right
            if state[0] < (self.road.number_lanes) + 1:
                state[0] = state[0] + 1
        elif action == LEFT:
            # go left
            if state[0] > 0:
                state[0] = state[0] - 1

        state[1] = state[1] - 1
        state[2] = state[2] - 1
        state[3] = state[3] - 1
        if (state[1] == 0):
            i = range(1, 7);
        else:
            i = range(state[1], state[1] + 1);

        if (state[2] == 0):
            j = range(1, 7);
        else:
            j = range(state[2], state[2] + 1);

        if (state[3] == 0):
            k = range(1, 7);
        else:
            k = range(state[3], state[3] + 1);

        if state[0] == 0 or state[0] == self.number_lanes + 1:
            state[-1] = 0
        else:
            state[-1] = 1 if state[state[0]] == 0 else 0

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
        self.road = Road(number_lanes=self.number_lanes, road_length=self.road_length)
        self.is_collided = 0
        self.collisions = 0
        self.current_lane = 1
        self.car_collided = None
        return self._getState()

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def _getState(self):
        self._checkCollisions()
        state = self.road.getState()
        state[0] = self.current_lane
        state[-1] = self.is_collided
        return state

    def _checkCollisions(self):
        self.is_collided = 0
        for i in range(self.road.number_lanes):
            for car in self.road.lanes[i]:
                if self.car_collided != car and car.collides(self.current_lane):
                    self.car_collided = car
                    self.collisions = self.collisions + 1
                    self.is_collided = 1


class Road():
    time_length = 1  # 0.1
    lane_proportion = 0.9

    def __init__(self, number_lanes=3, road_length=ROAD_LENGTH):
        self.number_lanes = number_lanes
        self.road_length = road_length
        self.lanes = [[] for i in range(self.number_lanes)]
        self.spaces = self.number_lanes * self.road_length

    def seed(self, seed):
        np.random.seed(seed)

    def findGap(self, lane):
        pos = 0
        for x in self.lanes[lane]:
            if (x.pos <= self.road_length) and (x.pos > pos):
                pos = x.pos

        return self.road_length - int(pos)

    def makeProbableCar(self):
        for i in range(self.number_lanes):
            trial = np.random.random()
            if trial < self.lane_proportion:
                self.makeCar(i)

    def makeCar(self, lane):
        c = Car(lane)
        if len(self.lanes[lane]) > 0:
            last = self.lanes[lane][-1]
            if last.pos < 1:
                c.pos = last.pos - 1
        self.lanes[lane].append(c)

    def getState(self):
        return [-1, self.findGap(0), self.findGap(1), self.findGap(2), -1]

    def update(self):
        for i in range(len(self.lanes)):
            temp = []
            for x in self.lanes[i]:
                x.update(self.time_length)
                if x.pos < self.road_length:
                    temp.append(x)
            self.lanes[i] = temp[:]
        self.makeProbableCar()


class Car():
    maxSpeed = 1  # 25
    speeds = [1, 1, 1]  # [0.1, 0.15, 0.2]

    def __init__(self, lane):
        self.lane = lane
        self.speed = self.speeds[lane] * self.maxSpeed
        self.pos = 0

    def collides(self, otherLane):
        if int(self.pos) == ROAD_LENGTH - 1:
            if otherLane == self.lane:
                return True
            else:
                return False
        else:
            return False

    def update(self, time):
        self.pos = self.pos + (self.speed * time)
