"""
Implements the fully observable gridworld environment with linear rewards.
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from gym import spaces, Env
from gym.utils import seeding

EPS = 0.00001

def categorical_sample(prob_n, np_random):
    """
    From Gym.DiscreteEnv
    Sample from categorical distribution
    Each row specifies class probabilities
    """
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    return (csprob_n > np_random.rand()).argmax()

ACTIONS =  tuple(range(4))
UP, RIGHT, DOWN, LEFT = ACTIONS

def gridworld_from_map(mapfile, t_p = [1,0,0,0,0], tabular=False, reward=[-0.01, 0.01, -0.01]):
    with open(mapfile) as f:
        map_str = f.read()
        array_map = np.array([list(i) for i in map_str.split('\n')])
    obstacle_map = (array_map == 'X')
    x = np.linspace(-2, -1, array_map.shape[0])
    r, d = np.meshgrid(x, x)

    #feature_map = np.random.normal(loc=0,scale=1,size=(array_map.shape[0], array_map.shape[1],3))
    feature_map = np.zeros((array_map.shape) + (3,))
    #feature_map = np.dstack([r,np.zeros(array_map.shape),d])
    #feature_map = feature_map - np.abs(feature_map.max())
    #feature_map = np.zeros(array_map.shape + (3,))
    feature_map[array_map == 'S'] += np.array([0,0,0])
    feature_map[array_map == 'g'] += np.array([0,70,0])
    feature_map[array_map == 'B'] += np.array([0,0,20])
    feature_map[array_map == 'R'] += np.array([20,0,0])
    feature_map[array_map == 'Y'] += np.array([20,20,0])
    feature_map[array_map == 'C'] += np.array([0,10,10])
    feature_map[array_map == 'M'] += np.array([10,0,10])
    feature_map[array_map == '*'] += np.array([0,800,0])
    feature_map[array_map == 'k'] += np.array([10000,0,0])
    feature_map[array_map == ' '] += np.array([0,0,0])

    termination_map = (array_map == '*') + (array_map == 'k')
    reward_map = (feature_map * np.array(reward)[np.newaxis,np.newaxis,:]).sum(axis=2)

    if tabular:
        nS = array_map.shape[0] * array_map.shape[1]
        i = np.eye(nS)
        i = i.reshape((array_map.shape[0], array_map.shape[1], nS))
        print(feature_map.shape, i.shape)
        feature_map = i

    start_state_enum = (array_map == 'S').argmax()
    start_state = np.unravel_index(start_state_enum, array_map.shape)
    start_state = feature_map[start_state]
    isd = ([np.hstack([[start_state_enum], start_state])], [1])

    return (array_map.shape, isd, t_p, obstacle_map, feature_map,
        termination_map, reward_map)

class GridWorld(Env):

    metadata = {'render.modes': ['human', 'ansi']}
    """

    isd: ([st],[ps]) Tuple of list of states, list of probabilities.

    """
    def __init__(self, shape, isd=None, transitions=[1,0,0,0,0], obstacle_map=None,
                 feature_map=None, termination_map=None, reward_map=None, gamma=0.99):
        nS = shape[0] * shape[1]

        X = shape[1]
        Y = shape[0]

        state_enum = np.arange(nS).reshape(shape)
        it = np.nditer(state_enum, flags=['multi_index'])

        P = {}

        try:
            termination_map[0, 0]
        except:
            print("Default Termination")
            termination_map = np.zeros(shape)
            termination_map[X - 1, Y - 1] = 1
        try:
            pass
            #feature_map.reshape(shape)
        except:
            print("Default Features")
            feature_map = state_enum
        try:
            pass
            #reward_map.reshape(shape)
        except:
            print("Default Reward")
            reward_map = np.ones(shape) * -0.1
            reward_map[X - 1, Y - 1] = 1
        try:
            obstacle_map.reshape(shape)
        except:
            print("Building with no obstacles")
            obstacle_map = np.zeros(shape)

        feature_map = np.dstack([state_enum,feature_map])

        try:
            isd[0][0]
            if abs(np.sum(isd[1]) - 1) > EPS:
                raise ValueError
        except:
            a,b,c = feature_map.shape
            isd = ([feature_map[i,j] for j in range(a) for i in range(b)],
                    [1/(a*b) for i in range(a*b)])

        while not it.finished:
            y, x = it.multi_index

            state = feature_map[y,x]

            #combine these someday
            next_states = [(state if y == 0 or obstacle_map[y-1, x] else feature_map[y - 1, x]),
                           (state if x == (X - 1) or obstacle_map[y, x+1] else feature_map[y, x+1]),
                           (state if y == (Y - 1) or obstacle_map[y+1, x] else feature_map[y+1, x]),
                           (state if x == 0 or obstacle_map[y, x - 1] else feature_map[y, x - 1]),
                           state]

            t = termination_map
            is_terminal = [
                (t[y, x] if y == 0 or obstacle_map[y-1, x] else t[y - 1, x]),
                (t[y, x] if x == (X - 1) or obstacle_map[y, x+1] else t[y, x+1]),
                (t[y, x] if y == (Y - 1) or obstacle_map[y+1, x] else t[y+1, x]),
                (t[y, x] if x == 0 or obstacle_map[y, x - 1] else t[y, x - 1]),
                t[y,x]
            ]

            r = reward_map
            next_reward = [
                (r[y, x] if y == 0 or obstacle_map[y-1, x] else r[y - 1, x]),
                (r[y, x] if x == (X - 1) or obstacle_map[y, x+1] else r[y, x+1]),
                (r[y, x] if y == (Y - 1) or obstacle_map[y+1, x] else r[y+1, x]),
                (r[y, x] if x == 0 or obstacle_map[y, x - 1] else r[y, x - 1]),
                r[y, x]
            ]

            P[tuple(state)] = {i : [] for i in ACTIONS}

            for i,action in enumerate(ACTIONS):
                p = np.hstack([np.roll(transitions[:-1], i),transitions[-1]])
                P[tuple(state)][action] = [(p[i], next_states[i], next_reward[i], is_terminal[i])
                                for i in range(len(next_states))]

            it.iternext()


        self.P = P
        self.isd = isd
        self.lastaction = None # for rendering
        self.nS = nS
        self.nA = len(ACTIONS)
        self.nF = feature_map.shape[2] - 1
        self.shape = shape

        self.gamma = gamma

        self.feature_map = feature_map
        self.termination_map = termination_map
        self.obstacle_map = obstacle_map
        self.reward_map = reward_map

        self.action_space = spaces.Discrete(self.nA)
        print(feature_map.shape)
        self.state_space = spaces.Box(low=0, high=1000, shape=(feature_map.shape[2] - 1,))
        self.observation_space = self.state_space
        self._seed()
        self._reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _reset(self):
        self.state = self.isd[0][categorical_sample(self.isd[1], self.np_random)]
        return self.state[1:]

    def _step(self, a):
        transitions = self.P[tuple(self.state)][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, state, r, d = transitions[i]
        self.state = state
        self.lastaction = a
        return (state[1:], r, d, {"prob" : p})

    def _render(self, mode='human', close=False):
        if close:
            return

        outfile = StringIO() if mode == 'ansi' else sys.stdout

        grid = np.arange(self.nS).reshape(self.shape)
        it = np.nditer(grid, flags=['multi_index'])
        while not it.finished:
            y, x = it.multi_index
            s = self.feature_map[y, x]

            if np.all(self.state == s):
                output = "@"
            elif self.termination_map[y, x]:
                output = "T"
            elif self.obstacle_map[y,x]:
                output = "X"
            else:
                output = ' '
                #sten = np.arange(self.nS).reshape(self.shape)[y, x]
                #output = " {}   ".format(sten)
                #output = output[:-(len(str(sten)) + 1)]

            #if x == 0:
                #output = output.lstrip()
            #if x == self.shape[1] - 1:
            #    output = output.rstrip()

            outfile.write(output)

            if x == self.shape[1] - 1:
                outfile.write("\n")

            it.iternext()


if __name__ == '__main__':
    g = gridworld_from_map('./gridworld_maps/two_rooms.map')
    g.render()
    print(g.step(LEFT))
    g.step(LEFT)
    g.render()
    """
    g = GridWorld((8,8))
    initial_state = g.reset()
    print(initial_state)
    g.render()
    print(g.step(LEFT))
    g.step(LEFT)
    g.step(LEFT)
    g.step(LEFT)
    g.step(LEFT)
    g.render()
    """



