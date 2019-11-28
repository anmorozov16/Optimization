import pandas as pd
import numpy as np


class Train:
    def __init__(self, input_data, y_data, model_y):
        self.input_data = np.array(input_data)
        self.model_y = model_y
        self.y_data = np.array(y_data)

    def target(self, opt_input, index):
        parameters = np.empty((1, len(opt_input) + len(self.input_data[0])))
        parameters[0, :len(self.input_data[0])] = self.input_data[index]
        parameters[0, len(self.input_data[0]):] = opt_input
        return self.model_y.predict(parameters)[0]


class Recommendations:
    def __init__(self, begin_optimization_data, train=None, len_steps=1., limits=[-np.inf, np.inf]):
        self.begin_optimization_data = begin_optimization_data
        self.opt_data = begin_optimization_data.copy()
        self.train = train
        self.parameter_index, self.index = 0, 0
        self.limits = limits
        self.nb_actions = 3
        self.last_action = -1
        self.sum_reward = 0.

        if type(len_steps) in (int, float):
            self.len_steps = [len_steps for _ in range(len(self.opt_data))]
        elif type(len_steps) in (np.array, list, tuple, pd.Series) and len(len_steps) != len(self.opt_data):
            raise BaseException('Expected step length {}, but was given {}'.format(len(self.opt_data),
                                                                                   len(len_steps)))
        else:
            self.len_steps = len_steps

    def reward(self):
        target = self.train.target(self.opt_data, self.index)
        return target - self.train.y_data[self.index]

    def observation(self, input_data=None):
        if input_data is None:
            input_data = self.train.input_data[self.index]
        return input_data, (self.parameter_index, self.opt_data[self.parameter_index])

    def _return(self, input_data, is_end=False):
        if input_data is None:
            reward = 0.
            if is_end:
                reward = self.reward()
                self.sum_reward += reward
                print('reward: {} sum reward: {}'.format(reward, self.sum_reward))
            return self.observation(), reward, self.index >= len(self.train.input_data) - 1, {}
        return self.observation(input_data), is_end

    def step(self, action, predict_data=None):
        if predict_data is None and self.train is None:
            raise BaseException('train.object is none')

        if (self.last_action, action) in ((1, 0), (0, 1)):  # если действия не меняются
            action = 2
        self.last_action = action

        if action == 0 and\
                self.limits[1] > self.opt_data[self.parameter_index] + self.len_steps[self.parameter_index]:
            self.opt_data[self.parameter_index] += self.len_steps[self.parameter_index]
        elif action == 1 and\
                self.limits[0] < self.opt_data[self.parameter_index] - self.len_steps[self.parameter_index]:
            self.opt_data[self.parameter_index] -= self.len_steps[self.parameter_index]
        else:
            self.parameter_index += 1
            if self.parameter_index == len(self.opt_data):
                self.parameter_index = 0
                self.index += 1
                _result = self._return(predict_data, is_end=True)
                self.opt_data = self.begin_optimization_data.copy()
                return _result
        return self._return(predict_data)

    def predict(self, model, input_data):
        obs = self.observation(input_data)
        result = False
        while not result:
            action = np.argmax(model.predict([np.array([[obs[0]]]),
                                              np.array([[obs[1]]])]))
            opt_param = self.opt_data.copy()
            obs, result = self.step(action, input_data)
        return opt_param

    def reset(self):
        self.parameter_index, self.index = 0, 0
        self.opt_data = self.begin_optimization_data.copy()
        self.sum_reward = 0.
        return self.observation()

