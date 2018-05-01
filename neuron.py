import random
import numpy as np

# learning rate
lr = 0.1

def sign(x):
    return int(x > 0)

class Neuron:

    def __init__(self, num_weights=2):
        # randomly choose weights
        self.weights = np.array([random.random()*2-1 for x in range(num_weights)])

    def guess(self, inputs):
        sum = 0
        for i in range(len(self.weights)):
            sum += inputs[i] * self.weights[i]

        return sign(sum)

    def train(self, inputs, target):
        guess1 = self.guess(inputs)
        error = target - guess1

        for i in range(len(self.weights)):
            self.weights[i] += error * (inputs[i] / sum(inputs)) * lr
