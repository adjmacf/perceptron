import numpy as np
import matplotlib.pyplot as plt
from neuron import *
import time

if __name__ == "__main__":
    start = time.time()
    p = Neuron()

    plt.plot([0,1000],[0,1000])

    # train the perceptron on many random points
    points = []
    for i in range(200000):
        points.append((random.randint(0,1000), random.randint(0,1000)))

        x = points[i][0]
        y = points[i][1]

        p.train(points[i], int(y - x > 0) )

        #plt.scatter(x, y, c="g")

    # check perceptron on untrained data
    for i in range(500):
        x = random.randint(0,1000)
        y = random.randint(0,1000)

        if p.guess([x, y]):
            plt.scatter(x, y, c="b")
        else:
            plt.scatter(x, y, c="r")


    # plot where the neuron thinks the line is
    plt.plot([0, p.weights[1] * 1000 / p.weights[1]], [0, p.weights[0] * -1000 / p.weights[1]], c="g")
    print(p.weights)

    plt.show()

    end = time.time()
    print(end - start)
