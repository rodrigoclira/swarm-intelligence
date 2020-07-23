#https://github.com/DEAP/deap/blob/master/deap/benchmarks/__init__.py
#author: rcls@ecomp.poli.br

import numpy as np
import math
import operator
from functools import reduce
from deap.benchmarks import *

class Problem():
    def __init__(self, name, dimensions, pMax, pMin, vMax, vMin):
        self.name = name
        self.dim = dimensions
        self.pMax = pMax
        self.pMin = pMin
        self.vMax = vMax
        self.vMin = vMin

    def sample(self):
        return list(np.random.uniform(low=self.pMin, high=self.pMax, size=self.dim))

    def custom_sample(self):
        return list(np.repeat(self.pMin, repeats=self.dim) \
               + np.random.uniform(low=0, high=1, size=self.dim) *\
               np.repeat(self.pMax - self.pMin, repeats=self.dim))

    def evaluate(self, positions):
        raise NotImplementedError

class Sphere(Problem):
    def __init__(self, dim):
        Problem.__init__(self, "Esfera", dim, 100, -100, 100, -100)

    def evaluate(self, positions):
        total=0

        for position in positions:
            total += position ** 2
        return total

class Rastrign(Problem):
    def __init__(self, dim):
        Problem.__init__(self, "Rastrign", dim, 5.2, -5.2, 5.2, -5.2)

    def evaluate(self, positions):
        A = 10
        total = A * len(positions)
        for position in positions:
            total +=  position**2 - 10 * math.cos(2.0 * math.pi * position)

        return total

class Ackley(Problem):
    def __init__(self, dim):
        Problem.__init__(self, 'Ackley', dim,  32.0, -32.0,  32.0, -32.0)

    def evaluate(self, positions): #https://www.sfu.ca/~ssurjano/ackley.html
        N = len(positions)
        return - 20 * math.exp(-0.2*math.sqrt((1.0/N) * sum(x**2 for x in positions))) \
             - math.exp((1.0/N) * sum(math.cos(2*math.pi*x) for x in positions)) + 20 + math.e

class Rosembrock(Problem):
    def __init__(self, dim):
        Problem.__init__(self, 'Rosembrock', dim,  30.0, -30.0,  30.0, -30.0)

    def evaluate(self, positions):
        return sum(100 * (x * x - y)**2 + (1. - x)**2 \
                for x, y in zip(positions[:-1], positions[1:]))

class Griewank(Problem):
    def __init__(self, dim):
        Problem.__init__(self, 'Griewank', dim,  600.0, -600.0,  600.0, -600.0)

    def evaluate(self, positions):
        return 1.0/4000.0 * sum(x**2 for x in positions) - \
        reduce(operator.mul, (math.cos(x/math.sqrt(i+1.0)) for i, x in enumerate(positions)), 1) + 1

class Schwefel12(Problem):
    def __init__(self, dim):
        Problem.__init__(self, 'Schwefel 1.2', dim,  100.0, -100.0,  100.0, -100.0)

    def evaluate(self, positions):
        positions = np.array(positions)
        return np.sum([np.sum(positions[:i]) ** 2
                   for i in range(len(positions))])

if __name__ == '__main__':
    print (Sphere(3).evaluate([2,2,2]))
    print (Sphere(3).sample())
    print (Ackley(3).evaluate([0,0,0]))
