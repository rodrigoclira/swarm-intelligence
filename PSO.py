#!/usr/bin/python
#-*- coding: utf-8 -*-
from random import uniform
from math import pi
from math import cos
from math import sqrt
import numpy as np
from util.objectiveFunc import *
from util.helper import betterthan
import matplotlib.pyplot as plt

DEBUG = False

#Comunication topology
GBEST=1
LBEST=2

class Particle():
    def __init__(self, label, problem, c1, c2, w):

        if DEBUG:
            print ("Creating particle %d" % label)

        self.problem = problem
        self.position=[]
        self.velocity=[]
        self.fitness=0.0
        self.label = label
        self.pbest = []
        self.c1 = c1
        self.c2 = c2
        self.w = w
        #self.pbestFitness = float('inf') #minimization
        self.pbestFitness = 0 #maximization

        #inicializando
        self.position = self.problem.sample()
        self.velocity = [0]*self.problem.dim
        #while len(self.position) < self.problem.dimensions:
        #    value = uniform(self.problem.vMin, self.problem.vMax)
        #    self.velocity.append(value)

        #print ('Init', self.position)
        self.pbest = self.position[:]

    def calculate_fitness(self, positions):
        value = self.problem.evaluate(positions)

        #trabalhando como se fosse uma maximixação
        if value == 0:
            return float('inf')
        else:
            return 1/value


    def update_fitness(self):
        self.last_fitness = self.fitness
        self.fitness=self.calculate_fitness(self.position)

        #if self.fitness > self.pbestFitness:
        if betterthan(self.fitness, self.pbestFitness, maximization=True):
            self.pbestFitness = self.fitness
            self.pbest = self.position[:]

    def update_position(self):
        pos=0

        while pos < len(self.velocity):
            self.position[pos] += self.velocity[pos]
            if (self.position[pos] < self.problem.pMin):
                self.position[pos] = self.problem.pMin
                self.velocity[pos]= -self.velocity[pos]
            elif (self.position[pos] > self.problem.pMax):
                self.position[pos] = self.problem.pMax
                self.velocity[pos]= -self.velocity[pos]
            pos+=1


    def update_velocity(self, iteration, max_iteration, gBest, cleck):
        pos=0
        while pos<len(self.position):


            if cleck:
                inertialFactor = 1
                constriction_factor = self.constriction_factor(iteration)
            else:
                inertialFactor = self.w(iteration)
                constriction_factor = 1

            self.velocity[pos] = constriction_factor * (self.velocity[pos] * inertialFactor +  \
            self.c1(iteration) * uniform(0,1) * (self.pbest[pos] - self.position[pos]) + \
            self.c2(iteration) * uniform(0,1) * (gBest[pos] - self.position[pos]))


            if (self.velocity[pos] > self.problem.vMax):
                self.velocity[pos] = self.problem.vMax
            elif (self.velocity[pos] < self.problem.vMin):
                self.velocity[pos] = self.problem.vMin

            pos+=1

    def constriction_factor(self, it):
        """It was found that when rho < 4, the swarm would slowly
        "spiral" toward and around the best found solution in the
        search space with no guarantee of convergence, while for
        rho > 4 convergence would be quick and guaranteed. (Defining a Standard
        for Particle Swarm Optimization)"""

        rho = self.c1(it) + self.c2(it)
        if rho < 4:
            print(f"rho = {rho}")

        return (2 / abs(2 - rho - sqrt(rho**2 - 4 * rho)))

    def __str__(self):
        return "(%02d)\nVelocidade=%s\nPosicao=%s" % (self.label, self.velocity, self.position)

class PSO():
    def __init__(self,number_particles, comunication_topology, max_iteration, c1, c2, inertialFactor, clerck, problem):

        self._swarm = []
        self.problem = problem
        self.comunication_topology = comunication_topology
        self.max_iteration = max_iteration
        self.clerck = clerck
        self.c1 = c1
        self.c2 = c2
        self.w = inertialFactor
        #self.bestFitness = float('inf')  #minimization
        self.bestFitness = 0 #maximization
        self.gbestPosition = []

        if DEBUG:
            print ('\n\nInicializando o PSO com %s partículas' % (number_particles))

        for nparticle in range(number_particles):
            particle = Particle(nparticle, problem, c1, c2, self.w)

            particle.update_fitness()
            #if particle.fitness > self.bestFitness:
            if betterthan(particle.fitness , self.bestFitness, maximization=True):
                self.bestFitness = particle.fitness
                self.gbestPosition = particle.position[:]

            self._swarm.append(particle)

    def __lbest(self, particleA, particleB):
        #if particleA.fitness > particleB.fitness:
        if betterthan(particleA.fitness, particleB.fitness, maximization=True):
            return particleA.position[:]
        else:
            return particleB.position[:]

    def run(self):
        it = 0
        best_fitness = []
        while it < self.max_iteration:
            if DEBUG:
                print ("\nExecuting %d " % it)

            for label, particle in enumerate(self._swarm):

                if self.comunication_topology == LBEST:
                    #Melhor dos vizinhos
                    best=self.__lbest(self._swarm[(label - 1) % len(self._swarm)], self._swarm[(label + 1) % len(self._swarm)])
                elif self.comunication_topology == GBEST:
                    #Melhor de todos
                    best=self.gbestPosition[:]
                else:
                    exit(0)

                particle.update_velocity(it, self.max_iteration, best, self.clerck)
                particle.update_position()
                particle.update_fitness()

                #if particle.fitness > self.bestFitness:
                if betterthan(particle.fitness, self.bestFitness, maximization=True):
                    self.bestFitness = particle.fitness
                    self.gbestPosition = particle.position[:]

            best_fitness.append(self.bestFitness)

            if DEBUG:
                print ("####################################")
                print ('MELHOR FITNESS ',self.bestFitness )
                print ('MELHOR POSICAO ', self.gbestPosition)
                print ("####################################")

            it+=1

        return (best_fitness[-1], best_fitness, self.gbestPosition)


if __name__ == '__main__':
    sphere = Sphere(30)
    clerk = True
    #lambda it : 0.9 - it * 0.5/10000 #decaimento linear

    print ("Gbest - clerk - sphere")
    for  i in range(30):
        swarm = PSO(30, GBEST, 10000, lambda _: 2.05, lambda _: 2.05, lambda it : 0.9 - it * 0.5/10000, clerk, sphere)
        ut, bt, coef = swarm.run()
        print (1/ut)

    print ("Lbest - clerk - sphere")
    for  i in range(30):
        swarm = PSO(30, LBEST, 10000, lambda _: 2.05, lambda _: 2.05, lambda it : 0.9 - it * 0.5/10000, clerk, sphere)
        ut, bt, coef = swarm.run()
        print (1/ut)

