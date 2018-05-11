#!python3
import sys
import numpy as np
import random
from copy import deepcopy
import itertools

# Seperate a array into "index" array and "value" array
def seperate_array(initial_array):
    index_array = []
    value_array = []
    for i in initial_array:
        index_array.append(i[0])
        value_array.append(i[1])
    return index_array,value_array


def main(features, interactions,max_evaluations):

    global feature_data
    global interactions_data
    global num_solution
    global best_solution
    global best_value
    global num_evaluations
    global feature_length

    # Read data from .txt file
    feature_data = readTXT(features)
    interactions_data = readTXT(interactions)

    # 1: popsize ← desired population size
    num_solution = 8

    # 5: Best ← ✷
    best_solution = None

    # Length of feature
    feature_length = len(feature_data)

    # Initial solutions
    initial_solutions = initialize_population(feature_length,num_solution)

    num_evaluations = 0
    # 6: repeat
    while num_evaluations < max_evaluations:
        num_evaluations += 1

        # 7: for each individual Pi ∈ P do
        for value in initial_solutions:

            # 8: AssessFitness(Pi)
            solution_value = assess_fitness(value)

            # 9: if Best = ✷ or Fitness(Pi) > Fitness(Best) then
            if (best_solution == None) or solution_value > best_solution:

                # 10: Best ← Pi
                best_solution = solution_value
                best_value = value

        # Print
        print("Best value of configuration in ", num_evaluations, " iteration: ", best_solution)

        # 11: Q ← {}
        Q = []

        # 12: for popsize/2 times do
        for i in range(int(num_solution/2)):

            # 13: Parent Pa ← SelectWithReplacement(P)
            Parent_A = select(initial_solutions)
            # 14: Parent Pb ← SelectWithReplacement(P)
            Parent_B = select(initial_solutions)

            # 15: Children Ca, Cb ← Crossover(Copy(Pa), Copy(Pb))
            Ca,Cb = crossover(deepcopy(Parent_A),deepcopy(Parent_B))

            # 16: Q ← P ∪ {Mutate(Ca), Mutate(Cb)}
            tweak_Ca = tweak(Ca)
            tweak_Cb = tweak(Cb)
            Q.append(tweak_Ca)
            Q.append(tweak_Cb)

        # 17: P ← Q
        initial_solutions = deepcopy(Q)

    print("Best configuration: ",best_value)
    print("Best value: ",best_solution)
    pass

def optimize(population):
    # TODO: to implement
    pass


def initialize_population(length,num_solution):
    # 2: P ← {}
    P_array = []
    # 3: for popsize times do
    for i in range(num_solution):
        x = np.random.randint(2,size = length)
        # 4: P ← P ∪ {new random individual}
        P_array.append(x)
    return P_array
    pass


# def copy(solution):
#     x = random.randrange(num_solution)
#     return solution[x]
#     pass

# Algorithm 22 Bit-Flip Mutation
def tweak(solution):
    # 1: p ← probability of flipping a bit
    p = 1/feature_length

    # 2: !v ← boolean vector $v1, v2, ...vl% to be mutated
    v = solution

    # 3: for i from 1 to l do
    for i in range(feature_length):
        # 4: if p ≥ random number chosen uniformly from 0.0 to 1.0 inclusive then
        if p >= np.random.rand():
            # 5: vi ← ¬(vi)
            if(v[i] == 0):
                v[i] = 1
            else:
                v[i] = 0
    # 6: return !v
    return v
    pass


# Algorithm 30 Fitness-Proportionate Selection
def select(solution):
    # 2: global !p ← population copied into a vector of individuals $p1, p2, ..., pl%
    P =  deepcopy(solution)

    # 3: global !f ← $f1, f2, ..., fl% fitnesses of individuals in !p in the same order as !p
    F = list()
    for value in P:
        F.append(assess_fitness(value))

    F = np.cumsum(F, dtype=float)
    random_value = random.uniform(0,F[-1])

    # Selected Solution
    for i in range(1,len(F)):
        index = 0
        if F[i-1] < random_value and random_value <= F[i]:
            index = i
            break

    return P[index]
    pass

# Algorithm 25 Uniform Crossover
def crossover(solution_a, solution_b):
    # 1: p ← probability of swapping an index
    p = 1/feature_length
    a = solution_a
    b = solution_b
    for i in range(feature_length):
        if p >= np.random.rand():
            temp = deepcopy(a[i])
            a[i] = deepcopy(b[i])
            b[i] = deepcopy(temp)
    return a,b
    pass


def assess_fitness(solution):
    # Fitness value
    fitness_point = 0
    set_configuration = []

    for index,value in enumerate(solution):
        if(value == 1):
            fitness_point += feature_data[index][-1]
            set_configuration.append(feature_data[index][0])
    index_interactions, value_interactions = seperate_array(interactions_data)
    set_configuration = list(itertools.chain(*set_configuration))
    for index, value in enumerate(index_interactions):
        result = all(item in set_configuration for item in value)
        if(result):
            fitness_point += value_interactions[index]

    return fitness_point

def readTXT(path):
    result = []
    with open(path) as f:
        lines = f.readlines()
    for line in lines:
        name = line.split(" ")[0][:-1]
        names = name.split("#")
        value = float(line.split(" ")[1].strip())
        configuration = [names, value]
        result.append(configuration)
    return result


# if __name__ == "__main__":
#     # input scheme: run_genetic_alg.py model_features.txt model_interactions.txt
#     if len(sys.argv) != 3:
#         print("Not a valid input! Please use:" + \
#         "python3 run_genetic_alg.py model_features.txt model_interactions.txt")
#         sys.exit(0)
#     features = readTXT(sys.argv[1])
#     interactions = readTXT(sys.argv[2])
#     main(features, interactions)

main("bdbc_feature.txt", "bdbc_interactions.txt",1000)