
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


def main(features, interactions,max_evaluations = 1000,num_solution = 16):

    global feature_data
    global interactions_data
    global feature_length

    # Read data from .txt file
    feature_data = readTXT(features)
    interactions_data = readTXT(interactions)

    #Best ← {}
    best_solution = None

    # History of Best
    best_history = []

    # Length of feature
    feature_length = len(feature_data)

    # Initial solutions
    initial_solutions = initialize_population(feature_length,num_solution)

    num_evaluations = 0
    # Repeat
    while num_evaluations < max_evaluations:
        num_evaluations += 1

        # For each individual "value" ∈ initial_solutions do
        for value in initial_solutions:

            # AssessFitness(value)
            solution_value = assess_fitness(value)

            # if Best = {} or Fitness(value) > Fitness(Best) then
            if (best_solution == None) or solution_value > best_solution:

                #  Best ← value
                best_solution = solution_value
                best_value = value
                best_history.append(solution_value)

        # Print
        print("Best value of configuration in ", num_evaluations, " iteration: ", best_solution)

        #  Q ← {}
        Q = []

        # for popsize/2 times do
        for i in range(int(num_solution/2)):

            # Parent Pa ← SelectWithReplacement(P)
            Parent_A = select(initial_solutions)
            # Parent Pb ← SelectWithReplacement(P)
            Parent_B = select(initial_solutions)

            # Children Ca, Cb ← Crossover(Copy(Pa), Copy(Pb))
            Ca,Cb = crossover(copy(Parent_A),copy(Parent_B))

            # Q ← P ∪ {Mutate(Ca), Mutate(Cb)}
            tweak_Ca = tweak(Ca)
            tweak_Cb = tweak(Cb)
            Q.append(tweak_Ca)
            Q.append(tweak_Cb)

        # P ← Q
        initial_solutions = copy(Q)

    print("Best configuration: ",best_value)
    print("Best value: ",best_solution)
    print("History of Best value: ",best_history)
    pass

def optimize(population):
    # TODO: to implement
    pass


def initialize_population(length,num_solution):
    # P ← {}
    P_array = []
    # for popsize times do
    for i in range(num_solution):
        x = np.random.randint(2,size = length)
        # P ← P ∪ {new random individual}
        P_array.append(x)
    return P_array
    pass


def copy(solution):
    return deepcopy(solution)
    pass

# Bit-Flip Mutation
def tweak(solution):
    # p ← probability of flipping a bit
    p = 1/feature_length

    # !v ← boolean vector $v1, v2, ...vl% to be mutated
    v = solution

    # for i from 1 to l do
    for i in range(feature_length):
        # if p ≥ random number chosen uniformly from 0.0 to 1.0 inclusive then
        if p >= np.random.rand():
            # vi ← ¬(vi)
            if(v[i] == 0):
                v[i] = 1
            else:
                v[i] = 0
    return v
    pass


# Fitness-Proportionate Selection
def select(solution):
    # global !p ← population copied into a vector of individuals $p1, p2, ..., pl%
    P =  copy(solution)

    # global !f ← $f1, f2, ..., fl% fitnesses of individuals in !p in the same order as !p
    F = list()
    for value in P:
        F.append(abs(assess_fitness(value)))

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

# Uniform Crossover
def crossover(solution_a, solution_b):
    # p ← probability of swapping an index
    p = 1/feature_length
    a = solution_a
    b = solution_b
    for i in range(feature_length):
        if p >= np.random.rand():
            a[i],b[i] = b[i],a[i]
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

main("bdbc_feature.txt", "bdbc_interactions.txt")