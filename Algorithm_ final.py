from time import sleep
from turtle import pd

from GeneticAlgorithm import GeneticAlgorithm
import matplotlib.pyplot as plt

from SimulatedAnnealing import SimulateAnnealing
from labels import data_headers
import numpy as np
from AntColony import AntColony
from GS import GS
from GS2 import GS2
from G22 import GS22
from GA5 import GSnew
#from GS5_1 import GSnew
from SA2 import SA2
from SAnew import SimulateAnnealing


def calculatePopulationStatistics(population):
    # Fittest sample, weakest sample, number of attacks
    num_attacks = 0
    for sample in population:
        if (sample['attack']):
            num_attacks += 1
    return [population[-1]['fitness'], population[0]['fitness'], num_attacks]
    # --------------------calculate the numbers, return the num of attacks---------------------------


# Runs the algorithm 
def Algorithm():
    print("1.GeneticAlgorithm")
    algorithm = GeneticAlgorithm(False, 18, "nmap", True)
    model = algorithm.getModel()
    final_population = algorithm.run(10, 20, 20)
    print("Algorithm Execution Finished")

    # Show the final population statistics
    statistics = calculatePopulationStatistics(final_population)
    print(final_population)
    print(" ")
    print("Run Statistics")
    print("Most Fit Sample Fitness: " + str(statistics[0]))
    print("Least Fit Sample Fitness: " + str(statistics[1]))
    print("Number of Attack Samples: " + str(statistics[2]))

    # From the final population, only pick samples that are NOT attacks
    only_benign = []
    num_benign = 0
    for sample in final_population:
        if (sample['attack'] == 0):        # change here (sample['attack'][0] == 0): if (sample['attack'][0] == 0), there will be no parameters in only_benign
            only_benign.append(sample)
            num_benign = num_benign+1
    print("Number of Benign Samples:" + str(num_benign))

    print("Most Fit Sample")
    #---------------------------------------

    print(only_benign[len(only_benign)-1]['sample'])    #-----------------with the highest fitness, [-1] the last one
    print(only_benign[len(only_benign)-1]['fitness'])

    print("seed attack:")
    print(algorithm.ATTACK)

    print('end--------------------------------------')


def GSTest():
    print("1.GeneticAlgorithm")
    #algorithm = GS(False, 0, "nmap", True)
    #model = algorithm.getModel()
    all_benign =[]
    x= []
    most_fitness = []
    for m in range(0,1):
        x.append(m)
        algorithm = GSnew(False, 15, "teardrop", True)
        final_population = algorithm.run(5, 250, 30)
        print("Algorithm Execution Finished")

        # Show the final population statistics
        # print(final_population)
        # print('------------------------------------------------------------------------')
        # sleep(10)
        statistics = calculatePopulationStatistics(final_population)
        print(final_population)
        print(" ")
        print("Run Statistics")
        most_fitness.append(statistics[0])
        print("Most Fit Sample Fitness: " + str(statistics[0]))
        print("Least Fit Sample Fitness: " + str(statistics[1]))
        print("Number of Attack Samples: " + str(statistics[2]))

        # From the final population, only pick samples that are NOT attacks
        only_benign = []
        num_benign = 0
        for sample in final_population:
            if (sample['attack'] == 0):        # change here (sample['attack'][0] == 0): if (sample['attack'][0] == 0), there will be no parameters in only_benign
                only_benign.append(sample)
                num_benign = num_benign+1
        print("Number of Benign Samples:" + str(num_benign))

        print("Most Fit Sample")
        #---------------------------------------

        print(only_benign[len(only_benign)-1]['sample'])    #-----------------with the highest fitness, [-1] the last one
        print("Most Fit Sample fitness")
        print(only_benign[len(only_benign)-1]['fitness'])

        print("seed attack:")
        print(algorithm.ATTACK)
        all_benign.append(num_benign)
        print('now all benign:-------------------')
        print(all_benign)

    #print(only_benign[len(only_benign)-1])
    print(all_benign)
    plt.figure(1)
    plt.plot(x,most_fitness)
    plt.show()


def GSTest01():
    print("1.GeneticAlgorithm")
    all = 0
    #algorithm = GS(False, 0, "nmap", True)
    #model = algorithm.getModel()
    all_benign =[]
    for m in range(0,10):
        algorithm = GS(False, 0, "nmap", True)
        final_population = algorithm.run(1, 30, 30)
        all = all +final_population

        print(final_population/10)

def GSTest0():
    print("GeneticAlgorithm")
    all = 0
    numb = 0
    all_fitness = 0
    #algorithm = GS(False, 0, "nmap", True)
    #model = algorithm.getModel()
    all_benign =[]
    all_all = []
    all_num = []
    all_F1 = []
    x = []
    best_F = []
    for n in range(0,10,1):
        for m in range(0,1):
            algorithm = GS2(False, 10, "teardrop", True)
            final_population,num_b,all_fitness1,best_fitness = algorithm.run(1, 100, 30)
            # print("============================here ===================================")
            # print(final_population)
            # print(num_b)
            all = all +final_population
            numb = numb+num_b
            all_fitness = all_fitness+all_fitness1
            best_F.append(best_fitness)

        # print("============================here ===================================")
        # print(all/10)
        # print(numb/300)
        all_all.append(all/10)
        all_num.append(numb/300)
        if numb >0:
            all_F1.append(all_fitness/numb)
        else:
            all_F1.append(0)
        x.append(n)
        all = 0
        numb = 0

#     #-----------------------------------------------------------------------------------
#     plt.figure(3)
#     plt.plot(x, best_F)
#     plt.show()
# #-----------------------------------------------
    print("2.GeneticAlgorithm")
    all = 0
    numb = 0
    all_fitness = 0
    #algorithm = GS(False, 0, "nmap", True)
    #model = algorithm.getModel()
    all_benign =[]
    all_all2 = []
    all_num2 = []
    all_F2 = []
    x = []
    for n in range(0,10,1):
        for m in range(0,2):
            algorithm = GS22(False, 10, "teardrop", True)
            final_population,num_b,all_fitness2 = algorithm.run(1, 30, 30)
            all = all +final_population
            numb = numb+num_b
            all_fitness = all_fitness + all_fitness2
        # print(all/10)
        # print(numb/300)
        all_all2.append(all/10)
        all_num2.append(numb/300)
        if numb >0:
            all_F2.append(all_fitness/numb)
        else:
            all_F2.append(0)
        x.append(n)
        all = 0
        numb = 0
    plt.figure(1)
    plt.plot(x, all_all)
    plt.plot(x, all_all2)
    plt.figure(2)
    plt.plot(x, all_num)
    plt.plot(x, all_num2)
    plt.figure(3)
    plt.plot(x, all_F1)
    plt.plot(x, all_F2)

    plt.show()

def GSTestnew():
    print("Genetic Algorithm based Adversarial Examplesm")  #-----> START
    all = 0
    numb = 0
    #algorithm = GS(False, 0, "nmap", True)
    #model = algorithm.getModel()
    all_benign =[]
    all_all = []
    all_num = []
    x = []
    print('attack type: Smurf')
    algorithm = GSnew(False, 15, "smurf", False) # debug:False, mutation:15%, attack type: smurf, Save model: False

    only_benign = []
    num_benign = 0
    final_population,num_b = algorithm.run(10, 100, 60)
    for sample in final_population:
        if (sample['attack'] == 0):
            only_benign.append(sample)
            num_benign = num_benign + 1
    print("Number of Benign Samples:" + str(num_benign))

    print("Most Fit Sample")
    # ---------------------------------------

    print(only_benign[len(only_benign) - 1]['sample'])  # -----------------with the highest fitness, [-1] the last one
    print(only_benign[len(only_benign) - 1]['fitness'])

    print("seed attack:")
    print(algorithm.ATTACK)

GSTestnew()
# GSTest()
# GSTest0()
