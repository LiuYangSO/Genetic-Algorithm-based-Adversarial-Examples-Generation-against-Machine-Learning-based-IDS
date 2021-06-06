
import graphviz
from numpy.ma import zeros
from sklearn.model_selection import cross_val_score

from labels import data_headers, attack_dict
from sklearn import tree
from sklearn import datasets,ensemble
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score

from labels import data_headers, attack_generation_labels
from Model4 import Model
import pandas as pd
import random
from tabulate import tabulate
import matplotlib.pyplot as plt
import numpy as np
from SA2 import SA2



class GSnew:

    def __init__(self, debug, mutation, attack, save_model):
        self.DEBUG = debug
        self.MUTATION_PERCENTAGE = mutation
        print('【1】 Cerate Models')
        self.model_IDS = Model(debug, attack, save_model)

        # create model for each method: D,R,S,P,K represent decisoin tree, randomforest, SVM, kneibourghs and pipeline
        self.model_D = self.model_IDS.generate_model('D')
        self.model_R = self.model_IDS.generate_model('R')
        self.model_S = self.model_IDS.generate_model('S')
        self.model_P = self.model_IDS.generate_model('P')
        self.model_K = self.model_IDS.generate_model('K')

        print(' 5 Models Created! ')

        self.model = self.model_D   #------> Could change the basic model here
        # here we choose decision tree as the main IDS to select the results from GA

        self.ATTACK = self.select_attack(attack)
        print('【2】 Generate Attack Samples with GA')
        print("Seed attack for algorithm run")
        print(self.ATTACK)
        self.Btest = [1]
        self.Atest = pd.DataFrame([self.ATTACK], columns=data_headers)

        # Get our models
        print("10-fold cross-validation：Modeling")

        #  Initialize variables
        self.non_attack
        self.num = 0
        self.max = self.fitness(self.model, self.ATTACK, self.non_attack)
        self.average = 0
        self.loop = 0
        self.I1 = [None]*len(self.ATTACK)
        self.I2 = [None]*len(self.ATTACK)
        self.AB1 = [0,0]
        self.geneA = []
        self.geneB = []
        self.A = 0
        self.B = 0
        self.nAA = 0
        self.nBB = 0
        self.nAB = 0
        self.nBA = 0
        self.AA = []
        self.BB = []
        self.AB = []
        self.BA = []
        self.abab= []
        self.cAB = []
        self.cAA = []
        self.cBA = []
        self.cBB = []
        self.attackall = []
        self.benignall = []
        self.tj1 = 0
        self.tjt1 = 0
        self.tj2= 0
        self.tj3 = 0
        self.tj4 = 0
        self.tj5 = 0
        self.tj6 = 0
        self.zong= 0
        self.Tj1 = []
        self.Tj2 = []
        self.Tj3 = []
        self.Tj4 = []
        self.Tj5 = []
        self.Tj6 = []
        self.Zong = []
        self.Tj0 = []
        self.mark(self.ATTACK)

        # select the features that can be changed:
        # core part of the features and the strings should not be changed
        for i, v in enumerate(self.ATTACK):
            if (type(v) is not str and self.validId(i)):
                # if attack_generation_labels[data_headers[i]][-1] <100:
                #     self.attackall.append(zeros(ceil(attack_generation_labels[data_headers[i]][-1]/100)))
                #     self.benignall.append(zeros(ceil(attack_generation_labels[data_headers[i]][-1]/100)))
                self.attackall.append(zeros(101))
                self.benignall.append(zeros(101))
                # else:
                #     self.attackall.append(zeros(100))
                #     self.benignall.append(zeros(100))
            else:
                self.attackall.append(0)
                self.benignall.append(0)

        test_df = pd.read_csv("Datasets/NSL-KDD/KDDTest+.txt", header=None, names=data_headers)
        # Pick a random attack of attack_type from here
        attacks = test_df.loc[test_df['attack'] == attack]
        print(len(attacks))
        # take the colum of 'attack' also we can know of 'iloc'
        bad_D = 0
        bad_R = 0
        bad_S = 0
        bad_P = 0
        bad_K = 0
        bad5 = 0
        # test the bad samples[for testing]
        for i in range(0,len(attacks)):
            aaa=(attacks.iloc[[i]])

            if self.evaluate_sample(self.model_D, aaa.values[0])[0] == 0:
                bad_D = bad_D +1
            if self.evaluate_sample(self.model_R, aaa.values[0])[0] == 0:
                bad_R = bad_R +1
            if self.evaluate_sample(self.model_S, aaa.values[0])[0] == 0:
                bad_S = bad_S +1
            if self.evaluate_sample(self.model_P, aaa.values[0])[0] == 0:
                bad_P = bad_P +1
            if self.evaluate_sample(self.model_K, aaa.values[0])[0] == 0:
                bad_K = bad_K +1
        if self.DEBUG:
            print('model:')
            print(bad_D)
            print('model1:')
            print(bad_R)
            print('model2:')
            print(bad_S)
            print('model3:')
            print(bad_P)
            print('model4:')
            print(bad_K)
            print('model5:')
            print(bad5)
        # check the samples


    def mark(self, sample):
        for i in range(len(sample)):
            self.geneA.append(0)
            self.geneB.append(0)
            self.AB.append(0)
            self.BB.append(0)
            self.BA.append(0)
            self.AA.append(0)
            self.abab.append([0])
            self.abab[i].append(0)
            self.abab[i].append(0)
            self.abab[i].append(0)


    def select_attack(self, attack_type):
        test_df = pd.read_csv("Datasets/NSL-KDD/KDDTest+.txt", header=None, names=data_headers)
        # Pick a random attack of attack_type from here
        attacks = test_df.loc[test_df['attack'] == attack_type]
        # take the colum of 'attack' also we can know of 'iloc'

        n =True
        while n==True:
            sample_attack = attacks.sample(1)
            if self.evaluate_sample1(self.model,sample_attack.values[0])[0] == 1:
                n = False
        # select an attack sample as the seed for GA,
        # but we should make sure that the classifier make the right classificatioin ast the first time

        non_attacks = test_df.loc[test_df['attack'] == 'normal']
        non_attack = non_attacks.sample(1)
        self.non_attack = non_attack.values[0]
        # selec a non attack sample as the father of the initial group

        return sample_attack.values[0]

    def getModel(self):
        return self.model  # give the model type

    def getSeedAttack(self):
        return self.ATTACK  # gieve the type of attack

    # Function to test if a feature variable is valid for the algorithm execution
    # is this a variable the alogrithm should be able to use
    def validId(self, idx):
        if (idx == 1 or idx == 2 or idx == 3 or idx == 6 or idx == 41 or idx == 42):
            return False
        else:
            return True

        # protocol type, service, flag, land and attack feature variables are the core of attacks, so we don't need to
        # change these parts

    # Function used to evaluate a single sample on the model
    def evaluate_sample(self, model, sample):

        sample_df = pd.DataFrame([sample], columns=data_headers)

        dropped = sample_df.drop(["unknown", "attack", "protocol_type", "service", "flag"], axis=1)
        # encoded_df = pd.get_dummies(dropped, columns=["protocol_type", "service", "flag"])
        pred = model.predict(dropped)
        sample_test = pd.DataFrame([sample], columns=data_headers)
        if pred[0] == 1:
            self.Btest.append(1)
        else:
            self.Btest.append(0)

        # self.Atest.apply(sample_test)
        self.Atest = pd.concat([self.Atest,sample_test])

        # clf.predict
        return (pred)

    def evaluate_sample1(self, model, sample):

        sample_df = pd.DataFrame([sample], columns=data_headers)

        dropped = sample_df.drop(["unknown", "attack", "protocol_type", "service", "flag"], axis=1)
        # encoded_df = pd.get_dummies(dropped, columns=["protocol_type", "service", "flag"])
        pred = model.predict(dropped)
        return (pred)

    # Function to calculate the deviation of a sample from some other sample (normally the initial seed sample)
    def deviation(self, initial_sample, sample):
        total_deviation = 0.0
        num = 0
        for i, v in enumerate(sample):
            # Only measure deviation on int values
            if (type(v) is not str and self.validId(i)):
                # Factor in the value range for the value

                max_value = attack_generation_labels[data_headers[i]][-1]
                this_deviation = 1 + (abs(v - initial_sample[i]) / max_value)  # calculate the deviatioin
                this_deviation = (abs(v - initial_sample[i]))  # calculate the deviatioin
                if initial_sample[i] == 0:
                    this_deviation = (abs(v - initial_sample[i]))
                else:
                    this_deviation = (abs(v - initial_sample[i]) / initial_sample[i])
                total_deviation += this_deviation
                num = num + 1
        # print(num)
        return total_deviation

    # Fitness Function
    def fitness(self, model, initial_sample, sample):
        attack_eval = self.evaluate_sample(model, sample)[0]
        _deviation = self.deviation(initial_sample, sample)

        if (attack_eval == 0):
            # Extra fitness for not being an attack
            # print(self.num)

            return 0.5 * (_deviation)
        else:

            return (0+_deviation)


    # Function to generate some mutation to produce genetic drift in the population
    def add_mutation(self, sample):
        for i, v in enumerate(sample):

            if (not self.validId(i)):
                # Skip protocol type, service, flag, land and attack feature variables
                continue

            # Mutate each gene or feature variable with 5% change
            rand = random.randint(0, 100)

            if (rand < self.MUTATION_PERCENTAGE and self.variableCanMutate(sample, i)):
                # Mutate by picking from a random index within allowable range
                max_range = attack_generation_labels[data_headers[i]][-1]
                index = random.randint(0, max_range)
                new_value = attack_generation_labels[data_headers[i]][index]
                sample[i] = new_value

        if self.loop == 0 and not self.evaluate_sample(self.model, sample):
            print(self.num)
            self.average = self.average + self.num
            self.loop = 1
            print('--------------first attack sample!-----------------------')

            # sleep(10)

        return sample

        # Function to implement more constraints on variable mutation

    # Used to ensure variable dependency is upheld for feature variables
    # Only ever called if the variable is going to mutate, so can do conditional updates here too
    def variableCanMutate(self, sample, featureVariable):
        # If num of root feature variable, user must be logged in as root first
        if (featureVariable == 15 and sample[13] == 0):
            return False

        # If the variable is diff srv rate and is mutating, its value must always be
        # 1 - same srv rate as these add to 1. So set it now
        if (featureVariable == 29):
            sample[featureVariable] = 1 - sample[28]
            return False
        return True

    def statisticGene(self,offspring,sample):
        for i, v in enumerate(sample):
            if offspring[i] != v:
                if self.evaluate_sample(self.model, sample)[0] == 1:
                    if self.evaluate_sample(self.model, offspring)[0] == 1: #AA
                        self.abab[i][2] = self.abab[i][2] +1
                    else:                                                  #AB
                        self.abab[i][0] = self.abab[i][0] + 1
                else:                                                            #BA
                    if self.evaluate_sample(self.model, offspring)[0] == 1:  # BA
                        self.abab[i][1] = self.abab[i][1] + 1
                    else:  # BB
                        self.abab[i][3] = self.abab[i][3] + 1
            else:
                continue

    # Function for generating an offspring given two samples
    def generate_offspring(self, sample1, sample2):
        offspring = [None] * len(sample1)
        for v in range(len(sample2)):
            if (v == 1 or v== 2 or v == 3):
                # For protocol type, service, and flag, we simply match the parents value
                offspring[v] = sample2[v]
                continue

            # Take genes from each parent.
            rand = random.randint(0, 1)
            if (rand):
                # Take from sample 1
                offspring[v] = sample1[v]
            else:
                offspring[v] = sample2[v]
        # Add mutation to the offspring
        # return offspring
        self.num = self.num + 1
        offspring = self.add_mutation(offspring)
        return offspring

    def generate_offspring_0(self, sample1, sample2):
        offspring = [None] * len(sample1)
        for i, v in enumerate(sample2):

            if (i == 1 or i == 2 or i == 3):
                # For protocol type, service, and flag, we simply match the parents value
                offspring[i] = sample2[i]
                continue
            if (not self.validId(i)):
                offspring[i] = sample2[i]
                # Skip protocol type, service, flag, land and attack feature variables
                continue
            # Take features from each parent.

            rand = random.randint(0, 1)
            if (rand):
                # Take from sample 1
                offspring[i] = sample1[i]
            else:
                offspring[i] = sample2[i]
        self.num = self.num + 1

        offspring = self.add_mutation(offspring)
        self.statisticGene(offspring,sample1)
        self.statisticGene(offspring,sample2)
        return offspring

    def key_extractor(self, sample):
        return sample['fitness']

    def sample_extractor(self, sample):
        return sample['sample']

    # Prints out information for  the current population
    def display_population_statistics(self, population):
        print("FITTEST SAMPLE: " + str(population[-1]['fitness']))
        print("WEAKEST SAMPLE: " + str(population[0]['fitness']))
        num_attacks = 0
        for sample in population:
            if (sample['attack']):
                num_attacks += 1
        print("NUMBER OF SAMPLE ATTACKS: " + str(num_attacks))
        print(" ")

    # Prints out the current population
    def display_population(self, population):
        # Output to a html table
        table = tabulate(population, tablefmt="html", headers=list(attack_generation_labels.keys()))
        table_file = open("final_samples.html", "w")
        table_file.write(table)
        table_file.close()

    def run(self, iterations, offspring_number, fittest_num):
        # Breed initial population
        print("Breeding Initial Population")
        print(self.evaluate_sample(self.model, self.ATTACK))
        population = []
        X = []
        Y = []
        count = 0
        for i in range(offspring_number):
            population.append(self.generate_offspring_0(self.ATTACK, self.ATTACK))

        for sample in population:
            if self.evaluate_sample(self.model, sample)[0] == 1:
                self.A = self.A +1
                for idx, val in enumerate(self.ATTACK):
                    if sample[idx] == val:
                        self.geneA[idx] = self.geneA[idx]+1
                for idx, val in enumerate(sample):
                    if (type(val) is not str and self.validId(idx)):
                        self.attackall[idx][int(100* val/attack_generation_labels[data_headers[idx]][-1])] = 1
            else:
                for idx, val in enumerate(sample):
                    if (type(val) is not str and self.validId(idx)):
                        self.benignall[idx][int(100 *val/attack_generation_labels[data_headers[idx]][-1])] = 1
                self.B = self.B +1
                for idx, val in enumerate(self.ATTACK):
                    if sample[idx] == val:
                        self.geneB[idx] = self.geneB[idx]+1


        print("Running Genetic Algorithm")
        for j in range(iterations):
            self.tjt = []
            self.tjz = []
            print("GENERATION: " + str(j))
            offspring = []

            self.tj1 = 0
            self.tjt1 = 0
            self.tj2 = 0
            self.tj3 = 0
            self.tj4 = 0
            self.tj5 = 0
            self.tj6 = 0
            self.zong = 0

            for index in range(offspring_number):
                parent1 = random.randint(0, len(population) - 1)
                parent2 = random.randint(0, len(population) - 1)

                offspring.append(self.generate_offspring(population[parent1], population[parent2]))
            # Place offspring in population
            population.extend(offspring)
            # Evaluate the fittest_num samples to go through to next population
            fittest_samples = []
            for sample in population:
                sample_fitness = self.fitness(self.model, self.ATTACK, sample)
                is_attack = self.evaluate_sample(self.model, sample)

                # whether the IDS classified this as an attack or not
                fittest_samples.append({'fitness': sample_fitness, 'sample': sample, 'attack': is_attack})
            fittest_samples.sort(key=self.key_extractor,reverse=True)  # key_extractor is the fitness
            # Trim population if too large
            population2 = []
            if (len(fittest_samples) > fittest_num):
                population = fittest_samples[len(fittest_samples) - fittest_num:]
            for sample in population:
                population2.append(sample['sample'])
            for samplea in population2:
                sample_df = pd.DataFrame([samplea], columns=data_headers)
                dropped = sample_df.drop(["unknown", "attack", "protocol_type", "service", "flag"], axis=1)
                pred = self.model_D.predict(dropped)
                if pred == 1:
                    self.tj1 = self.tj1+1
                pred = self.model_R.predict(dropped)
                if pred == 1:
                    self.tj2 = self.tj2+1
                pred = self.model_S.predict(dropped)
                if pred == 1:
                    self.tj3 = self.tj3 + 1
                pred = self.model_P.predict(dropped)
                if pred == 1:
                    self.tj4 = self.tj4 + 1
                pred = self.model_K.predict(dropped)
                if pred == 1:
                    self.tj5 = self.tj5 + 1
                self.zong = self.zong +1
            print(self.tj1/self.zong)
            print(self.tj2/self.zong)
            print(self.tj3/self.zong)
            print(self.tj4/self.zong)
            print(self.tj5/self.zong)
            print(self.tj6/self.zong)
            print(self.zong)

            self.Tj1.append(self.tj1/self.zong)
            self.Tj2.append(self.tj2/self.zong)
            self.Tj3.append(self.tj3/self.zong)
            self.Tj4.append(self.tj4/self.zong)
            self.Tj5.append(self.tj5/self.zong)
            self.Tj6.append(self.tj6/self.zong)
            self.Tj0.append((self.tj1/self.zong+self.tj2/self.zong+self.tj3/self.zong+self.tj5/self.zong+self.tj6/self.zong)/5)
            if len(self.Zong) >= 1:
                self.Zong.append(self.zong + self.Zong[-1])
            else:
                self.Zong.append(self.zong)
            sample_fitness2 = 0
            for sample2 in population:
                sample_fitness2 = sample2['fitness']
                X.append(sample_fitness2)
                count = count + 1
                Y.append(count)
            raw_population = population


            population = list(map(self.sample_extractor, population))

        plt.figure(1)
        plt.plot(self.Zong, self.Tj1,label = 'Decision Tree')
        plt.plot(self.Zong, self.Tj2,label = 'Random forest')
        plt.plot(self.Zong, self.Tj3,label = 'SVM')
        # plt.plot(self.Zong, self.Tj4)
        plt.plot(self.Zong, self.Tj4,label = 'LR')
        plt.plot(self.Zong, self.Tj5,label = 'Kneighbor')
        plt.legend()

        plt.xlabel('Offsprings')
        plt.ylabel('Escape Rate')

        plt.figure(2)
        plt.plot(self.Zong, self.Tj0, label='Average')
        plt.grid()
        plt.xlabel('Offsprings')
        plt.ylabel('Escape Rate')

        plt.show()


        print(self.evaluate_sample(self.model, self.ATTACK))

        only_benign = []
        num_benign = 0
        for sample in raw_population:
            if (sample[
                'attack'] == 0):
                only_benign.append(sample)
                num_benign = num_benign + 1
        print("Number of Benign Samples:" + str(num_benign))

        print("Most Fit Sample")
        # ---------------------------------------
        if len(only_benign)>0:
            print(
                only_benign[len(only_benign) - 1]['sample'])  # -----------------with the highest fitness, [-1] the last one
            print(only_benign[len(only_benign) - 1]['fitness'])

            best_sample = only_benign[len(only_benign) - 1]['sample']

            print(self.ATTACK)
            print("watch here")

            print(self.fitness(self.model, self.ATTACK, only_benign[len(only_benign) - 1]['sample']))
            print(self.fitness(self.model, self.ATTACK, self.ATTACK))
            train_df = self.Atest
            # pick the data that we want and then dropoff the label

            # TESTING -- DROP PROTOCOL TYPE SERVICE AND FLAG
            train_df = train_df.drop('protocol_type', axis=1)
            train_df = train_df.drop('service', axis=1)
            train_df = train_df.drop('flag', axis=1)
            clf = tree.DecisionTreeClassifier(max_depth=5)
            x = train_df.drop(["unknown", "attack"], axis=1)

            y = self.Btest
            clf = clf.fit(x, y)
            score = clf.score(x, y)
            print(score)
            print("good！！！！！！good ！！！！！")
            dot_data = tree.export_graphviz(clf, out_file = None, feature_names=x.columns,
                                            class_names=["Benign", "Attack"])
            graph = graphviz.Source(dot_data)

            graph.render()

        print("num=" + str(self.num))
        plt.figure(1)
        plt.plot(Y, X)

        print("this is the result!!!!!")
        plt.show()


        x_range = labels=(data_headers)
        # print(x_range)
        width = 0.2

        index_A = np.arange(len(x_range))  #
        index_B = index_A + width  #

        # bar figure
        plt.barh(index_A, width=self.geneA, height=width, color='b', label='A')
        plt.barh(index_B, width=self.geneB, height=width, color='r', label='B')
        plt.yticks(index_A, x_range)

        plt.grid(True)
        plt.legend()
        plt.show()

        geneall = [a + b for a, b in zip(self.geneA, self.geneB)]
        geneAinB =[a / b for a,b in zip(self.geneA, geneall)]

        plt.barh(index_A, width=geneAinB, height=width, color='b', label='A')
        plt.yticks(index_A, x_range)
        plt.grid(True)
        plt.legend()
        plt.show()


        index_AB = np.arange(len(x_range))  #
        index_BA = index_AB + width  #
        index_AA = index_BA + width  #
        index_BB = index_AA + width  #
        index_P = index_BB + width  #

        for i in range(len(self.ATTACK)):

            self.cAB.append(self.abab[i][0])
            self.cBA.append(self.abab[i][1])
            self.cAA.append(self.abab[i][2])
            self.cBB.append(self.abab[i][3])
        geneAB  = [a + b for a,b in zip(self.cAB, self.cBA)]
        geneA_B =  [abs(a - b) for a,b in zip(self.cAB, self.cBA)]
        geneALL = [a + b + c + d for a,b,c,d in zip(self.cAB,self.cBA,self.cAA,self.cBB)]
        geneALL=  [a + b for a,b in zip(self.cAB, self.cBA)]
        cp = []
        for a,b in zip(geneA_B, geneALL):
            if b==0:
                cp.append(0)
            else:
                #cp.append(100*(a / b))
                cp.append(100*a/b)

        plt.barh(index_AB, width=self.cAB, height=width, color='b', label='AB')
        plt.barh(index_BA, width=self.cBA, height=width, color='r', label='BA')
        plt.barh(index_AA, width=self.cAA, height=width, color='g', label='AA')
        plt.barh(index_BB, width=self.cBB, height=width, color='y', label='BB')
        plt.barh(index_P, width=cp, height=2*width, color='c', label='P')
        plt.yticks(index_A, x_range)
        plt.grid(True)
        plt.legend()
        plt.show()

        print(self.average)
        print('self avareage')
        # return self.average
        return (raw_population,1)
