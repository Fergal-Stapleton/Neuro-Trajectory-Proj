"""Entry point to evolving the neural network. Start here."""
from __future__ import print_function
from evolver import Evolver
from tqdm import tqdm
from load_data import *
import csv
import datetime
import time
import logging
import sys
import math
#import h5py


class GeneticAlgorithm:
    def __init__(self, path, params, model, data_set):
        self.path = path + '/genetic_algorithm'
        self.data_set = data_set
        self.params = params
        self.model = model
        self.population = 10
        self.generations = 5

        self.create_dirs()

    def create_dirs(self):
        os.makedirs(self.path)
        os.makedirs(self.path + '/models')
        os.makedirs(self.path + '/plots')
        os.makedirs(self.path + '/confusion_matrix')
        os.makedirs(self.path + '/conf_matrix_csv')
        os.makedirs(self.path + '/conf_matrix_details')

    def run(self, mo_type):
        print("***Evolving for %d generations with population size = %d***" % (self.generations, self.population))
        if mo_type == "naive":
            self.generate()
        elif mo_type == "nsga-ii":
            self.generate_nsga2()

    # This repetitious but I dont to re-train the model here, just save the Pareto selected networks
    # save_genomes and train_simplified functions are train_genomes function split into two parts
    def save_genomes(self, genomes, writer, i):
        logging.info("***train_networks(networks, dataset)***")
        pbar = tqdm(total=len(genomes))

        for genome in genomes:
            # FS: going to get objectives out here
            #genome.train(self.model, self.data_set, self.path, i)

            parameters = list()
            params_csv = list()

            for p in self.params:
                parameters.append(genome.geneparam[p])
                params_csv.append(str(genome.geneparam[p]))

            params_csv.append(str(i+1))
            params_csv.append(genome.accuracy)
            #self.x_err, self.x_max, self.y_err, self.y_max
            params_csv.append(genome.x_err)
            params_csv.append(genome.x_max)
            params_csv.append(genome.y_err)
            params_csv.append(genome.y_max)
            for k in range(len(genome.fitness_vector)):
                params_csv.append(genome.fitness_vector[k])
                if (math.isnan(genome.fitness_vector[k]) == True):
                    print(genome.fitness_vector[k])

            row = params_csv
            writer.writerow(row)
            pbar.update(1)

        pbar.close()

    def train_genomes(self, genomes, writer, i):
        logging.info("***train_networks(networks, dataset)***")
        pbar = tqdm(total=len(genomes))

        for genome in genomes:
            # FS: going to get objectives out here
            genome.train(self.model, self.data_set, self.path, i)

            parameters = list()
            params_csv = list()

            for p in self.params:
                parameters.append(genome.geneparam[p])
                params_csv.append(str(genome.geneparam[p]))

            params_csv.append(str(i+1))
            params_csv.append(genome.accuracy)
            #self.x_err, self.x_max, self.y_err, self.y_max
            params_csv.append(genome.x_err)
            params_csv.append(genome.x_max)
            params_csv.append(genome.y_err)
            params_csv.append(genome.y_max)
            for i in range(len(genome.fitness_vector)):
                params_csv.append(genome.fitness_vector[i])
                if (math.isnan(genome.fitness_vector[i]) == True):
                    print(genome.fitness_vector[i])

            row = params_csv
            writer.writerow(row)
            pbar.update(1)

        pbar.close()

    def train_simplified(self, genomes, writer, i):
        logging.info("***train_networks(networks, dataset)***")
        pbar = tqdm(total=len(genomes))

        for genome in genomes:
            # FS: going to get objectives out here
            genome.train_short(self.model, self.data_set, self.path, i)


        pbar.close()

    def generate(self):
        logging.info("***generate(generations, population, all_possible_genes, dataset)***")
        t_start = datetime.datetime.now()
        t = time.time()

        evolver = Evolver(self.params)
        genomes = evolver.create_population(self.population)

        print(" ...opening result.csv")
        ofile = open(self.path + '/result.csv', "w", newline='')
        writer = csv.writer(ofile, delimiter=',')

        table_head = list()
        for p in self.params:
             table_head.append(str(p))

        table_head.append("Gen")
        table_head.append("accuracy")
        table_head.append("x_err")
        table_head.append("x_max")
        table_head.append("y_err")
        table_head.append("y_max")
        if (genomes[0].fitness_vector == 3):
            table_head.append("obj1")
            table_head.append("obj2")
            table_head.append("obj3")
        elif (genomes[0].fitness_vector == 2):
            table_head.append("obj1")
            table_head.append("obj2")
        row = table_head
        writer.writerow(row)

        # Evolve the generation.
        for i in range(self.generations):
            logging.info("***Now in generation %d of %d***" % (i + 1, self.generations))
            self.print_genomes(genomes)

            # Train and get accuracy for networks/genomes.
            self.train_genomes(genomes, writer, i)

            #print("completed 1st gen")
            #sys.exit()

            # Get the average accuracy for this generation.
            average_accuracy = self.get_average_accuracy(genomes)

            # Print out the average accuracy each generation.
            logging.info("Generation average: %.2f%%" % (average_accuracy * 100))
            logging.info('-'*80)

            # Evolve, except on the last iteration.
            if i != self.generations - 1:
                genomes = evolver.evolve(genomes)

        # Sort our final population according to performance.
        genomes = sorted(genomes, key=lambda x: x.accuracy, reverse=True)

        # FS - Print out fitness vector here

        # Print out the top 5 networks/genomes.
        self.print_genomes(genomes[:5])

        ofile.close()
        total = time.time() - t
        m, s = divmod(total, 60)
        h, m = divmod(m, 60)
        d, h = divmod(h, 24)
        t_stop = datetime.datetime.now()
        file = open(self.path + '/total_time.txt', 'w')
        file.write('Start : ' + str(t_start) + '\n')
        file.write('Stop : ' + str(t_stop) + '\n')
        file.write('Total :' + "%d days, %d:%02d:%02d" % (d, h, m, s) + '\n')
        file.close()

    def generate_nsga2(self):
        logging.info("***generate(generations, population, all_possible_genes, dataset)***")
        t_start = datetime.datetime.now()
        t = time.time()

        evolver = Evolver(self.params)
        genomes = evolver.create_population(self.population)

        print(" ...opening result.csv")
        ofile = open(self.path + '/result.csv', "w", newline='')
        writer = csv.writer(ofile, delimiter=',')

        table_head = list()
        for p in self.params:
             table_head.append(str(p))

        table_head.append("Gen")
        table_head.append("accuracy")
        table_head.append("x_err")
        table_head.append("x_max")
        table_head.append("y_err")
        table_head.append("y_max")
        if (genomes[0].fitness_vector == 3):
            table_head.append("obj1")
            table_head.append("obj2")
            table_head.append("obj3")
        elif (genomes[0].fitness_vector == 2):
            table_head.append("obj1")
            table_head.append("obj2")
        row = table_head
        writer.writerow(row)

        # Dont want to save this model, but require initial fitness
        # a little bit repetitive but okay
        self.train_simplified(genomes, writer, 0)
        evolver.fast_nondominated_sort(genomes)
        evolver.calculate_crowding_distance(genomes)
        #sys.exit()

        # Evolve the generation.
        for i in range(self.generations):
            logging.info("***Now in generation %d of %d***" % (i + 1, self.generations))
            self.print_genomes(genomes)

            combined_pop = evolver.combine_pop(genomes)
            print(combined_pop)

            # Train and get accuracy for networks/genomes.

            self.train_simplified(combined_pop, writer, i)
            evolver.fast_nondominated_sort(genomes)

            new_population = []
            front_num = 0
            while len(new_population) + len(evolver.fronts[front_num]) <= self.population:
                evolver.calculate_crowding_distance(evolver.fronts[front_num])
                new_population.extend(evolver.fronts[front_num])
                front_num += 1

            evolver.calculate_crowding_distance(evolver.fronts[front_num])
            evolver.fronts[front_num].sort(key=lambda individual: individual.crowding_distance, reverse=True)

            new_population.extend(evolver.fronts[front_num][0:self.population - len(new_population)])

            genomes = new_population
            self.save_genomes(genomes, writer, i)
            #print("completed 1st gen")
            #sys.exit()

            # Get the average accuracy for this generation.
            average_accuracy = self.get_average_accuracy(genomes)

            # Print out the average accuracy each generation.
            logging.info("Generation average: %.2f%%" % (average_accuracy * 100))
            logging.info('-'*80)

            # Evolve, except on the last iteration.

        # Sort our final population according to performance.
        genomes = sorted(genomes, key=lambda x: x.accuracy, reverse=True)

        # FS - Print out fitness vector here

        # Print out the top 5 networks/genomes.
        self.print_genomes(genomes[:5])

        ofile.close()
        total = time.time() - t
        m, s = divmod(total, 60)
        h, m = divmod(m, 60)
        d, h = divmod(h, 24)
        t_stop = datetime.datetime.now()
        file = open(self.path + '/total_time.txt', 'w')
        file.write('Start : ' + str(t_start) + '\n')
        file.write('Stop : ' + str(t_stop) + '\n')
        file.write('Total :' + "%d days, %d:%02d:%02d" % (d, h, m, s) + '\n')
        file.close()

    @staticmethod
    def get_average_accuracy(genomes):
        total_accuracy = 0
        for genome in genomes:
            total_accuracy += genome.accuracy

        return total_accuracy / len(genomes)

    @staticmethod
    def print_genomes(genomes):
        logging.info('-'*80)
        for genome in genomes:
            genome.print_genome()
