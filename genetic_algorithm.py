"""Entry point to evolving the neural network. Start here."""
from __future__ import print_function
from evolver import Evolver
from evolver_moead import Evolver_moead
from evolver_moead import Evolver_moead_gra
from tqdm import tqdm
from load_data import *
from copy import deepcopy
import csv
import datetime
import time
import logging
import sys
import math
from itertools import compress
#import h5py


class GeneticAlgorithm:
    def __init__(self, path, params, model, data_set, run_num):
        self.path = path + '/genetic_algorithm'
        self.data_set = data_set
        self.params = params
        self.model = model
        self.total_run = 5
        self.population = 10
        self.generations = 2
        self.decomp = 0
        self.delta_T = 3
        self.run_n = run_num
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
        if mo_type == "naive-tournament-select":
            self.generate()
        elif mo_type == "nsga-ii":
            self.generate_nsga2()
        elif mo_type == "moead":
            self.generate_moead(mo_type)
        elif mo_type == "moead_gra":
            self.generate_moead_gra(mo_type)

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
            params_csv.append(str(genome.u_ID))
            params_csv.append(genome.accuracy)
            params_csv.append(genome.score)
            #params_csv.append(genome.history)
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
            genome.train(self.model, self.data_set, self.path, i, self.generations, self.run_n)

            parameters = list()
            params_csv = list()

            for p in self.params:
                parameters.append(genome.geneparam[p])
                params_csv.append(str(genome.geneparam[p]))

            params_csv.append(str(i+1))
            params_csv.append(genome.accuracy)
            params_csv.append(genome.score)
            #params_csv.append(genome.history)
            params_csv.append(genome.u_ID)
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

    def train_simplified(self, genomes, writer, i):
        logging.info("***train_networks(networks, dataset)***")
        pbar = tqdm(total=len(genomes))

        for pop_index, genome in enumerate(genomes):
            # FS: going to get objectives out here
            genome.train_short(self.model, self.data_set, self.path, i, self.generations, self.run_n, pop_index)


        pbar.close()

    def train_simplified_gra(self, genomes, writer, i, bin_mask):
        logging.info("***train_networks(networks, dataset)***")
        pbar = tqdm(total=len(genomes))

        inc = 0
        for pop_index, genome in enumerate(genomes):
            # FS: going to get objectives out here
            if bin_mask[inc] == 1:
                genome.train_short(self.model, self.data_set, self.path, i, self.generations, self.run_n, pop_index)
            inc = inc + 1

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
        table_head.append("ID")
        table_head.append("accuracy")
        table_head.append("score")
        #table_head.append("history")
        table_head.append("x_err")
        table_head.append("x_max")
        table_head.append("y_err")
        table_head.append("y_max")
        if (len(genomes[0].fitness_vector) == 3):
            table_head.append("obj1")
            table_head.append("obj2")
            table_head.append("obj3")
        elif (len(genomes[0].fitness_vector) == 2):
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
        table_head.append("ID")
        table_head.append("accuracy")
        table_head.append("score")
        table_head.append("x_err")
        table_head.append("x_max")
        table_head.append("y_err")
        table_head.append("y_max")
        if (len(genomes[0].fitness_vector)== 3):
            table_head.append("obj1")
            table_head.append("obj2")
            table_head.append("obj3")
        elif (len(genomes[0].fitness_vector) == 2):
            table_head.append("obj1")
            table_head.append("obj2")
        row = table_head
        writer.writerow(row)

        # Dont want to save this model, but require initial fitness
        # a little bit repetitive but okay
        self.train_simplified(genomes, writer, 0)
        for genome in genomes:
            print('geneome before')
            print(genome.fitness_vector)
        evolver.fast_nondominated_sort(genomes)
        for genome in genomes:
            print('geneome after')
            print(genome.fitness_vector)
        evolver.calculate_crowding_distance(genomes)
        #sys.exit()

        # Evolve the generation.
        for i in range(self.generations):
            logging.info("***Now in generation %d of %d***" % (i + 1, self.generations))
            self.print_genomes(genomes)

            combined_pop = evolver.combine_pop(genomes)
            if len(combined_pop) != len(genomes):
                print('combined_pop is not the correct size')

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
            #self.model_genome, self.model_filepath
            #if(i == int(self.generations - 1)):
            #    for genome in genomes:
            #        genome.model.save(filepath = str(path) + '/models/model_' + str(file_name) + '_gen_' + str(i) + '_run_' + str(run_n) + '_' + str(self.u_ID) + '.h5')

            # To save space delete all models not in P+1
            genome_id_list = []
            for genome in genomes:
                genome_id_list.append(genome.genome_filename)
            for filename in os.listdir(str(self.path) + '/models/'):
                print(filename)
                if filename not in genome_id_list:
                    os.remove(str(self.path) + '/models/' + filename)
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


    def generate_moead(self, mo_type):
        logging.info("***generate(generations, population, all_possible_genes, dataset)***")
        t_start = datetime.datetime.now()
        t = time.time()

        evolver = Evolver_moead(self.params)
        genomes = evolver.create_population(self.population)
        extArchivePop = []

        print(" ...opening result.csv")
        ofile = open(self.path + '/result.csv', "w", newline='')
        writer = csv.writer(ofile, delimiter=',')

        table_head = list()
        for p in self.params:
             table_head.append(str(p))

        table_head.append("Gen")
        table_head.append("ID")
        table_head.append("accuracy")
        table_head.append("score")
        table_head.append("x_err")
        table_head.append("x_max")
        table_head.append("y_err")
        table_head.append("y_max")
        if (len(genomes[0].fitness_vector)== 3):
            table_head.append("obj1")
            table_head.append("obj2")
            table_head.append("obj3")
        elif (len(genomes[0].fitness_vector) == 2):
            table_head.append("obj1")
            table_head.append("obj2")
        row = table_head
        writer.writerow(row)

        myFitParent = [[None for x in range(len(genomes[0].fitness_vector))] for y in range(self.population)]
        myFitOffspring = [[None for x in range(len(genomes[0].fitness_vector))] for y in range(self.population)]
        # since genome is object will just use counter to keep track
        self.train_simplified(genomes, writer, 0)
        for genome in genomes:
            print('geneome before')
            print(genome.fitness_vector)
        for count, genome in enumerate(genomes):
            myFitParent[count] = genome.fitness_vector

        # Initialize MOEAD
        evolver.initialize(myFitParent)

        # Evolve the generation.
        for i in range(self.generations):
            new_population = [None]*int(self.population)
            # Crossover and Mutation
            # Select IDs for parents, (This is how it is done in our Java code and works quite nice for problem solving ...etc)

            for indivs in range(0, self.population, 2):
                parent1_id, parent2_id = evolver.returnParentsSelection(indivs)
                offspring1, offspring2 = evolver.applyCrossover(genomes[parent1_id], genomes[parent2_id])
                new_population[indivs] = offspring1
                if (indivs+1 < self.population):
                    new_population[indivs+1] = offspring2

            #test new pop and original are same length
            #assert len(new_population) == len(genomes), "Population sizes differ between offspring and parent"
            new_population = evolver.applyMutation(new_population)
            # new_population
            # myFitOffspring

            self.train_simplified(new_population, writer, i)

            for count2, genome in enumerate(new_population):
                myFitOffspring[count2] = genome.fitness_vector

            # Apply decomp approach - Note in Java code this falls under the generational() method in main class file
            evolver.solve(i, genomes, new_population, myFitParent, myFitOffspring, extArchivePop, mo_type)
            genomes = deepcopy(evolver.parent_pop)
            #new_population = evolver.offspring_pop
            myFitParent = deepcopy(evolver.parent_fit)
            #myFitOffspring = evolver.offspring_fit
            extArchivePop = deepcopy(evolver.extPop)

            # Remove duplicates and dominated solutions from extPop
            # TODO
            solutionsTuple = [(evolver.fitnessMO(genome), genome) for genome in extArchivePop ]
            #print(solutionsTuple)
            solutions = [x[0] for x in solutionsTuple]

            if len(genome.fitness_vector) == 3:
                obj1 = []
                obj2 = []
                obj3 = []
                for h in range(len(solutions)):
                    obj1.append(solutions[h][0])
                    obj2.append(solutions[h][1])
                    obj3.append(solutions[h][2])
                costs = np.column_stack((np.array(obj1), np.array(obj2), np.array(obj3)))
            elif len(genome.fitness_vector) == 2:
                obj1 = []
                obj2 = []
                for h in range(len(solutions)):
                    obj1.append(solutions[h][0])
                    obj2.append(solutions[h][1])
                costs = np.column_stack((np.array(obj1), np.array(obj2)))
            bool_non_dom_sol_df = evolver.is_pareto_efficient_simple(costs)

            extArchivePop = list(compress(extArchivePop, bool_non_dom_sol_df))

            genome_id_list = []
            for genome in extArchivePop:
                genome_id_list.append(genome.genome_filename)
            for filename in os.listdir(str(self.path) + '/models/'):
                print(filename)
                if filename not in genome_id_list:
                    os.remove(str(self.path) + '/models/' + filename)
            #self.save_genomes(genomes, writer, i)

            self.save_genomes(extArchivePop, writer, i)

            # Get the average accuracy for this generation.
            average_accuracy = self.get_average_accuracy(extArchivePop)

            # Print out the average accuracy each generation.
            logging.info("Generation average: %.2f%%" % (average_accuracy * 100))
            logging.info('-'*80)

            # Evolve, except on the last iteration.

        # Sort our final population according to performance.
        genomes = sorted(extArchivePop, key=lambda x: x.accuracy, reverse=True)

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



    def generate_moead_gra(self, mo_type):
        # printout various arrays for testing purposes
        printout = True

        logging.info("***generate(generations, population, all_possible_genes, dataset)***")
        t_start = datetime.datetime.now()
        t = time.time()

        evolver = Evolver_moead_gra(self.params)
        genomes = evolver.create_population(self.population)
        extArchivePop = []

        print(" ...opening result.csv")
        ofile = open(self.path + '/result.csv', "w", newline='')
        writer = csv.writer(ofile, delimiter=',')

        table_head = list()
        for p in self.params:
             table_head.append(str(p))

        table_head.append("Gen")
        table_head.append("ID")
        table_head.append("accuracy")
        table_head.append("score")
        table_head.append("x_err")
        table_head.append("x_max")
        table_head.append("y_err")
        table_head.append("y_max")
        if (len(genomes[0].fitness_vector)== 3):
            table_head.append("obj1")
            table_head.append("obj2")
            table_head.append("obj3")
        elif (len(genomes[0].fitness_vector) == 2):
            table_head.append("obj1")
            table_head.append("obj2")
        row = table_head
        writer.writerow(row)

        myFitParent = [[None for x in range(len(genomes[0].fitness_vector))] for y in range(self.population)]
        myFitOffspring = [[None for x in range(len(genomes[0].fitness_vector))] for y in range(self.population)]
        # since genome is object will just use counter to keep track
        self.train_simplified(genomes, writer, 0)

        #genome_inc = 0
        for genome in genomes:
            print('genome before')
            print(genome.fitness_vector)
        for count, genome in enumerate(genomes):
            myFitParent[count] = genome.fitness_vector
        print(myFitParent)
        #sys.exit()
        # Initialize MOEAD
        evolver.initialize(myFitParent)

        # Evolver child class doesnt have access to pop size
        evolver.util = [0.0]*int(self.population)

        # Going to store the fitness history in a list so that we can call for our utility function
        g_hist = []
        g_hist.append(deepcopy(evolver.util))

        poi = [0.5]*int(self.population)
        u = [None]*int(self.population)

        # Evolve the generation.
        max_updates = int(self.generations * self.population)
        update_counter_eval = int(self.population) # We first fully evaluate the population
        update_counter_utility = 0
        i = 0
        bin_archive = []
        while (update_counter_eval < max_updates):
            print('')
            print('Gen number:')
            print(i)
            print('')
            new_population = [None]*int(self.population)
            temp_population = [None]*int(self.population)
            bin_mask = [0]*int(self.population)

            # Crossover and Mutation
            # Select IDs for parents, (This is how it is done in our Java code and works quite nice for problem solving ...etc)

            for indivs in range(0, self.population, 2):

                #if randdom
                parent1_id, parent2_id = evolver.returnParentsSelection(indivs)
                offspring1, offspring2 = evolver.applyOnePointCrossover(genomes[parent1_id], genomes[parent2_id])
                new_population[indivs] = offspring1
                if (indivs+1 < self.population):
                    new_population[indivs+1] = offspring2

            #test new pop and original are same length
            #assert len(new_population) == len(genomes), "Population sizes differ between offspring and parent"
            new_population = evolver.applyMutation(new_population)
            # new_population
            # myFitOffspring

            # HERE
            # Calculate probability
            # output: poi
            rand = random.random()
            # The simplest approach is to generate a temp pop and only evaluate
            #print(genomes)
            temp_population = deepcopy(genomes)
            #print(temp_population)
            # self.delta_T = 3

            for j in range(self.population):
                if rand <= poi[j]:
                    bin_mask[j] = 1
                    temp_population[j] = new_population[j]
            #print(temp_population)



            update_counter_eval = update_counter_eval + int(sum(bin_mask))
            bin_archive.append(deepcopy(bin_mask))

            # TEST:
            if printout == True:
                print("------------------------------------------------")
                print("")
                print("Utility aggregation function - u ")
                print(u)
                print("")
                print("Probabilty of improvement - poi ")
                print(poi)
                print("")
                print("Randomly genetrated number to create Binary mask")
                print(rand)
                print("")
                print("Current Binary mask")
                print(bin_mask)
                print("")
                print("Full Binary mask archive")
                print(bin_archive)
                print("")
                print("G_hist")
                print(g_hist)
                print("")
                print("Current evals processed")
                print(update_counter_eval)


            # Since we only evaluate updated values
            self.train_simplified_gra(temp_population, writer, 0, bin_mask)
            for genome in temp_population:
                print('genome before')
                print(genome.fitness_vector)
            for count2, genome in enumerate(temp_population):
                myFitOffspring[count2] = genome.fitness_vector

            # Apply decomp approach - Note in Java code this falls under the generational() method in main class file
            #solve(self, gen, genomes, new_population, myFitParent, myFitOffspring, extArchivePop):
            evolver.solve(i, genomes, temp_population, myFitParent, myFitOffspring, extArchivePop, mo_type)
            genomes = deepcopy(evolver.parent_pop)
            #new_population = evolver.offspring_pop
            myFitParent = deepcopy(evolver.parent_fit)
            #myFitOffspring = evolver.offspring_fit
            extArchivePop = deepcopy(evolver.extPop)

            g_hist.append(deepcopy(evolver.util))


            if update_counter_utility == self.delta_T:
                # i + 1 because g_hist is one ahead
                u = evolver.utility_aggreg_func(g_hist, i+1, self.delta_T)
                poi = evolver.prob_of_improv(u)
                update_counter_utility = -1
            update_counter_utility = int(update_counter_utility) + 1



            # Remove duplicates and dominated solutions from extPop
            # TODO
            solutionsTuple = [(evolver.fitnessMO(genome), genome) for genome in extArchivePop ]
            #print(solutionsTuple)
            solutions = [x[0] for x in solutionsTuple]

            #self.save_genomes(extArchivePop, writer, i)

            if len(genome.fitness_vector) == 3:
                obj1 = []
                obj2 = []
                obj3 = []
                for h in range(len(solutions)):
                    obj1.append(solutions[h][0])
                    obj2.append(solutions[h][1])
                    obj3.append(solutions[h][2])
                costs = np.column_stack((np.array(obj1), np.array(obj2), np.array(obj3)))
            elif len(genome.fitness_vector) == 2:
                obj1 = []
                obj2 = []
                for h in range(len(solutions)):
                    obj1.append(solutions[h][0])
                    obj2.append(solutions[h][1])
                costs = np.column_stack((np.array(obj1), np.array(obj2)))
            bool_non_dom_sol_df = evolver.is_pareto_efficient_simple(costs)

            extArchivePop = list(compress(extArchivePop, bool_non_dom_sol_df))

            genome_id_list = []
            for genome in extArchivePop:
                genome_id_list.append(genome.genome_filename)
            for filename in os.listdir(str(self.path) + '/models/'):
                print(filename)
                if filename not in genome_id_list:
                    os.remove(str(self.path) + '/models/' + filename)
            #self.save_genomes(genomes, writer, i)

            self.save_genomes(extArchivePop, writer, i)

            # Output bin mask
            output_file = open(self.path + '/bin_mask.txt', 'w')
            for b in bin_archive:
                str_list = ''.join(str(e) for e in b)
                output_file.write(str_list + '\n')

            output_file2 = open(self.path + '/g_hist.txt', 'w')
            for g in g_hist:
                str_list = ''.join(str(e) for e in g)
                # in orginal experimentation this was output_file.write() as such bin_mask contains both g_hist and bin_acrhive,
                output_file2.write(str_list + '\n')

            output_file.close()
            output_file2.close()

            # Get the average accuracy for this generation.
            average_accuracy = self.get_average_accuracy(extArchivePop)

            # Print out the average accuracy each generation.
            logging.info("Generation average: %.2f%%" % (average_accuracy * 100))
            logging.info('-'*80)

            # Technically i represnts our generation number
            i = i + 1

            # Evolve, except on the last iteration.

        # Sort our final population according to performance.
        genomes = sorted(extArchivePop, key=lambda x: x.accuracy, reverse=True)

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

        denom = len(genomes)
        if denom == 0:
            denom = 1
        return total_accuracy / denom

    @staticmethod
    def print_genomes(genomes):
        logging.info('-'*80)
        for genome in genomes:
            genome.print_genome()
