"""
Class that holds a genetic algorithm for evolving a network.

Inspiration:

    http://lethain.com/genetic-algorithms-cool-name-damn-simple/
"""
from __future__ import print_function

import random
import copy

from functools import reduce
from operator import add
from genome import Genome
from idgen import IDgen
from allgenomes import AllGenomes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Evolver():
    """Class that implements genetic algorithm."""
    def __init__(self, all_possible_genes, retain=0.5, random_select=0.3, mutate_chance=0.5):
        """Create an optimizer.

        Args:
            all_possible_genes (dict): Possible genome parameters
            retain (float): Percentage of population to retain after
                each generation
            random_select (float): Probability of a rejected genome
                remaining in the population
            mutate_chance (float): Probability a genome will be
                randomly mutated

        """
        self.all_possible_genes = all_possible_genes
        self.retain = retain
        self.random_select = random_select
        self.mutate_chance = mutate_chance
        self.fronts = []
        self.pop = []
        self.num_of_tour_particips = 3
        self.tournament_prob = 0.9

        #set the ID gen
        self.ids = IDgen()

    def create_population(self, count):
        """Create a population of random networks.

        Args:
            count (int): Number of networks to generate, aka the
                size of the population

        Returns:
            (list): Population of network objects

        """
        pop = []
        i = 0

        while i < count:
            # Initialize a new genome.
            genome = Genome(self.all_possible_genes, {}, self.ids.get_next_ID(), 0, 0, self.ids.get_Gen())

            # Set it to random parameters.
            genome.set_genes_random()

            if i == 0:
                #this is where we will store all genomes
                self.master = AllGenomes(genome)
            else:
                # Make sure it is unique....
                while self.master.is_duplicate(genome):
                    genome.mutate_one_gene()

            # Add the genome to our population.
            pop.append(genome)

            # and add to the master list


            if i > 0:
                self.master.add_genome(genome)

            i += 1
            #self.fronts = [[]]

        return pop

    @staticmethod
    def fitness(genome):
        """Return the accuracy, which is our fitness function."""
        return genome.accuracy

    @staticmethod
    def fitnessTournamentSelect(genome):
        """Return aggregation of fitness function"""
        return sum(genome.fitness_vector) / len(genome.fitness_vector)

    @staticmethod
    def fitnessMO(genome):
        """Return the accuracy, which is our fitness function."""
        return genome.fitness_vector

    def grade(self, pop):
        """Find average fitness for a population.

        Args:
            pop (list): The population of networks/genome

        Returns:
            (float): The average accuracy of the population

        """
        summed = reduce(add, (self.fitness(genome) for genome in pop))
        return summed / float((len(pop)))

    def breed(self, mom, dad):
        """Make two children from parental genes.

        Args:
            mother (dict): genome parameters
            father (dict): genome parameters

        Returns:
            (list): Two network objects

        """
        children = []

        #where do we recombine? 0, 1, 2, 3, 4... N?
        #with four genes, there are three choices for the recombination
        # ___ * ___ * ___ * ___
        #0 -> no recombination, and N == length of dictionary -> no recombination
        #0 and 4 just (re)create more copies of the parents
        #so the range is always 1 to len(all_possible_genes) - 1
        pcl = len(self.all_possible_genes)
        recomb_loc = random.randint(1,pcl - 1)
        child1 = {}
        child2 = {}

        #enforce defined genome order using list
        keys = list(self.all_possible_genes)
        keys = sorted(keys)

        #*** CORE RECOMBINATION CODE ****
        for x in range(0, pcl):
            if x < recomb_loc:
                child1[keys[x]] = mom.geneparam[keys[x]]
                child2[keys[x]] = dad.geneparam[keys[x]]
            else:
                child1[keys[x]] = dad.geneparam[keys[x]]
                child2[keys[x]] = mom.geneparam[keys[x]]

        # Initialize a new genome
        # Set its parameters to those just determined
        # they both have the same mom and dad
        genome1 = Genome(self.all_possible_genes, child1, self.ids.get_next_ID(),
                         mom.u_ID, dad.u_ID, self.ids.get_Gen())
        genome2 = Genome(self.all_possible_genes, child2, self.ids.get_next_ID(),
                         mom.u_ID, dad.u_ID, self.ids.get_Gen())

        #at this point, there is zero guarantee that the genome is actually unique
        # Randomly mutate one gene
        #if self.mutate_chance > random.random():
        #    genome1.mutate_one_gene()

        #if self.mutate_chance > random.random():
        #    genome2.mutate_one_gene()

        #do we have a unique child or are we just retraining one we already have anyway?
        #while self.master.is_duplicate(genome1):
        #    genome1.mutate_one_gene()

        #self.master.add_genome(genome1)

        #while self.master.is_duplicate(genome2):
        #    genome2.mutate_one_gene()

        #self.master.add_genome(genome2)

        children.append(genome1)
        children.append(genome2)

        return children

    def fast_nondominated_sort(self, genomes):
        self.fronts = [[]]
        for genome in genomes:
            genome.domination_count = 0
            genome.dominated_solutions = []
            for other_genome in genomes:
                if genome.dominates(other_genome):
                    genome.dominated_solutions.append(other_genome)
                elif other_genome.dominates(genome):
                    genome.domination_count += 1
            if genome.domination_count == 0:
                genome.rank = 0
                self.fronts[0].append(genome)
        i = 0
        while len(self.fronts[i]) > 0:
            temp = []
            for genome in self.fronts[i]:
                for other_genome in genome.dominated_solutions:
                    other_genome.domination_count -= 1
                    if other_genome.domination_count == 0:
                        other_genome.rank = i+1
                        temp.append(other_genome)
            i = i+1
            self.fronts.append(temp)

    def calculate_crowding_distance(self, front):
        if len(front) > 0:
            solutions_num = len(front)
            for individual in front:
                individual.crowding_distance = 0

            for m in range(len(front[0].fitness_vector)):
                front.sort(key=lambda individual: individual.fitness_vector[m])
                front[0].crowding_distance = 10**9
                front[solutions_num-1].crowding_distance = 10**9
                m_values = [individual.fitness_vector[m] for individual in front]
                print('m values')
                print(m_values)
                scale = max(m_values) - min(m_values)
                if scale == 0:
                    scale = 1
                for i in range(1, solutions_num-1):
                    front[i].crowding_distance += (front[i+1].fitness_vector[m] - front[i-1].fitness_vector[m])/scale

    def crowding_operator(self, individual, other_individual):
        #print(individual.rank)
        #print(other_individual.rank)
        if (individual.rank < other_individual.rank) or \
            ((individual.rank == other_individual.rank) and (individual.crowding_distance > other_individual.crowding_distance)):
            return 1
        else:
            return -1

    def tournament(self, population):
        participants = random.sample(population, self.num_of_tour_particips)
        best = None
        for participant in participants:
            #print(len(participants))
            #print(participant)
            #print(best)
            if (best == None) or (self.crowding_operator(participant, best) == 1 and self.choose_with_prob(self.tournament_prob)):
                best = participant
        return best

    def tournament_naive(self, population):
        participants = random.sample(population, self.num_of_tour_particips)
        best = None
        for participant in participants:
            print(len(participants))
            print(participant)
            print(best)
            if (best == None) or (self.crowding_operator(participant, best) == 1 and self.choose_with_prob(self.tournament_prob)):
                best = participant
        return best

    def choose_with_prob(self, prob):
        if random.random() <= prob:
            return True
        return False

    # https://github.com/baopng/NSGA-II/blob/29a6ec33f87b32a7fb2596091e1a51c897106e7b/nsga2/utils.py#L24
    def combine_pop(self, pop):
        """Evolve a population of genomes using NSGA-II. Steps involved
            1) Create new pop using corssover and mutation
            2) Combine
            3) Use non-dominated

        Args:
            pop (list): A list of genome parameters

        Returns:
            (list): The evolved population of networks

        """
        self.ids.increase_Gen()

        new_pop = copy.deepcopy(pop)
        np_idx = 0

        for np_idx in range(0, len(pop)-1, 2):
            # This allows proper selection pressure for crossover operation
            parent1 = self.tournament(pop)
            parent2 = parent1
            while parent1 == parent2:
                parent2 = self.tournament(pop)

            # Recombine and mutate
            babies = self.breed(parent1, parent2)
            # the babies are guaranteed to be novel

            if np_idx < len(pop):
                new_pop[np_idx] = babies[0]
                new_pop[np_idx+1] = babies[1]

        np_idx = 0
        for genome in new_pop:
            if self.mutate_chance > random.random():
                gtc = copy.deepcopy(genome)

                #while self.master.is_duplicate(gtc):
                gtc.mutate_one_gene()

                gtc.set_generation( self.ids.get_Gen() )

                new_pop[np_idx] = gtc
                np_idx += 1
                self.master.add_genome(gtc)

        pop.extend(new_pop)

        return pop

    # will rename evolve at some stage
    def evolve(self, pop):
        """Evolve a population of genomes.

        Args:
            pop (list): A list of genome parameters

        Returns:
            (list): The evolved population of networks

        """
        #increase generation
        self.ids.increase_Gen()

        # Get scores for each genome
        graded = [(self.fitness(genome), genome) for genome in pop]
        solutionsTuple = [(self.fitnessMO(genome), genome) for genome in pop]
        #print(solutionsTuple)
        solutions = [x[0] for x in solutionsTuple]
        genome_hash = [x[1] for x in solutionsTuple]
        acc = [x[0] for x in graded]
        obj1 = []
        obj2 = []
        obj3 = []
        accVec = []
        hash = []
        for i in range(len(solutions)):
            obj1.append(solutions[i][0])
            obj2.append(solutions[i][1])
            obj3.append(solutions[i][2])
            accVec.append(acc[i])
            hash.append(genome_hash[i])

        #print(np.column_stack((np.array(obj1), np.array(obj2), np.array(obj3), np.array(acc))))
        #costs = np.column_stack((np.array(obj1), np.array(obj2)))
        costs = np.column_stack((np.array(obj1), np.array(obj2), np.array(obj3)))
        #print(costs)

        bool_non_dom_sol_df = is_pareto_efficient_simple(costs)

        df = pd.DataFrame(costs, columns=['obj1', 'obj2', 'obj3'])
        df['hash'] = hash
        df['non_dominated'] = bool_non_dom_sol_df
        non_dom = sum(bool_non_dom_sol_df)

        #non_dom_df = costs[bool_non_dom_sol_df,:]
        #print(bool_non_dom_sol_df)
        #print(df)
        df = df.sort_values('non_dominated', ascending=False)

        print("df")
        print(df)
        print("")


        #plt.scatter(obj1, obj2)
        #plt.show()
        #plt.scatter(df['obj1'], df['obj2'])
        #plt.show()

        #and use those scores to fill in the master list
        for genome in pop:
            self.master.set_accuracy(genome)

        # Sort on the scores.
        # FS: This is where we use Pareto Dominance instead of ACC
        #     We will use a sort to find which networks have the most desirable fitness

        # for fitness vector
        graded = df['hash'].to_list()
        # for accuracy
        #graded = [x[1] for x in sorted(graded, key=lambda x: x[0], reverse=True)]

        #print(graded2)
        print(graded)
        sys.exit(0)

        # Get the number we want to keep unchanged for the next cycle.
        elitist_length = int(len(graded)*self.retain)
        new_gen_length = len(pop) - int(len(graded)*self.retain)

        # In this first step, we keep the 'top' X percent (as defined in self.retain)
        # We will not change them, except we will update the generation
        elitist_retain = graded[:elitist_length]
        # use this to fill pop after mutation and crossover
        remainder = graded[new_gen_length:]
        # since our fitness is binary we'll only select non-dominated, i.e these are our fitest
        # Also our pop size is low so reampling indeces multiple times is not nesecary
        # This is not true tournament selection as such

        # TOURNAMENT SELECT - NAIVE AGGREGATE

        non_dom_selection = graded[:non_dom]


        #print("graded[:retain_length]: " + str(elitist_retain))
        #print("graded[new_gen_length:]: " + str(remainder))
        # "For the lower scoring ones, randomly keep some anyway.
        # This is wasteful, since we _know_ these are bad, so why keep rescoring them without modification?
        # At least we should mutate them"

        # 2021/08/16 FS: original code is incorrect.
        #    1) makes our pop size variable which is Wrong (This means our pop of networks will keep growing!!!)
        #    2) even if an indiv. has previously been used but was not deemed fit, this does not mean it wont be useful for crossover
        #       at a later stage
        #    3) even though duplicates are undesirable, only evolving the worst individuals in the population is just wrong

        new_pop = pop
        np_idx = 0

        for np_idx in range(0, len(pop)-1, 2):
            if self.mutate_chance > random.random():
                # This allows proper selection pressure for crossover operation
                parents = random.sample(range(len(pop)-1), k=5)

                tournament_idx = bool_non_dom_sol_df[parents]

                #print(parents)
                #print(tournament_idx)
                sorted_parents = [x for _, x in sorted(zip(tournament_idx, parents), reverse=True)]
                # dont want to always select lowest indeces so will shuffle non dominated indexex
                true_range = int(sum(tournament_idx))
                # if they are all false (dominated) dont try shuffle
                if true_range > 0:
                    srt_tmp = sorted_parents[0:true_range]
                    random.shuffle(srt_tmp)
                    for i in range(true_range):
                        sorted_parents[i] = srt_tmp[i]

                #print(sorted_parents)
                #sys.exit()

                i_male = sorted_parents[0]
                i_female = sorted_parents[1]

                male = pop[i_male]
                female = pop[i_female]

                # Recombine and mutate
                babies = self.breed(male, female)
                # the babies are guaranteed to be novel

                if np_idx < len(pop):
                    new_pop[np_idx] = babies[0]
                    new_pop[np_idx+1] = babies[1]

        np_idx = 0
        for genome in pop:
            if self.mutate_chance > random.random():
                gtc = copy.deepcopy(genome)

                #while self.master.is_duplicate(gtc):
                gtc.mutate_one_gene()

                gtc.set_generation( self.ids.get_Gen() )

                new_pop[np_idx] = gtc
                np_idx += 1
                self.master.add_genome(gtc)


        new_generation = random.sample(new_pop, len(pop))

        #mo_type = 'naive-rand'
        # NAIVE - TOURNAMENT SELECTION W/ OBJECTIVE AGGREGATE (as described in the NeuroEvolutionary paper)
        mo_type = 'naive-tournament-select'

        if mo_type == 'naive-rand':
            for i in range(elitist_length):
                new_generation[i] = elitist_retain[i]
        elif mo_type == 'naive-tournament-select':
            for i in range(elitist_length):
                retain1, retain2 = random.choices(pop, k=2)
                #HERE
                fit1 = self.fitnessTournamentSelect(retain1)
                fit2 = self.fitnessTournamentSelect(retain2)
                if fit1 > fit2:
                    new_generation[i] = retain1
                elif fit2 >= fit1:
                    new_generation[i] = retain2

        #print("pop and new gen lengths")
        #print('old pop: ' + str(len(pop)))
        #print('new pop: ' + str(len(new_generation)))

        # Original approach failed this
        assert len(pop) == len(new_generation)

        #sys.exit()

        return new_generation



# Pareto dominance code taken from:
# https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python
def is_pareto_efficient_simple(costs):
    """
    Find the pareto-efficient points
    :params: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient]<c, axis=1)  # Keep any point with a lower cost
            is_efficient[i] = True  # And keep self
    return is_efficient
