
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
import math

class Evolver_moead():
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
        #self.fronts = []
        self.pop = []
        self.num_of_tour_particips = 3
        self.tournament_prob = 0.9
        self.m = None
        self.pop_size = None
        self.dist_mat  = None
        self.neighbour_table = None
        self.weights = None
        self.neighbour_size = 3
        self.decomp_type = 0
        self.ideal_point = None
        self.nadir_point = None

        self.parent_pop = None
        self.offspring_pop = None
        self.parent_fit = None
        self.offspring_fit = None
        self.extPop = None

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
        self.m =  len(pop[0].fitness_vector)
        self.pop_size = len(pop)
        self.dist_mat = [[None for x in range(self.pop_size)] for y in range(self.pop_size)]
        self.neighbour_table = [ None for y in range(self.pop_size)]
        self.weights  = [[None for x in range(self.m)] for y in range(self.pop_size)]
        self.ideal_point = [[1.0 for x in range(self.m)] for y in range(self.pop_size)]
        self.nadir_point = [[0.0 for x in range(self.m)] for y in range(self.pop_size)]
        return pop

    def initialize(self, myFit):
        for i in range(self.m):
            self.ideal_point[i] = 1.0
        self.init_weights()
        self.init_neighbour()
        for i in range(self.pop_size):
            self.update_ref(i, myFit)

    def easy_sort(self, tobesorted):
        # Just use list comprehension, no messing around like in java
        incremental_list = [i for i in range(len(tobesorted))]
        index = [x for _, x in sorted(zip(tobesorted, incremental_list))]
        return index

    def init_neighbour(self):
        for i in range(self.pop_size):
            self.dist_mat[i][i] = 0
            for j in range(int(i+1), self.pop_size):
                self.dist_mat[i][j] = self.dist_vec(self.weights[i], self.weights[j])
                self.dist_mat[j][i] = self.dist_mat[i][j]

        for i in range(self.pop_size):
            index = self.easy_sort(self.dist_mat[i]) # full pop sorted in terms of distance to current i
            array = []
            for j in range(self.neighbour_size):
                array.append(index[j]) # take closest indeces with respect to distance
            self.neighbour_table[i] = array


    def returnParentsSelection(self, indivs):
        # Return random values between 0 and neighbourhoos size
        list_increment = [i for i in range(self.neighbour_size)]
        # automatically assigns to original list
        random.shuffle(list_increment)
        id_l, id_k = list_increment[0] , list_increment[1]
        #print(self.neighbour_table)
        k = self.neighbour_table[indivs][id_k]
        l = self.neighbour_table[indivs][id_l]
        return k, l

    def applyCrossover(self, mom, dad):
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


        children.append(genome1)
        children.append(genome2)

        return children

    def applyOnePointCrossover(self, mom, dad):
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


        children.append(genome1)
        children.append(genome2)

        return children


    def applyMutation(self, new_pop):
        np_idx = 0
        for genome in new_pop:
            if self.mutate_chance > random.random():
                gtc = copy.deepcopy(genome)

                #while self.master.is_duplicate(gtc):
                gtc.mutate_one_gene()

                gtc.set_generation( self.ids.get_Gen() )

                new_pop[np_idx] = gtc

                self.master.add_genome(gtc)
            np_idx += 1 # In NSGA-II this is inside the loop but would cause bias here, as such increment outside the loop
        return new_pop

    def init_weights(self):
        if(self.m == 2):
            for i in range(self.pop_size):
                self.weights[i][0] = i / self.pop_size
                self.weights[i][1] = (self.pop_size - i) / self.pop_size
        elif(self.m == 3):
            # 3 -> 10
            # 4 -> 15
            self.weights = self.simplex_latice_hyperplane_weight_gen(self.m, self.check_dividion_by_pop_size(self.pop_size))
        #elif(self.m == 3 and standard == False):
        #    weight_temp = []
        #    for i in range(self.pop_size):
        #        for j in range(self.pop_size):
        #            if ((i + j) <= self.pop_size):
        #                k =  self.pop_size - i - j
        #                weight_ = [None] * 3
        #                weight_[0] = i / self.pop_size;
        #                weight_[1] = j / self.pop_size;
        #                weight_[2] = k / self.pop_size;
        #                weight_temp.append(weight_)
        #    print(weight_temp)
        #    weight_temp = sorted((x for x in weight_temp), key=lambda x: sum(x), reverse=False)
        #    #print(sum(x) for x in weight_temp)
        #    new_lists = []
        #    for nested in weight_temp:
        #        new_lists.append(sum(nested))
        #    print(new_lists)
        #    self.weights = weight_temp[:self.pop_size]
        else:
            print("Current implementation of moea/d cannot exceed 3 objectives")

    def check_dividion_by_pop_size(self, s):
        check = 0
        value = -2
        for i in range(s):
            check += i
            value += 1
            #print(check)
            if check == int(s):
                print('population size correct for chosen weight initialization method')
                print(value)
                return value
            elif check > int(s):
                print('population size incorrect for chosen weight initialization method')
                print('... nearest numbers that are correct:')
                print(check - value + 1)
                print(check)
                print('... exiting :(')
                sys.exit()

    # Based on code from Platypus - this has been simplified as we are not interested in divisions_inner
    #https://github.com/Project-Platypus/Platypus/blob/master/platypus/weights.py
    def simplex_latice_hyperplane_weight_gen(self, m, divisions_outer):
        """
        Returns weights that are uniformly distributed on the
        hyperplane intersecting

        Parameters
        ----------
        m : int
            The number of objectives.
        divisions_outer : int
            In platypus this is refered to as the number of divisions along the outer set of weights.
            To conceptualise though it would (n + 1) of the outer weights, so for instance if n = 3
            Then we have 4 weights on hypotenuse, adjacent and opposite:

            .
            . .
            . . .
            . . . .

        """

        def generate_recursive(weights, weight, left, total, index):
            if index == m - 1:
                weight[index] = float(left) / float(total)
                weights.append(copy.copy(weight))
            else:
                for i in range(left+1):
                    weight[index] = float(i) / float(total)
                    generate_recursive(weights, weight, left-i, total, index+1)

        def generate_weights(divisions):
            weights = []
            generate_recursive(weights, [0.0]*m, divisions, divisions, 0)
            return weights

        weights = generate_weights(divisions_outer)

        return weights

    # Based on code from Platypus - this has been simplified as we are not interested in divisions_inner
    #https://github.com/Project-Platypus/Platypus/blob/master/platypus/weights.py
    def simplex_latice_hyperplane_weight_test(self, m, divisions_outer):
        """
        Returns weights that are uniformly distributed on the
        hyperplane intersecting

        Parameters
        ----------
        m : int
            The number of objectives.
        divisions_outer : int
            In platypus this is refered to as the number of divisions along the outer set of weights.
            To conceptualise though it would (n + 1) of the outer weights, so for instance if n = 3
            Then we have 4 weights on hypotenuse, adjacent and opposite:

            .
            . .
            . . .
            . . . .

        """

        def generate_recursive(weights, weight, left, total, index):
            if index == m - 1:
                weight[index] = float(left)**2 / float(total) * 0.125
                weights.append(copy.copy(weight))
            else:
                for i in range(left+1):
                    weight[index] = float(i)**2/ float(total)
                    generate_recursive(weights, weight, left-i, total, index+1)

        def generate_weights(divisions):
            weights = []
            generate_recursive(weights, [0.0]*m, divisions, divisions, 0)
            return weights

        weights = generate_weights(divisions_outer)

        return weights

    def dist_vec(self, a, b):
        sum = 0.0
        for i in range(len(a)):
            sum += (a[i] - b[i])**2
        return np.sqrt(sum)

    def update_ref(self, i, myFit):
        for n in range(self.m):
            if (self.decomp_type != 3) and (myFit[i][n] < self.ideal_point[n]):
                self.ideal_point[n] = myFit[i][n]
            #if (self.decomp != 3) and (myFit[i][n] > self.nadir_point[n]):
            #    self.nadir_point[n] = myFit[i][n]

    def solve(self, gen, genomes, temp_population, myFitParent, myFitOffspring, extArchivePop, type):
        self.parent_pop = genomes
        self.offspring_pop = temp_population
        self.parent_fit = myFitParent
        self.offspring_fit = myFitOffspring
        self.extPop = extArchivePop
        self.util = [0.0]*self.pop_size

        for i in range(self.pop_size):
            self.update_ref(i, self.offspring_fit)
        for i in range(self.pop_size):
            self.decomp(i, type)

    def decomp(self, i, type):
        d = 0
        e = 0
        d_prime = 0
        e_prime = 0
        # This will keep track of each weight index and its d and e value
        # i.e find weight_index where arg_max (( e - d) / e )
        weight_index_list = []
        subproblem_fitness_diff = []
        d_list = []
        e_list = []

        # Randomizes the neighbourhood ids and break upon first update
        if type == 'moead':
            id_list = range(0, self.neighbour_size)
            for j in id_list:
                # Maybe order solutions on fitness????
                weight_index = self.neighbour_table[i][j]
                local_weights = self.weights[weight_index]
                if (self.decomp_type == 0):
                    #print('At least got here')
                    d = self.tcheycheffScalarObj(local_weights, self.offspring_fit[i])
                    e = self.tcheycheffScalarObj(local_weights, self.parent_fit[weight_index])
                elif (self.decomp_type == 1):
                    d = self.pbiScalarObj(local_weights, self.offspring_fit[i])
                    e = self.pbiScalarObj(local_weights, self.parent_fit[weight_index])
                elif (self.decomp_type == 2):
                    d = self.weightedScalarObj(local_weights, self.offspring_fit[i])
                    e = self.weightedScalarObj(local_weights, self.parent_fit[weight_index])
                if (d < e):
                    print('Neighbourhood update has occurred')
                    self.parent_pop[weight_index] = self.offspring_pop[i]
                    self.parent_fit[weight_index] = self.offspring_fit[i]
                    self.extPop.append(self.parent_pop[weight_index])
                    break

        if type == 'moead_gra':
            for j in range(self.neighbour_size):
                # Maybe order solutions on fitness????
                weight_index = self.neighbour_table[i][j]
                local_weights = self.weights[weight_index]
                if (self.decomp_type == 0):
                    #print('At least got here')
                    d = self.tcheycheffScalarObj(local_weights, self.offspring_fit[i])
                    e = self.tcheycheffScalarObj(local_weights, self.parent_fit[weight_index])
                elif (self.decomp_type == 1):
                    d = self.pbiScalarObj(local_weights, self.offspring_fit[i])
                    e = self.pbiScalarObj(local_weights, self.parent_fit[weight_index])
                elif (self.decomp_type == 2):
                    d = self.weightedScalarObj(local_weights, self.offspring_fit[i])
                    e = self.weightedScalarObj(local_weights, self.parent_fit[weight_index])
                #elif (self.decomp_type == 3):
                #    d = self.ipbiScalarObj(local_weights, self.offspring_fit[i])
                #    e = self.ipbiScalarObj(local_weights, self.parent_fit[weight_index])
                eps = 1e-50
                diff = (e - d + eps) / (e + eps)
                weight_index_list.append(weight_index)
                #offspring_index_list.append(i)
                subproblem_fitness_diff.append(diff)
                d_list.append(d)
                e_list.append(e)

                # Minimization

            #print(d_list)
            #print(e_list)
            #print(subproblem_fitness_diff)
            # NOTE: *** THIS CAN BE SWITCHED WITH SEMANTIC NEIGHBOURHOOD ORDERING *** (future paper???)
            d_prime, e_prime, k = self.replacement_strategy(weight_index_list, subproblem_fitness_diff, d_list, e_list, i)
            #print(d_prime)
            #print(e_prime)

            if (d_prime < e_prime):
                print('Neighbourhood update has occurred')
                self.parent_pop[k] = self.offspring_pop[i]
                self.parent_fit[k] = self.offspring_fit[i]
                self.extPop.append(self.parent_pop[k])
                self.util[k] = d_prime
                print(self.util[k])
                #sys.exit()
            #self.util[k] = e_prime



    def replacement_strategy(self, weight_index_list, subproblem_fitness_diff, d_list, e_list, i):

        index = subproblem_fitness_diff.index(max(subproblem_fitness_diff))
        weight_index = weight_index_list[index]
        local_weights = self.weights[weight_index]
        if (self.decomp_type == 0):
                d_prime = self.tcheycheffScalarObj(local_weights, self.offspring_fit[i])
                e_prime = self.tcheycheffScalarObj(local_weights, self.parent_fit[weight_index])
        elif (self.decomp_type == 1):
                d_prime = self.pbiScalarObj(local_weights, self.offspring_fit[i])
                e_prime = self.pbiScalarObj(local_weights, self.parent_fit[weight_index])
        elif (self.decomp_type == 2):
                d_prime = self.weightedScalarObj(local_weights, self.offspring_fit[i])
                e_prime = self.weightedScalarObj(local_weights, self.parent_fit[weight_index])
        return d_prime, e_prime, weight_index

    def tcheycheffScalarObj(self, _lambda, f):
        min_fun = float("inf")
        for n in range(self.m):
            diff = abs( f[n] - self.ideal_point[n] )
            if (_lambda[n] == 0):
                feval = 0.00001 * diff
            else:
                feval = diff * _lambda[n]
            if (feval < min_fun):
                min_fun = feval
        return min_fun

    def pbiScalarObj(self, _lambda, f):
        fun = float("inf")
        z_nad = 0
        theta = 0.1
        d1 = 0.0
        d2 = 0.0
        n1 = 0.0
        for n in range(self.m):
            d1 += ( f[n] - self.ideal_point[n] ) * self.ideal_point[n]
            n1 += _lambda[n]**2
        n1 = math.sqrt(n1)
        d1 = abs(d1) / n1

        for n in range(self.m):
            diff = (( f[n] - self.ideal_point[n] ) - d1 * (_lambda[n] / n1))**2
        d2 =  math.sqrt(d2)
        fun = (d1 + theta * d2)
        return fun

    def ipbiScalarObj(self, _lambda, f):
        # TODO
        return 0

    def weightedScalarObj(self, _lambda, f):
        for n in range(self.m):
            max_fun += f[n] * _lambda[n]
        return max_fun

    # Pareto dominance code taken from:
    # https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python
    def is_pareto_efficient_simple(self, costs):
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

    @staticmethod
    def fitnessMO(genome):
        """Return the accuracy, which is our fitness function."""
        return genome.fitness_vector

class Evolver_moead_gra(Evolver_moead):
    def __init__(self, *args, **kwargs):
        super(Evolver_moead_gra, self).__init__(*args, **kwargs)
        self.delta_T = 3
        self.util = None


    def utility_aggreg_func(self, g_hist, i, delta_T):
        # In the paper i is used to index pop, I've used j since I have already used i to dentote gen index
        u = [None]*int(self.pop_size)
        eps = 1e-50
        #g_hist_sum = 0
        for j in range(self.pop_size):
            if (g_hist[i][j] < g_hist[i - delta_T][j]):
                u[j] = (g_hist[i - delta_T][j] - g_hist[i][j] + eps) / (g_hist[i - delta_T][j] + eps)
            else:
                u[j] = 0
        return u

    def prob_of_improv(self, u):
        p = [None]*int(self.pop_size)
        u_max = max(u)
        eps = 1e-50
        if u_max > eps:
            for j in range(self.pop_size):
                p[j] = (u[j] + eps)/(u_max + eps)
        else:
            for j in range(self.pop_size):
                p[j] = 1
        return p
