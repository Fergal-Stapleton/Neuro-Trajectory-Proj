"""The genome to be evolved."""
from training_history_plot import TrainingHistoryPlot
from keras.callbacks import EarlyStopping
import random
import logging
import hashlib
import copy
import numpy as np
from keras import backend as K


class Genome():
    """
    Represents one genome and all relevant utility functions (add, mutate, etc.).
    """
    def __init__(self, all_possible_genes=None, geneparam={}, u_ID=0, mom_ID=0, dad_ID=0, gen=0):
        """Initialize a genome.

        Args:
            all_possible_genes (dict): Parameters for the genome
        """
        self.accuracy = 0.0
        self.all_possible_genes = all_possible_genes

        # (dict): represents actual genome parameters
        self.geneparam = geneparam
        self.u_ID = u_ID
        self.parents = [mom_ID, dad_ID]
        self.generation = gen

        #hash only makes sense when we have specified the genes
        if not geneparam:
            self.hash = 0
        else:
            self.update_hash()

    def update_hash(self):
        """
        Refesh each genome's unique hash - needs to run after any genome changes.
        """
        genh = ''
        for p in self.all_possible_genes:
            genh += str(self.geneparam[p])

        self.hash = hashlib.md5(genh.encode("UTF-8")).hexdigest()
        self.accuracy = 0.0

    def set_genes_random(self):
        """Create a random genome."""
        self.parents = [0, 0]
        for key in self.all_possible_genes:
            self.geneparam[key] = random.choice(self.all_possible_genes[key])

        self.update_hash()

    def mutate_one_gene(self):
        """Randomly mutate one gene in the genome.

        Args:
            network (dict): The genome parameters to mutate

        Returns:
            (Genome): A randomly mutated genome object

        """
        # Which gene shall we mutate? Choose one of N possible keys/genes.
        gene_to_mutate = random.choice(list(self.all_possible_genes.keys()))

        # And then let's mutate one of the genes.
        # Make sure that this actually creates mutation
        current_value = self.geneparam[gene_to_mutate]
        possible_choices = copy.deepcopy(self.all_possible_genes[gene_to_mutate])
        possible_choices.remove(current_value)
        print(self.geneparam[gene_to_mutate])
        print( possible_choices )
        self.geneparam[gene_to_mutate] = random.choice( possible_choices )
        self.update_hash()

    def set_generation(self, generation):
        """needed when a genome is passed on from one generation to the next.
        the id stays the same, but the generation is increased"""
        self.generation = generation

    def set_genes_to(self, geneparam, mom_ID, dad_ID):
        """Set genome properties.
        this is used when breeding kids

        Args:
            genome (dict): The genome parameters
        IMPROVE
        """
        self.parents = [mom_ID, dad_ID]
        self.geneparam = geneparam
        self.update_hash()

    def train_and_score(self, model_train, dataset, path):
        logging.info("Getting training samples")
        logging.info("Compling Keras model")

        batch_size = self.geneparam['batch_size']
        epochs = self.geneparam['epochs']

        parameters = list()

        for p in self.all_possible_genes:
            if p is 'batch_size':
                continue
            elif p is 'epochs':
                continue
            else:
                parameters.append(self.geneparam[p])

        print(parameters)
        input_shape = np.shape(dataset.X_train)
        model = model_train(input_shape, parameters)

        parameters.append(self.geneparam['batch_size'])
        parameters.append(self.geneparam['epochs'])
        # Helper: Early stopping.
        early_stopper = EarlyStopping(monitor='val_loss', min_delta=0.1, patience=5, verbose=0, mode='auto')
        # FS: 18/07/2021: This is problematic as it is set up for classification
        #                 I will comment out line 49 and 50 from training_history_plot
        history = TrainingHistoryPlot(path, dataset, parameters)
        print("Y_train [0] just before fit: " + str(dataset.Y_train.shape[0]))
        print("Y_train [1] just before fit: " + str(dataset.Y_train.shape[1]))
        print("Y_valid [0] just before fit: " + str(dataset.Y_valid.shape[0]))
        print("Y_valid [1] just before fit: " + str(dataset.Y_valid.shape[1]))
        model.fit(dataset.X_train, dataset.Y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  validation_data=(dataset.X_valid, dataset.Y_valid),
                  callbacks=[early_stopper, history])

        score = model.evaluate(dataset.X_valid, dataset.Y_valid, verbose=0)
        prediction = model.predict(dataset.X_test)

        # TEST here
        # TODO hardcoding image_sequence_length, this needs to be fixed
        # original code also has this hardcoded in load_data.py
        image_sequence_length = 3
        print(prediction)
        l1 = l1_objective(prediction, image_sequence_length)
        l2 = l2_objective(prediction, image_sequence_length)
        l3 = l3_objective(prediction, image_sequence_length)
        L = [l1, l2, l3]
        print(L)
        K.clear_session()
        # we do not care about keeping any of this in memory -
        # we just need to know the final scores and the architecture

        # 1 is accuracy. 0 is loss.
        return score[1]

    def train(self, model, trainingset, path):
        #don't bother retraining ones we already trained
        if self.accuracy == 0.0:
            self.accuracy = self.train_and_score(model, trainingset, path)

    def print_genome(self):
        """Print out a genome."""
        self.print_geneparam()
        logging.info("Acc: %.2f%%" % (self.accuracy * 100))
        logging.info("UniID: %d" % self.u_ID)
        logging.info("Mom and Dad: %d %d" % (self.parents[0], self.parents[1]))
        logging.info("Gen: %d" % self.generation)
        logging.info("Hash: %s" % self.hash)

    def print_genome_ma(self):
        """Print out a genome."""
        self.print_geneparam()
        logging.info("Acc: %.2f%% UniID: %d Mom and Dad: %d %d Gen: %d" % (self.accuracy * 100, self.u_ID,
                                                                           self.parents[0], self.parents[1],
                                                                           self.generation))
        logging.info("Hash: %s" % self.hash)

    # print nb_neurons as single list
    def print_geneparam(self):
        g = self.geneparam.copy()
        logging.info(g)

    # convert nb_neurons_i at each layer to a single list
    def nb_neurons(self):
      nb_neurons = [None] * 6
      for i in range(0,6):
        nb_neurons[i] = self.geneparam['nb_neurons_' + str(i+1)]

      return nb_neurons


# Eqn 3. NeuroTrajectory distance-based feedback
def l1_objective(p, image_sequence_length):
    """
    Calculate Eqn 3. NeuroTrajectory distance-based feedback
    :params: Numpy array of input data
    :return: list, each element is the distance-based feedback corresponding to the ith sequence of OGs
    """
    l1 = 0.0
    dest_list = []
    for i in range(p.shape[0]):
        temp = 0.0
        flag = 1
        if flag == 1:
            # Since our start point is implicitly always [0.0, 0.0]
            temp += np.sqrt((p[i][1] - 0.0)**2 + (p[i][0] - 0.0)**2)
            flag = 0
        # these go up by 2
        for j in range(2,(image_sequence_length-1)*2 -1):
            x2 = p[i][j + 1] # starts at 3
            x1 = p[i][j - 1] # starts at 1
            y2 = p[i][j + 0] # starts at 2
            y1 = p[i][j - 2] # starts at 0
            temp += np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        dest_list.append(temp/image_sequence_length)
    l1 = sum(dest_list)/p.shape[0]
    return l1

def l2_objective(p, image_sequence_length):
    """
    Calculate Eqn 4. NeuroTrajectory lateral velocity
    :params: Numpy array of input data
    :return: list, each element is the lateral velocity corresponding to the ith sequence of OGs
    """
    # Calculate velocity as rate of change in position across fixed sequence length
    l2 = 0.0
    vd = []
    for i in range(p.shape[0]):
        temp = 0.0
        flag = 1
        if flag == 1:
            # Since our start point is implicitly always [0.0, 0.0]
            temp += np.abs(p[i][0] - 0.0)
            flag = 0
        # these go up by 2
        for j in range(2,(image_sequence_length-1)*2 -1):
            temp += np.abs(p[i][j] - p[i][j - 2])
        vd.append(temp/image_sequence_length)
    l2 = sum(vd)/p.shape[0]
    return l2

def l3_objective(p, image_sequence_length):
    """
    Calculate Eqn 5. NeuroTrajectory longtitudinal velocity
    :params: Numpy array of input data
    :return: list, each element is the longtitudinal velocity corresponding to the ith sequence of OGs
    """
    # Calculate velocity as rate of change in position across fixed sequence length
    l3 = 0.0
    vf = []
    for i in range(p.shape[0]):
        temp = 0.0
        flag = 1
        if flag == 1:
            # Since our start point is implicitly always [0.0, 0.0]
            temp += np.abs(p[i][1] - 0.0)
            flag = 0
        # these go up by 2
        for j in range(2,(image_sequence_length-1)*2 -1):
            print(p[i][j + 1])
            print(p[i][j - 1])
            temp += np.abs(p[i][j + 1] - p[i][j - 1])
        vf.append(temp/image_sequence_length)
    print(vf)
    l3 = sum(vf)/p.shape[0]
    return l3
