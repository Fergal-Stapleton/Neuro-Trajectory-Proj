"""The genome to be evolved."""
from training_history_plot import TrainingHistoryPlot
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
import tensorflow.keras
import random
import logging
import hashlib
import copy
import numpy as np
import sys
import math
from keras import backend as K
import pandas as pd
import matplotlib.pyplot as plt

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
        self.x_err = 0.0
        self.x_max = 0.0
        self.y_err = 0.0
        self.y_max = 0.0
        self.fitness_vector = [0.0, 0.0, 0.0]
        self.score = None
        self.all_possible_genes = all_possible_genes
        self.rank = None
        self.crowding_distance = None
        self.domination_count = None
        self.dominated_solutions = None
        self.features = None

        # (dict): represents actual genome parameters
        self.geneparam = geneparam
        self.u_ID = u_ID
        self.parents = [mom_ID, dad_ID]
        self.generation = gen
        self.genome_filename = None

        #hash only makes sense when we have specified the genes
        if not geneparam:
            self.hash = 0
        else:
            self.update_hash()

    def dominates(self, other_individual):
        and_condition = True
        or_condition = False
        # This is a nice way of doing it, logically sound
        for first, second in zip(self.fitness_vector, other_individual.fitness_vector):
            and_condition = and_condition and first <= second
            or_condition = or_condition or first < second
        return (and_condition and or_condition)

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
        #print(self.geneparam[gene_to_mutate])
        #print( possible_choices )
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

    # Training for NSGA-II
    def train_and_score_simplified(self, model_train, dataset, path, i, gen_max, run_n, pop_index):
        logging.info("Getting training samples")
        logging.info("Compling Keras model")

        batch_size = self.geneparam['batch_size']
        epochs = self.geneparam['epochs']

        image_sequence_length = int((dataset.number_of_classes + 2) /2)
        parameters = list()
        file_name = ''

        for p in self.all_possible_genes:
            if p is 'batch_size':
                continue
            elif p is 'epochs':
                continue
            else:
                parameters.append(self.geneparam[p])
                file_name += str(self.geneparam[p]) + '_'

        print(parameters)
        print("")
        input_shape = np.shape(dataset.X_train)
        model = model_train(input_shape, parameters)

        parameters.append(self.geneparam['batch_size'])
        parameters.append(self.geneparam['epochs'])
        # Helper: Early stopping.
        early_stopper = EarlyStopping(monitor='val_loss', min_delta=0.00005, patience=15, verbose=0, mode='auto')
        # FS: 18/07/2021: This is problematic as it is set up for classification
        #                 I will comment out line 49 and 50 from training_history_plot
        history = TrainingHistoryPlot(path, dataset, parameters, i)

        print("****** CHECK *******")
        print("Individual being evaluated: ")
        print(pop_index)
        np.set_printoptions(threshold=np.inf)
        #print(dataset.Y_train)
        #sys.exit()

        model.fit(dataset.X_train, dataset.Y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  validation_data=(dataset.X_valid, dataset.Y_valid),
                  callbacks=[early_stopper, history]
                  )

        # Save last gen models
        #if(gen_max == int(i+1)):
        self.genome_filename = 'model_' + str(file_name) + '_gen_' + str(i) + '_run_' + str(run_n) + '_' + str(self.u_ID) + '.h5'
        model.save(filepath = str(path) + '/models/' + self.genome_filename)
        #filepath = str(path) + '/models/model_' + str(file_name) + '_gen_' + str(i) + '_run_' + str(run_n) + '.h5'
        score = model.evaluate(dataset.X_valid, dataset.Y_valid, verbose=0)
        prediction = model.predict(dataset.X_valid)
        witheld = model.predict(dataset.X_test)

        real = copy.deepcopy(dataset.Y_valid)
        witheld_real = copy.deepcopy(dataset.Y_test)

        #pred_acc2 = ((prediction -  real)**2).mean(axis=1)
        #pred_acc3 = self.mse(prediction, real, image_sequence_length)

        #dataset.Y_train
        def denormalize(array, min, max):
            return array*(max - min) + min

        def denormalize_x(array, min, max):
            return array[:, ::2]*(max - min) + min

        def denormalize_y(array, min, max):
            return array[:, 1::2]*(max - min) + min

        def y_addition(array):
            array_copy = copy.deepcopy(array)
            for i in range(1, int(array.shape[1]/2) ):
                #print(i)
                array_copy[:, i*2 + 1] = array_copy[:, i*2 + 1] + array_copy[:, (i*2 -1)]
                array_copy[:, i*2] = array_copy[:, i*2] + array_copy[:, (i*2 - 2)]
            array = array_copy
            return array


        #real_rescale = denormalize(real, dataset.ymax, dataset.ymin)
        #prediction_rescale = denormalize(prediction, dataset.ymax, dataset.ymin)

        #np.set_printoptions(threshold=np.inf)
        #print(real_rescale)
        #print(prediction_rescale)


        # Revert our prediction with ymin and ymax

        # While our prediction is done solely on our validation set (test is withheld), our objective valculation should be
        # based on the training set
        #     1) We would bias our evolutionary stratregy if we used validation
        #     2) Our 3rd objective requires the training data
        obj_training = model.predict(dataset.X_train)
        #obj_training_reverse_scale = denormalize(obj_training, dataset.ymax, dataset.ymin)

        #print(dataset.superscaler_x_min)
        #print(dataset.superscaler_x_max)
        #sys.exit()

        #obj_training  = denormalize(obj_training, dataset.superscaler_x_min, dataset.superscaler_x_max)
        #prediction = denormalize(prediction, dataset.superscaler_x_min, dataset.superscaler_x_max)
        #real = denormalize(real, dataset.superscaler_x_min, dataset.superscaler_x_max)

        #np.set_printoptions(precision=3)
        #for i in range(len(real)):
        #    print(str(prediction[i]) + '\r')
        #    print(str(real[i]) + '\r')
        #    print("\n")

        #sys.exit()

        obj_training[:, ::2]  = denormalize_x(obj_training, dataset.superscaler_x_min, dataset.superscaler_x_max)
        prediction[:, ::2] = denormalize_x(prediction, dataset.superscaler_x_min, dataset.superscaler_x_max)
        real[:, ::2] = denormalize_x(real, dataset.superscaler_x_min, dataset.superscaler_x_max)
        #dataset.Y_valid[:, ::2] = denormalize_x(dataset.Y_valid, data_set.superscaler_x_min, dataset.superscaler_x_max)

        obj_training[:, 1::2]  = denormalize_y(obj_training, dataset.superscaler_y_min, dataset.superscaler_y_max)
        prediction[:, 1::2]  = denormalize_y(prediction, dataset.superscaler_y_min, dataset.superscaler_y_max)
        real[:, 1::2]  = denormalize_y(real, dataset.superscaler_y_min, dataset.superscaler_y_max)
        #dataset.Y_valid[:, 1::2]  = normalize_y(dataset.Y_valid, dataset.superscaler_y_min, dataset.superscaler_y_max)

        witheld[:, 1::2]  = denormalize_y(witheld, dataset.superscaler_y_min, dataset.superscaler_y_max)
        witheld_real[:, 1::2]  = denormalize_y(witheld_real, dataset.superscaler_y_min, dataset.superscaler_y_max)

        obj_training = y_addition(obj_training)
        prediction = y_addition(prediction)
        real = y_addition(real)
        witheld = y_addition(witheld)
        witheld_real  = y_addition(witheld_real )

        #np.set_printoptions(precision=3)
        #for i in range(len(real)):
        #    print(str(prediction[i]) + '\r')
        #    print(str(real[i]) + '\r')
        #    print("\n")

        colList = ['x1','y1','x2', 'y2', 'x3', 'y3', 'x4', 'y4', 'x5', 'y5','x6', 'y6', 'x7', 'y7']

        #df_pred = pd.DataFrame(prediction,  columns=colList)
        #df_zeros = pd.DataFrame(0, index=np.arange(df_pred.size), columns=colList)

        #plt.xlabel('x')
        #plt.ylabel('y')
        #plt.title('Positional data for model with best accuracy in last Generation')
        #plt.scatter(df_zeros['x1'], df_zeros['y1'], marker='*')
        #plt.scatter(df_pred['x1'], df_pred['y1'], marker='o')
        #plt.scatter(df_pred['x2'], df_pred['y2'], marker='x')
        #plt.scatter(df_pred['x3'], df_pred['y3'], marker='x')
        #plt.scatter(df_pred['x4'], df_pred['y4'], marker='x')
        #plt.scatter(df_pred['x5'], df_pred['y5'], marker='x')
        #plt.scatter(df_pred['x6'], df_pred['y6'], marker='x')
        #plt.scatter(df_pred['x7'], df_pred['y7'], marker='x')
        #plt.set_ylim([0, 50])
        #plt.set_ylim([-5, 5])
        #ax = plt.gca()
        #ax.set_ylim([0, 50])
        #ax.set_xlim([-5, 5])
        #plt.show()
        #plt.clf()
        #K.clear_session()

        #dataset.Y_valid = y_addition(dataset.Y_valid)


        obj_training_reverse_scale = obj_training
        prediction_rescale = prediction

        pred_acc = self.rmse(prediction, real, image_sequence_length)

        pred_acc2 = self.rmse(witheld, witheld_real, image_sequence_length)
        #pred_acc2 = self.mse(prediction, real, image_sequence_length)
        sign_loss = self.customLoss(prediction, real, image_sequence_length)
        sign_loss2 = self.customLoss2(prediction, real, image_sequence_length)
        x_err, x_max = self.x_error(prediction, real, image_sequence_length)
        y_err, y_max = self.y_error(prediction, real, image_sequence_length)


        print("\n")
        print("\n")
        score.append(pred_acc)
        score.append(pred_acc2)
        #score.append(pred_acc3)
        score.append(sign_loss)
        score.append(sign_loss2)
        print(score)
        print("\n")
        print("\n")

        # This needs to be found out from GridSim or by diff'n timestamps of images
        t_delta = 0.2
        max_vel = 32.5 # 130 / 5 = 32.5
        l1 = l1_objective(prediction, image_sequence_length)
        l2 = l2_objective(prediction, t_delta, dataset.slide, image_sequence_length)
        l3 = l3_objective(prediction, t_delta, max_vel, dataset.slide, image_sequence_length)
        if math.isnan(l1) or  math.isnan(l2) or  math.isnan(l3):
            print("One or more objectives were stored as NaN, exiting...")
            sys.exit()

        #L = [pred_acc, l2, l3]
        L = [pred_acc, l1]

        K.clear_session()

        return pred_acc, x_err, x_max, y_err, y_max, L, score

    # Training for Naive approach
    def train_and_score(self, model_train, dataset, path, i, gen_max, run_n):
        logging.info("Getting training samples")
        logging.info("Compling Keras model")

        batch_size = self.geneparam['batch_size']
        epochs = self.geneparam['epochs']

        image_sequence_length = int((dataset.number_of_classes + 2) /2)
        parameters = list()
        file_name = ''

        for p in self.all_possible_genes:
            if p is 'batch_size':
                continue
            elif p is 'epochs':
                continue
            else:
                parameters.append(self.geneparam[p])
                file_name += str(self.geneparam[p]) + '_'

        print(parameters)
        print("")
        input_shape = np.shape(dataset.X_train)
        model = model_train(input_shape, parameters)

        parameters.append(self.geneparam['batch_size'])
        parameters.append(self.geneparam['epochs'])
        # Helper: Early stopping.
        early_stopper = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, verbose=0, mode='auto')
        # FS: 18/07/2021: This is problematic as it is set up for classification
        #                 I will comment out line 49 and 50 from training_history_plot
        history = TrainingHistoryPlot(path, dataset, parameters, i)
        print("Y_train [0] just before fit: " + str(dataset.Y_train.shape[0]))
        print("Y_train [1] just before fit: " + str(dataset.Y_train.shape[1]))
        print("Y_valid [0] just before fit: " + str(dataset.Y_valid.shape[0]))
        print("Y_valid [1] just before fit: " + str(dataset.Y_valid.shape[1]))

        #x_train_list = []
        #x_train_shape = []
        #x_valid_list = []

        #x_train_list = list(dataset.X_train.swapaxes(0, 1))
        #x_test_list = list(dataset.X_valid.swapaxes(0, 1))
        #x_valid_list = list(dataset.X_valid.swapaxes(0, 1))

        #for i in range(len(x_train_list)):
        #    print(np.shape(x_train_list[i]))

        #print(len(x_train_list))
        #print(len(x_train_list[0]))
        np.set_printoptions(threshold=np.inf)
        #print(dataset.X_train[0])
        #sys.exit()

        model.fit(dataset.X_train, dataset.Y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  validation_data=(dataset.X_valid, dataset.Y_valid),
                  callbacks=[early_stopper, history]
                  )

        print("Batch sizem: " + str(batch_size))
        model.save(filepath = str(path) + '/models/model_' + str(file_name) + '_gen_' + str(i) + '_run_' + str(run_n) + '_' + str(self.u_ID) + '.h5')
        #sys.exit()
        score = model.evaluate(dataset.X_valid, dataset.Y_valid, verbose=0)
        prediction = model.predict(dataset.X_valid)

        #results = model.evaluate(x_test, y_test, batch_size=128)


        print("print length of predictions: " + str(prediction.shape[0]))
        #print(prediction)
        #print(dataset.Y_train)
        real = copy.deepcopy(dataset.Y_valid)
        #dataset.Y_train
        def denormalize(array, max, min):
            return array*(max - min) + min

        def denormalize_x(array, min, max):
            return array[:, ::2]*(max - min) + min

        def denormalize_y(array, min, max):
            return array[:, 1::2]*(max - min) + min

        def y_addition(array):
            array_copy = copy.deepcopy(array)
            for i in range(1, int(array.shape[1]/2) ):
                #print(i)
                array_copy[:, i*2 + 1] = array_copy[:, i*2 + 1] + array_copy[:, (i*2 -1)]
                array_copy[:, i*2] = array_copy[:, i*2] + array_copy[:, (i*2 - 2)]
            array = array_copy
            return array

        #real_rescale = denormalize(real, dataset.ymax, dataset.ymin)
        #prediction_rescale = denormalize(prediction, dataset.ymax, dataset.ymin)

        np.set_printoptions(threshold=np.inf)
        #print(real_rescale)
        #print('prediction')
        #print(prediction_rescale)



        # Revert our prediction with ymin and ymax

        # While our prediction is done solely on our validation set (test is withheld), our objective valculation should be
        # based on the training set
        #     1) We would bias our evolutionary stratregy if we used validation
        #     2) Our 3rd objective requires the training data
        obj_training = model.predict(dataset.X_train)
        #obj_training_reverse_scale = denormalize(obj_training, dataset.ymax, dataset.ymin)

        obj_training[:, ::2]  = denormalize_x(obj_training, dataset.superscaler_x_min, dataset.superscaler_x_max)
        prediction[:, ::2] = denormalize_x(prediction, dataset.superscaler_x_min, dataset.superscaler_x_max)
        real[:, ::2] = denormalize_x(real, dataset.superscaler_x_min, dataset.superscaler_x_max)
        #dataset.Y_valid[:, ::2] = denormalize_x(dataset.Y_valid, data_set.superscaler_x_min, dataset.superscaler_x_max)

        obj_training[:, 1::2]  = denormalize_y(obj_training, dataset.superscaler_y_min, dataset.superscaler_y_max)
        prediction[:, 1::2]  = denormalize_y(prediction, dataset.superscaler_y_min, dataset.superscaler_y_max)
        real[:, 1::2]  = denormalize_y(real, dataset.superscaler_y_min, dataset.superscaler_y_max)
        #dataset.Y_valid[:, 1::2]  = normalize_y(dataset.Y_valid, dataset.superscaler_y_min, dataset.superscaler_y_max)

        obj_training = y_addition(obj_training)
        prediction = y_addition(prediction)
        real = y_addition(real)

        #pred_acc = self.rmse(prediction, real, image_sequence_length)
        pred_acc = self.customLoss(prediction, real, image_sequence_length)
        x_err, x_max = self.x_error(prediction, real, image_sequence_length)
        y_err, y_max = self.y_error(prediction, real, image_sequence_length)

        #np.set_printoptions(precision=3)
        #for i in range(len(real)):
        #    print(str(prediction[i]) + '\r')
        #    print(str(real[i]) + '\r')
        #    print("\n")
        #dataset.Y_valid = y_addition(dataset.Y_valid)


        obj_training_reverse_scale = obj_training
        prediction_rescale = prediction

        np.set_printoptions(threshold=np.inf)
        #print(prediction_rescale)

        # This needs to be found out from GridSim or by diff'n timestamps of images
        t_delta = 0.2
        max_vel = 32.5 # 130 / 5 = 32.5
        l1 = l1_objective(prediction, image_sequence_length)
        l2 = l2_objective(prediction, t_delta, dataset.slide, image_sequence_length)
        l3 = l3_objective(prediction, t_delta, max_vel, dataset.slide, image_sequence_length)
        if math.isnan(l1) or  math.isnan(l2):
            print("One or more objectives were stored as NaN, exiting...")
            sys.exit()

        # to get true velocity average x5
        #l3 = l3 * 5
        #L = [pred_acc, l2, l3]
        L = [pred_acc, l1]
        print(L)
        print(pred_acc)
        #sys.exit()
        K.clear_session()
        # we do not care about keeping any of this in memory -
        # we just need to know the final scores and the architecture

        # 1 is accuracy. 0 is loss.
        return pred_acc, x_err, x_max, y_err, y_max, L, score

    def rmse(self, pred, Y_train, image_sequence_length):
        acc_list = []
        x_tau = image_sequence_length
        for i in range(pred.shape[0]):
            temp = 0.0
            for j in range(x_tau):
                P_hat_y = pred[i][j] # starts at 0
                P_y = Y_train[i][j]
                temp += np.sqrt((P_hat_y - P_y)**2)
            acc_list.append(temp/x_tau)
        pred_acc = sum(acc_list)/pred.shape[0]
        return pred_acc

    # Just for testing purposes
    def mse(self, pred, Y_train, image_sequence_length):
        acc_list = []
        x_tau = image_sequence_length
        for i in range(pred.shape[0]):
            temp = 0.0
            for j in range(x_tau):
                P_hat_y = pred[i][j] # starts at 0
                P_y = Y_train[i][j]
                temp += (P_hat_y - P_y)**2
            acc_list.append(temp/x_tau)
        pred_acc = sum(acc_list)/pred.shape[0]
        return pred_acc

    def rmse_y_asymm(self, pred, Y_train, image_sequence_length):
        acc_list = []
        x_tau = (image_sequence_length-1)*2 -1
        #sign_count = 1
        for i in range(pred.shape[0]):
            temp = 0.0
            for j in range(2, x_tau, 2):
                #P_hat_x = pred[i][j - 1] # starts at 1
                P_hat_x = pred[i][j - 2] # starts at 0
                #P_x = Y_train[i][j - 1]
                P_x = Y_train[i][j - 2]
                temp += np.sqrt((np.abs(P_hat_x) - np.abs(P_x))**2)
                #if (np.sign(P_hat_x) == np.sign(P_hat_x)):
                #    sign_count +=1
            acc_list.append(temp/image_sequence_length)
        pred_acc = sum(acc_list)/pred.shape[0]
        return pred_acc


    def customLoss(self, pred, real, image_sequence_length):
        acc_list = []
        x_tau = (image_sequence_length-1)*2 -1
        #sign_count = 1
        for i in range(pred.shape[0]):
            temp_x = 0.0
            temp_y = 0.0
            sign_count = 1
            for j in range(2, x_tau, 2):
                #P_hat_y = pred[i][j - 1] # starts at 1
                P_hat_x = pred[i][j - 2] # starts at 0
                #P_y = real[i][j - 1]
                P_x = real[i][j - 2]
                temp_x += np.sqrt((np.abs(P_hat_x) - np.abs(P_x))**2)
                #temp_y += np.sqrt((P_hat_y - P_y)**2)
                if (np.sign(P_hat_x) == np.sign(P_x)):
                    sign_count +=1
                temp_x = temp_x / sign_count
                #print(sign_count)
            acc_list.append((temp_x)/image_sequence_length)
        return sum(acc_list)/pred.shape[0]

    def customLoss2(self, pred, real, image_sequence_length):
        acc_list = []
        x_tau = (image_sequence_length-1)*2 -1
        #sign_count = 1
        for i in range(pred.shape[0]):
            temp_x = 0.0
            temp_y = 0.0
            sign_count = 1
            for j in range(2, x_tau, 2):
                #P_hat_y = pred[i][j - 1] # starts at 1
                P_hat_x = pred[i][j - 2] # starts at 0
                #P_y = real[i][j - 1]
                P_x = real[i][j - 2]
                temp_x += np.sqrt((P_hat_x - P_x)**2)
                #temp_y += np.sqrt((P_hat_y - P_y)**2)
                if (np.sign(P_hat_x) == np.sign(P_x)):
                    sign_count +=1
                temp_x = temp_x / sign_count
                #print(sign_count)
            acc_list.append((temp_x)/image_sequence_length)
        return sum(acc_list)/pred.shape[0]

    def x_error(self, pred, Y_train, image_sequence_length):
        acc_list = []
        x_tau = (image_sequence_length-1)*2 -1
        for i in range(pred.shape[0]):
            temp = 0.0
            for j in range(2, x_tau, 2):
                P_hat_x = pred[i][j - 2] # starts at 1
                P_x = Y_train[i][j - 2]
                temp += np.abs((P_hat_x  - P_x))
            acc_list.append(temp/image_sequence_length)
        x_err = sum(acc_list)/pred.shape[0]
        x_max = np.max(acc_list)
        return x_err, x_max

    def y_error(self, pred, Y_train, image_sequence_length):
        acc_list = []
        x_tau = (image_sequence_length-1)*2 -1
        for i in range(pred.shape[0]):
            temp = 0.0
            for j in range(2, x_tau, 2):
                P_hat_y = pred[i][j - 1] # starts at 0
                P_y = Y_train[i][j - 1]
                temp += np.abs((P_hat_y  - P_y))
            acc_list.append(temp/image_sequence_length)
        y_err = sum(acc_list)/pred.shape[0]
        y_max = np.max(acc_list)
        return y_err, y_max

    def train(self, model, trainingset, path, i, gen_max, run_n):
        #don't bother retraining ones we already trained
        if self.accuracy == 0.0:
            self.accuracy, self.x_err, self.x_max, self.y_err, self.y_max, self.fitness_vector, self.score = self.train_and_score(model, trainingset, path, i, gen_max, run_n)

    def train_short(self, model, trainingset, path, i, gen_max, run_n, pop_index):
        #don't bother retraining ones we already trained
        if self.accuracy == 0.0:
            self.accuracy,self.x_err,self.x_max,self.y_err,self.y_max,self.fitness_vector, self.score = self.train_and_score_simplified(model, trainingset, path, i, gen_max, run_n, pop_index)

    def print_genome(self):
        """Print out a genome."""
        self.print_geneparam()
        logging.info("Acc: %.2f%%" % (self.accuracy * 100))
        if len(self.fitness_vector) == 3:
            logging.info("Fitness Vector: %d %d %d" % (self.fitness_vector[0], self.fitness_vector[1], self.fitness_vector[2]))
        elif len(self.fitness_vector) == 2:
            logging.info("Fitness Vector: %d %d" % (self.fitness_vector[0], self.fitness_vector[1]))
        logging.info("UniID: %d" % self.u_ID)
        logging.info("Mom and Dad: %d %d" % (self.parents[0], self.parents[1]))
        logging.info("Gen: %d" % self.generation)
        logging.info("Hash: %s" % self.hash)
        # wont be applicable in naive approach but print out anyway
        logging.info("Dom. Rank: %s" % str(self.rank))
        logging.info("Dom. count: %s" % str(self.domination_count))
        logging.info("Crowd. Dist: %s" % str(self.crowding_distance))

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
            # Y_test [0.1, 5.5, 0.3, 10.7]
            # Since our start point is implicitly always [0.0, 0.0]
            # (image_sequence_length-1)*2 -1 last coordinate
            x_tau = (image_sequence_length-1)*2 -2
            # (image_sequence_length-1)*2 -1 Second last coordinate
            y_tau = (image_sequence_length-1)*2 -1
            # P_ego <t+0> - P_dest <t+tau>
            temp += np.sqrt((p[i][x_tau] - 0.0)**2 + (p[i][y_tau] - 0.0)**2)
            flag = 0
        # these go up by 2,  P ego <t+i> - P dest <t+tau> , where i > 0
        for j in range(2, x_tau, 2):
            P_dest_x = p[i][x_tau] # starts at 3
            P_ego_x = p[i][j - 2] # starts at 1
            P_dest_y = p[i][y_tau] # starts at 2
            P_ego_y = p[i][j - 1] # starts at 0
            temp += np.sqrt((P_ego_x  - P_dest_x)**2 + (P_ego_y - P_dest_y)**2)
        dest_list.append(temp/image_sequence_length)
    l1 = sum(dest_list)/(p.shape[0]+1)
    return l1

def l2_objective(p, t_delta, slide, image_sequence_length):
    """
    Calculate Eqn 4. NeuroTrajectory max angular velocity
    :params: Numpy array of input data
    :return: list, each element is the lateral velocity corresponding to the ith sequence of OGs
    """
    # Calculate velocity as rate of change in position across fixed sequence length
    l2 = 0.0
    vd = []
    vector_len = (image_sequence_length-3)*2+1
    t_idx = -1
    idx = -1
    for i in range(p.shape[0]):
        temp = 0.0
        flag = 1
        t_idx += 1
        if slide == True:
            t_idx = idx + 1
        # np.arctan2(b[i+1][0] - b[i][0], b[i+1][1] - b[i][1]) - np.arctan2(b[i][0] - b[i-1][0], b[i][1] - b[i-1][1]))/0.2
        if flag == 1:
            # Since our start point is implicitly always [0.0, 0.0]
            idx = t_idx
            #
            temp += np.abs((np.arctan2(p[i][2] - p[i][0], p[i][3] - p[i][1]) - np.arctan2(p[i][0] - 0.0, p[i][1] - 0.0))/t_delta)
        if image_sequence_length > 3:
            for j in range(2,vector_len, 2):
                temp += np.abs((np.arctan2(p[i][j+2] - p[i][j], p[i][j+3] - p[i][j+1]) - np.arctan2(p[i][j] - p[i][j-2], p[i][j+1] - p[i][j-1]))/t_delta)
        vd.append(temp)
    l2 = sum(vd)/(p.shape[0]+1)
    return l2


def l3_objective(p, t_delta, max_vel, slide, image_sequence_length):
    """
    Calculate Eqn 5. NeuroTrajectory longtitudinal velocity
    :params: Numpy array of input data
    :return: list, each element is the longtitudinal velocity corresponding to the ith sequence of OGs
    """
    # Calculate velocity as rate of change in position across fixed sequence length
    l3 = 0.0
    vf = []
    vector_len = (image_sequence_length-1)*2 -1
    t_idx = -1
    idx = -1
    for i in range(p.shape[0]):
        temp = 0.0
        flag = 1
        t_idx += 1
        if slide == True:
            t_idx = idx + 1
        if flag == 1:
            # Since our start point is implicitly always [0.0, 0.0]
            idx = t_idx
            temp += np.abs(max_vel - ((p[i][1] - 0.0)/t_delta))
            flag = 0
        # these go up by 2
        for j in range(2,vector_len, 2):
            idx += 1
            temp += np.abs(max_vel - ((p[i][j+1] - p[i][j - 1])/t_delta))
        vf.append(temp/image_sequence_length)
    #print(vf)
    l3 = sum(vf)/(p.shape[0]+1)
    return l3

def l3_objective_old(p, t_delta_train, slide, image_sequence_length):
    """
    Calculate Eqn 5. NeuroTrajectory longtitudinal velocity
    :params: Numpy array of input data
    :return: list, each element is the longtitudinal velocity corresponding to the ith sequence of OGs
    """
    # Calculate velocity as rate of change in position across fixed sequence length
    l3 = 0.0
    vf = []
    vector_len = (image_sequence_length-1)*2 -1
    t_idx = -1
    idx = -1
    for i in range(p.shape[0]):
        temp = 0.0
        flag = 1
        t_idx += 1
        if slide == True:
            t_idx = idx + 1
        if flag == 1:
            # Since our start point is implicitly always [0.0, 0.0]
            idx = t_idx
            temp += (p[i][1] - 0.0)/t_delta_train[idx]
            flag = 0
        # these go up by 2
        for j in range(2,vector_len, 2):
            idx += 1
            temp += (p[i][j] - p[i][j - 2])/t_delta_train[idx]
        vf.append(temp/image_sequence_length)
    #print(vf)
    l3 = sum(vf)/(p.shape[0]+1)
    return l3

def calculate_velocities(p, t_delta_train, slide, image_sequence_length):
    """
    Calculate Eqn 5. NeuroTrajectory longtitudinal velocity
    :params: Numpy array of input data
    :return: list, each element is the longtitudinal velocity corresponding to the ith sequence of OGs
    """
    # Calculate velocity as rate of change in position across fixed sequence length
    l3 = 0.0
    vf = []
    vector_len = (image_sequence_length-1)*2 -1
    t_idx = -1
    idx = -1
    for i in range(p.shape[0]):
        temp = 0.0
        flag = 1
        t_idx += 1
        if slide == True:
            t_idx = idx + 1
        if flag == 1:
            # Since our start point is implicitly always [0.0, 0.0]
            idx = t_idx
            temp += (p[i][1] - 0.0)/t_delta_train[idx]
            flag = 0
        # these go up by 2
        for j in range(2,vector_len):
            idx += 1
            temp += (p[i][j] - p[i][j - 2])/t_delta_train[idx]
        vf.append((temp/image_sequence_length))
    return vf
