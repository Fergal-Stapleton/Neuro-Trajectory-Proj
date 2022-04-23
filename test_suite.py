"""
Test suite - series of functions to insure various functions and methods perform correctly
"""

from evolver_moead import Evolver_moead
import numpy as np
import matplotlib.pyplot as plt
import sys


population = 25
population2 = 45
population3 = 28

PARAMETERS_LSTM = {#'hidden_units': [8, 16, 32, 64],
                   'hidden_units': [100, 125, 150, 175, 200, 225, 250],
                   'dropout_parameter': [0.2, 0.25, 0.3, 0.35, 0.4, 0.5],
                   #'batch_size': [8, 16, 24, 32],
                   'momentum': [0.8, 0.85, 0.9, 0.95],
                   #'batch_size': [25, 32],
                   'batch_size': [50, 75, 100, 125],
                   'epochs': [10, 20, 30, 40, 50],
                   #'epochs': [5, 10],
                   'loss_function': ['mean_squared_error', 'logcosh'],
                   'optimizer': ['rmsprop', 'nadam', 'sgd', 'adagrad', 'adadelta', 'adam', 'adamax'],
                   'lstm_cells':[1, 2, 3, 4],
                   'dropout': [0.05, 0.1, 0.15, 0.2, 0.25],
                   'cnn_flattened_layer_1': [256, 512, 768, 1024],
                   'cnn_flattened_layer_2': [256, 512, 768, 1024],
                   'lstm_flattened_layer_1': [64, 128, 256, 512],
                   'lstm_flattened_layer_2': [64, 128, 256, 512],
                   }

# sum of first n natural numbers
def sofnnn(n):
    return (float(n)*(float(n)+1))/2

def check_pop_size(s):
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



def test_weight_init(test_obj):
    #test_obj.
    pass

def plot_3d(weight_array, colours):
    ax = plt.axes(projection='3d')
    #print(colours)
    ax.scatter(weight_array[0], weight_array[1], weight_array[2], c=colours, linewidth=0.5)
    ax.view_init(45,0)
    plt.draw()
    plt.show()
    plt.pause(1)
    plt.clf()

def main():
    test_obj = Evolver_moead(PARAMETERS_LSTM)
    test_obj_2 = Evolver_moead(PARAMETERS_LSTM)
    test_obj_3 = Evolver_moead(PARAMETERS_LSTM)

    # Test when requiring larger pop size
    test_genomes = test_obj.create_population(population)

    # Test when requiring very small pop size, i.e weight initializing
    test_genomes_2 = test_obj_2.create_population(population2)
    print('number of objectives:'+ str(test_obj.m))
    print('population size:'+ str(test_obj.pop_size))
    print('dist_mat size: '+ str(np.array(test_obj.dist_mat).shape))
    #assert len(test_obj.dist_mat[0]) == len(test_obj.dist_mat[0]), 'Dimensional mismatch in distance matrix'
    print('neighbour_table size: '+ str(np.array(test_obj.neighbour_table).shape))
    print('weights size: '+ str(np.array(test_obj.weights).shape))
    print('ideal_point size: '+ str(np.array(test_obj.ideal_point).shape))

    test_obj_2.init_weights()
    print(test_obj_2.weights)
    check_pop_size(population2)
    weight_array = np.array(test_obj_2.weights).ravel(order='F').reshape((3, population2))
    #print(sofnnn(3))
    #print(sofnnn(4))
    #print(sofnnn(5))
    #print(sofnnn(6))
    #print(sofnnn(7))
    #print(sofnnn(8))

    print(weight_array)
    ax = plt.axes(projection='3d')
    ax.scatter(weight_array[0], weight_array[1], weight_array[2], linewidth=0.5);
    #plt.show()
    test_obj_2.init_neighbour()
    dist_mat_rounded = [[round(val, 2) for val in sublst] for sublst in test_obj_2.dist_mat]
    for i in dist_mat_rounded:
        print(i)
    print(test_obj_2.neighbour_table)

    for list_closest in test_obj_2.neighbour_table:
        colours = ['#0000FF']*population2
        count = 0
        for i in list_closest:
            colours[i] = '#FF0000'
            if count == 0:
                colours[i] = '#000000'
                count = 1
        plot_3d(weight_array, colours)

    test_genomes_3 = test_obj_3.create_population(population3)
    test_obj_3.init_weights()
    test_obj_3.init_neighbour()

    local_weights = test_obj_3.weights[15]
    print(local_weights)


    #print(test_obj.nadir_point)

    f0 = [0.21, 0.35, 0.44]
    f1 = [0.04, 0.17, 0.23]
    test_obj_3.ideal = [0.1, 0.1, 0.1]

    print("")
    print("TCH - test")
    d = test_obj_3.tcheycheffScalarObj(local_weights, f0)
    print(f0)
    print(d)
    print("")
    e = test_obj_3.tcheycheffScalarObj(local_weights, f1)
    print(f1)
    print(e)

    print("")

    print("")
    print("WGT - test")
    d = test_obj_3.weightedScalarObj(local_weights, f0)
    print(f0)
    print(d)
    print("")
    e = test_obj_3.weightedScalarObj(local_weights, f1)
    print(f1)
    print(e)





if __name__ == '__main__':
    #var = str(int(sys.argv[1]) % 2)
    #with tf.device('/device:GPU:'+var):
    main()
