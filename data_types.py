DATA_SET_INFO = {#data_set_path': '/ichec/home/users/fergals/neuroTraj/Neuro-Trajectory-Proj/images_splited',
                 #'data_set_path': '/ichec/work/nuim01/fergals/neuroTraj2/images_splited',
                 #'data_set_path': '/ichec/work/mucom002c/fergals/neuroTraj2/images_splited',
                 'data_set_path': 'D:/Main/GitHub/Neuro-Trajectory/Neuro-Trajectory-Proj/images_splited',
                 'image_height': 128,
                 'image_width': 128,
                 'image_channels': 3,
                 'image_depth': 1,
                 #'num_classes': 4,
                 #'num_classes': 8,
				 'num_classes': 14,
                 #'num_classes': 18,
                 # x1 and y1 are alway zero so no need to predict
                 #'classes_name': ['x2', 'y2', 'x3', 'y3']
                 #'classes_name': ['x2', 'y2', 'x3', 'y3', 'x4', 'y4', 'x5', 'y5']
				 'classes_name': ['x2', 'y2', 'x3', 'y3', 'x4', 'y4', 'x5', 'y5','x6', 'y6', 'x7', 'y7', 'x8', 'y8']
                 #'classes_name': ['x2', 'y2', 'x3', 'y3', 'x4', 'y4', 'x5', 'y5','x6', 'y6', 'x7', 'y7', 'x8', 'y8', 'x9', 'y9', 'x10', 'y10']
                 #'classes_name': ['city', 'country_road', 'highway', 'parking_lot', 'traffic_jam']
                 }

#PATH_SAVE_FIG = '/ichec/work/mucom002c/fergals/neuroTraj2/train/'
#PATH_SAVE_FIG = '/ichec/work/nuim01/fergals/neuroTraj2/train/'
PATH_SAVE_FIG = './train/'



PARAMETERS_LSTM = {#'hidden_units': [8, 16, 32, 64],
                   'hidden_units': [100, 125, 150, 175, 200, 225, 250],
                   'dropout_parameter': [0.2, 0.25, 0.3, 0.35, 0.4, 0.5],
                   #'batch_size': [8, 16, 24, 32],
                   'momentum': [0.8, 0.85, 0.9, 0.95],
                   'batch_size': [50, 75, 100, 125],
                   #'batch_size': [64, 128, 192, 256],
                   'epochs': [10, 20, 30, 40, 50],
                   'loss_function': ['mean_squared_error', 'logcosh'],
                   'optimizer': ['rmsprop', 'nadam', 'sgd', 'adagrad', 'adadelta', 'adam', 'adamax'],
                   'lstm_cells':[1, 2, 3, 4],
                   'dropout': [0.05, 0.1, 0.15, 0.2, 0.25],
                   'cnn_flattened_layer_1': [256, 512, 768, 1024],
                   'cnn_flattened_layer_2': [256, 512, 768, 1024],
                   'lstm_flattened_layer_1': [64, 128, 256, 512],
                   'lstm_flattened_layer_2': [64, 128, 256, 512]
                   }
