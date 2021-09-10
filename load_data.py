import numpy as np
import pandas as pd
import os
from PIL import Image
from loadTraj import LoadTraj
import sys
from natsort import index_natsorted
import random


def shuffle_in_unison(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)


class LoadData(object):
    def __init__(self, type, data_path, image_height, image_width, image_channels, image_depth, num_classes):
        self.type = type
        self.data_path = data_path
        self.image_height = image_height
        self.image_width = image_width
        self.image_channels = image_channels
        self.image_depth = image_depth
        self.number_of_classes = num_classes
        self.image_sequence_length = int((num_classes + 2) /2)
        self.data_was_loaded = False
        self.data_was_saved = False
        self.X_train = None
        self.Y_train = None
        self.X_valid = None
        self.Y_valid = None
        self.X_test = None
        self.Y_test = None
        # Nee this for caluclating velocity based objectives - delta v = delta x / delta t
        self.t_delta_train = None
        self.t_delta_test = None
        self.t_delta_valid = None
        # Obj. Likely only need Obj_validationm
        self.vel_l = None
        self.vel_a = None
        self.vel_l_train = None
        self.vel_a_train = None
        self.vel_l_test = None
        self.vel_a_test = None
        self.vel_l_valid = None
        self.vel_a_valid = None
        self.Obj_train = None
        self.Obj_test = None
        self.Obj_validation  = None
        # Slide
        self.slide = False
        self.shuffle = True
        # Max and Min - need to revert to real coordinates
        #-5.66668848861903 Max X:51.31507599420365
        #-2.7186277031692385 max = 19.73575844802872
        self.ymin = -1.632622394172697
        self.ymax = 33.11593567960517

        self.load = {"dgn": self.load_dgn_data,
                     "conv3d": self.load_conv3d_data,
                     "lstm_sliding": self.load_lstm_data,
                     "lstm_bucketing": self.load_lstm_data}

        self.create_data_sets()

    def resize_image(self, sample):
        if sample.size != (self.image_width, self.image_height):
            sample = sample.resize((self.image_width, self.image_height), Image.ANTIALIAS)

        sample = np.asarray(sample, dtype="int32")
        sample = np.expand_dims(sample, axis=0)
        return sample

    def get_input_sequences(self, sequences, y, class_idx, source_dir_path_complete, files):
        """
        Creates the image sequences for our input images. class_idx is incremented

        Called as:
            self.get_input_sequences(self.X_train, self.Y_train,class_idx, f + '/', files)
        :params:
        :return:
        """
        # timestamp is not allowed to jump more than 0.3 seconds from one frame to another
        max_timestamp_delta = 300000  # [microseconds]
        sequences_counter = 0
        selected_sequences_per_class = 4500
        #image_sequence_length = int((self.number_of_classes + 2) /2)
        skip = False
        # This will be used to deermine objective velocity
        t_delta_list = []

        final_sequences = np.empty(shape=(0,self.image_sequence_length,self.image_width, self.image_height,self.image_channels))
        # Will convert y data to pandas as it is easier to do comparisons
        colList = []
        colList.append('image')
        print('image sequence length: ' + str(self.image_sequence_length))

        # Create column headers
        for i in range(0,self.image_sequence_length-1):
            colList.append('x'+str(i+1))
            colList.append('y'+str(i+1))

        print(colList)
        df_y = pd.DataFrame(y,  columns=colList)
        df_y = df_y.sort_values(by="image", key=lambda x: np.argsort(index_natsorted(df_y["image"])))
        df_y = df_y.reset_index(drop=True)
        #df_vel_l = pd.DataFrame(vel_l,  columns=['image', 'velocity'])
        #df_vel_l = df_vel_l.sort_values(by="image", key=lambda x: np.argsort(index_natsorted(df_vel_l["image"])))
        #df_vel_l = df_vel_l.reset_index(drop=True)
        #df_vel_a = pd.DataFrame(vel_a,  columns=['image', 'velocity'])
        #df_vel_a = df_vel_a.sort_values(by="image", key=lambda x: np.argsort(index_natsorted(df_vel_a["image"])))
        #df_vel_a = df_vel_a.reset_index(drop=True)
        #print(df_y.head())
        #print(df_vel_l.head())
        #print(df_vel_a.head())
        df_y_process = pd.DataFrame()

        df = pd.read_csv("images/state_buf.txt", sep='\t')
        #pos_l = df['CarPositionY'].to_list()
        # = df['Velocity_angular'].to_list()

        # Test for large jumps in position
        if (df_y['y2'].any().astype(float) > 50):
            print(df_y[df_y['y2'].astype(float) > 50])
        # lets be smart and assume that the indexes may have gaps
        # Why? its is easier to remove images relating to collisions than to restart a simulation (this includes build up to and aftermath)
        #print(files)
        file_index = [f[0].split('_')[0] for f in files]
        #print(file_index)
        #print(files)

        # the next 20 or so lines get the start and end of the gap sequences of a file
        # so.... for instance if we have:
        # input:
        #    file_index = [0, 1, 2, 5, 6, 7, 8, 12, 13, 14, 23, 24, 25, 26, 27, 36, 37, 38]
        # output:
        #    start_list = [0, 5, 12, 23, 36]
        #    endlist = [2, 8, 14, 27, 38]
        # Why is this important? we need to pop the first sequence and last trajectory inbetween these gaps to satisfy
        # the t - tau sequnces and t + tau trajectories conditions
        end_list = []
        start_list = []
        inc = int(file_index[0])
        start_flag = True
        sync_flag = True
        for i in range(len(file_index)):
            #print(file_index[i])
            #print(inc)
            batch_idx = 0
            # As soon as the index and increment go our of sync declare end of batch index
            if (int(file_index[i]) != inc) and sync_flag == True:
                end_list.append(file_index[i-1])
                batch_idx += 1
                sync_flag = False
                start_flag = True
            # re- sync
            while(int(file_index[i]) != inc):
                inc += 1
            # While everything is nice
            if (int(file_index[i]) == inc) and start_flag == True:
                start_list.append(file_index[i])
                start_flag=False
                sync_flag=True
                inc += 1
            elif (int(file_index[i]) == inc) and start_flag == False:
                sync_flag=True
                inc += 1
            if int(file_index[-1]) == inc:
                end_list.append(file_index[-1])

        #if self.slide == True:
        start_idx = 1
        #else:
        #    start_idx = 0
        index_prev = self.get_index_from_file_name(files[start_idx][0])


        # create sequences till the sequence_lenght to last image is processed or till the number of sequences
        # desired by the user is reached
        print(end_list)
        print(start_list)
        final_sequence_file_popped = []
        sequence_list = []
        for j in range(len(start_list)):
            sequences = np.empty(shape=(0,self.image_sequence_length,self.image_width, self.image_height,self.image_channels))
            # We should have a filename which correponds to the start of each image sequence
            # Any filenames left over that do not correspond with this list must be dropped
            # to insure dimensionality is correct
            start_sequence_file = []
            # Create a list of implausaible sequences to remove at the end
            remove_implausible_list = []

            #if self.type is 'lstm_sliding':
            #    slide = True

            # check if there are enough images to build a sequence
            if len(files) < self.image_sequence_length:
                print('Not enough images to build a sequence. Minimum number of consecutive grid images must be :',
                      self.image_sequence_length)
                return -1

            # build up a sequence of x images
            sequence = np.zeros(shape=(1, self.image_sequence_length, self.image_height, self.image_width, self.image_channels))

            #index_prev = self.get_index_from_file_name(files[start_idx][0])
            #timestamp_prev = self.get_timestamp_from_file_name(files[start_idx][0], source_dir_path_complete)
            first_time = False
            first_slides_count = 0
            remove_flag = False
            while ((start_idx + self.image_sequence_length) <= len(files)) and (int(start_list[j]) <= int(file_index[start_idx]) <= int(end_list[j])):
                t_delta_final = 0.0
                t_delta_tmp = 0.0
                #vel_l_tmp = 0.0
                #vel_a_tmp = 0.0

                #print(file_index[start_idx])
                #print(int(end_list[j]))
                # stop parsing data if the requested number of sequences have been collected
                if sequences_counter >= selected_sequences_per_class:
                    break

                sequence_complete = True
                sample_sequence_counter = 0

                # update previous timestamp in case of the sliding mode sequence generation
                #if self.slide is True:
                index_prev = self.get_index_from_file_name(files[start_idx - 1][0])


                # Jump 3 at a time


                temp_sequence_length = self.image_sequence_length

                for idx in range(start_idx, start_idx + self.image_sequence_length):
                    #print(idx)
                    t_delta = 0.0


                    #if(idx == start_idx):
                    #print(files[idx][0])
                    start_sequence_file.append(files[idx - 1][0])
                    #print(df_y[df_y.index[df_y['image'] == str(files[idx][0])]])

                    # get index of current image
                    index_curr = self.get_index_from_file_name(files[idx][0])

                    timestamp_curr = self.get_timestamp_from_file_name(files[idx][0], source_dir_path_complete)
                    timestamp_prev = self.get_timestamp_from_file_name(files[idx - 1][0], source_dir_path_complete)

                    # compute time elapsed since the previous image index
                    delta = index_curr - index_prev
                    t_delta = timestamp_curr - timestamp_prev
                    #pos_l_delta = pos_l[idx] - pos_l[idx-1]
                    #print(pos_l_delta)

                    # save current index
                    index_prev = index_curr


                    # check if the elapsed time is plausible
                    if delta  < 0:
                        print('Implausible index. Image happened in the past (before the previous one). Skip sequence')
                        # Use start_idx as this will be the reference filename that will be appended to start_sequence_file
                        print(files[start_idx-1][0])
                        remove_implausible_list.append(files[start_idx - 1][0])
                        print("")

                        # set the current IDX as the start point for the next sequence
                        #start_idx = idx
                        sequence_complete = False

                        #y = np.delete(y, 1, 0)
                        # drop current sequence since the images are not consecutive
                        break

                    # check if the timestamp did not jumped more than 200 ms
                    if t_delta  > 0.25:
                        print('index jump. Drop current sequence')
                        print(files[start_idx - 1][0])
                        #remove_implausible_list.append(files[start_idx][0])
                        #for seq_i in range(sample_sequence_counter):
                        remove_implausible_list.append(files[start_idx - 1][0])

                        print('index interrupted at: ', timestamp_curr)
                        print('Delta index is : ', t_delta , ' sec')
                        sequence_complete = False
                        # drop current sequence since the imageces are not consecutive
                        break

                    # increment samples in sequence counter
                    sample_sequence_counter += 1
                    #timestamp_prev = timestamp_curr

                    # save image in the current sequence
                    image = Image.open(source_dir_path_complete + files[idx][0])
                    image.thumbnail((self.image_width, self.image_height), Image.ANTIALIAS)
                    image = np.asarray(image, dtype="int32")
                    sequence[0][sample_sequence_counter - 1] = image

                    # set true the sequence complete flag
                    sequence_complete = True

                # BREAK to here
                current = files[start_idx - 1][0]

                # slide the start for the next sequence with 1 position
                if self.slide == True:
                    start_idx += 1
                elif sequence_complete == False:
                    if idx < (int(end_list[j]) - self.image_sequence_length):
                        start_idx = start_idx + self.image_sequence_length
                        #remove_flag == True
                    else:
                        start_idx = idx + 1
                        print("working")
                # slide the start for the next sequence with the processed samples till now (like jump to the next 10 samples)
                elif self.slide == False:
                    start_idx = idx + 1

                if sequence_complete == True:

                    sequence_list.append(current)
                    sequences = np.concatenate((sequences, sequence), axis=0)
                    sequence_output = np.zeros(shape=(1, self.number_of_classes))
                    #sequence_output[0][class_idx] = 1
                    #y = np.concatenate((y, sequence_output), axis=0)
                    sequences_counter += 1

                # refresh the sequence for the next series

                sequence = np.zeros(shape=(1, self.image_sequence_length, self.image_height,
                                           self.image_width, self.image_channels))


            # list comprehension to remove items from one list using another
            #remove_implausible_list.append(files[start_idx][0])
            if remove_implausible_list != []:
                #print("removing")
                final_sequence_file = [x for x in start_sequence_file if x not in remove_implausible_list]
            else:
                final_sequence_file = start_sequence_file


            #print(remove_implausible_list)

            #print(final_sequence_file)

            df_tmp = df_y[df_y['image'].isin(final_sequence_file)]



            df_tmp = df_tmp[:-1]
            if df_y_process.empty:
                df_y_process = df_tmp
            else:
                df_y_process = pd.concat([df_y_process, df_tmp])


            #t_delta_list = t_delta_list[:-self.image_sequence_length]
            #print(final_sequence_file_popped)
            final_sequences = np.concatenate((final_sequences, np.delete(sequences, 1, 0)), axis=0)


        df_y = df_y_process
        #indeces = df_y[df_y['y2'].astype(float) > 20].index.tolist()
        #df_y = df_y[df_y['y2'].astype(float) <= 20]
        #final_sequences = np.delete(final_sequences, indeces, 0)

        df_seq = pd.DataFrame(sequence_list)
        df_seq.to_csv('seq.csv')
        df_y.to_csv('traj.csv')
        #df_vel_l = df_vel_l[df_vel_l['image'].isin(final_sequence_file_popped)]
        #df_vel_a = df_vel_a[df_vel_a['image'].isin(final_sequence_file_popped)]
        #print(df_y.shape[0])


        #sys.exit()
        y = df_y.iloc[:, 1:].to_numpy()
        #vel_l = vel_l[:, 1:]
        #vel_a = vel_a[:, 1:]
        #print(t_delta_list)
        #sys.exit()
        #print(y.shape[0])
        #print(y)
        # Test
        if(final_sequences.shape[0] != y.shape[0]):
            print("Error: the sequence length of image sequences does not match our trajectory instances")
            print("   Source path     " + source_dir_path_complete)
            print("   Sequence dim.   " + str(final_sequences.shape[0])+' '+str(sequences.shape[1]))
            print("   Trajectory dim. " + str(y.shape[0])+' '+str(y.shape[1]))
            print("   Time deltas dim." + str(len(t_delta_list)))
            print(" ")
            sys.exit()
        return final_sequences, y

    def load_lstm_data(self, folder):
        class_idx = 0



        for directories in os.listdir(self.data_path + '/' + folder + '/'):
            dir = os.path.join("", directories)
            f = self.data_path + '/' + folder + '/' + dir
            print('Training class : ', dir)

            # 07/16/2021 FS: The original code here is very poor - GridSim does not seem to save sequentially, for example the files are saved as
            #     ('250_image.png', 1626267753.6708562), ('0_image.png', 1626267753.6728625), ('1_image.png', 1626267753.67582), ('2_image.png', 1626267753.6778226)
            #     Here image 250_image.png was saved before 0_image.png ( potentially there is some code in Grid Sim that saves and exits these but is -1 indexed)
            #     As such this makes the code very buggy when creating sequences. Will need to sort instead based on name prefix (since files are named iteratively and sequentially).
            #           Original:   key=lambda item: item[1]
            #           New:        key=lambda item: np.argsort(index_natsorted(item[0]))
            #     This may be reverted if data source does not follow file name saving convention
            d = {}
            for item in os.listdir(f):
                d[item] = os.path.getctime(f + '/' + item)
            files = sorted(d.items(), key=lambda item: int(item[0].split('_')[0]))
            # And fixing...
            print("files")
            #print(files)

            nr_of_found_samples = len(files)
            print('Found :', nr_of_found_samples)

            if folder == 'training':
                self.X_train, self.Y_train = self.get_input_sequences(self.X_train, self.Y_train,
                                                                      class_idx, f + '/', files)
            elif folder == 'testing':
                self.X_test, self.Y_test = self.get_input_sequences(self.X_test, self.Y_test,
                                                                    class_idx, f + '/', files)
            elif folder == 'validation':
                self.X_valid, self.Y_valid = self.get_input_sequences(self.X_valid, self.Y_valid,
                                                                      class_idx, f + '/', files)
            class_idx += 1

    def load_dgn_data(self, folder):
        selected_images_per_class = 4500
        class_idx = 0

        for directories in os.listdir(self.data_path + '/' + folder):
            loaded_samples = 0
            dir = os.path.join("", directories)
            print(folder, ' class : ', dir)

            files = next(os.walk(self.data_path + '/' + folder + '/' + dir))[2]
            nr_of_found_samples = len(files)
            print('Found :', nr_of_found_samples, ' samples (images)')

            for file in files:
                if loaded_samples >= selected_images_per_class:
                    break

                sample = Image.open(self.data_path + '/' + folder + '/' + dir + '/' + file)
                sample = self.resize_image(sample)

                # NO - this is wrong
                y = np.zeros(shape=(1, self.number_of_classes))
                y[0][class_idx] = 1

                if folder == 'training':
                    print (np.shape(self.X_train), np.shape(sample))
                    self.X_train = np.concatenate((self.X_train, sample), axis=0)
                    self.Y_train = np.concatenate((self.Y_train, y), axis=0)
                elif folder == 'validation':
                    print(np.shape(self.X_valid), np.shape(sample))
                    self.X_valid = np.concatenate((self.X_valid, sample), axis=0)
                    self.Y_valid = np.concatenate((self.Y_valid, y), axis=0)
                elif folder == 'testing':
                    print(np.shape(self.X_test), np.shape(sample))
                    self.X_test = np.concatenate((self.X_test, sample), axis=0)
                    self.Y_test = np.concatenate((self.Y_test, y), axis=0)

                loaded_samples += 1

            class_idx += 1

    def load_conv3d_data(self, folder):
        self.load_dgn_data(folder)

        if folder == 'training':
            self.X_train = self.X_train.reshape(np.shape(self.X_train)[0], self.image_depth, self.image_width,
                                                self.image_height, self.image_channels)
        elif folder == 'validation':
            self.X_valid = self.X_valid.reshape(np.shape(self.X_valid)[0], self.image_depth, self.image_width,
                                                self.image_height, self.image_channels)
        elif folder == 'testing':
            self.X_test = self.X_test.reshape(np.shape(self.X_test)[0], self.image_depth, self.image_width,
                                              self.image_height, self.image_channels)

    def load_new_data(self):
        print(self.data_path)
        #self.number_of_classes = len(next(os.walk(self.data_path + '/training/'))[1])
        print('Number of classes found : ', self.number_of_classes)
        print('Model type : ',self.type)
        # FS: 'is' is not correct, changing to '=='. 'is' should only be used to check objects are the same
        #      as a result this was always reverting to else statement and causing the input array shape to
        #      be incorrect for sequencing
        if self.type == 'lstm_sliding' or self.type == 'lstm_bucketing':
            self.X_train = np.zeros(shape=(0, self.image_sequence_length, self.image_height, self.image_width, self.image_channels))
            self.X_valid = np.zeros(shape=(0, self.image_sequence_length, self.image_height, self.image_width, self.image_channels))
            self.X_test = np.zeros(shape=(0, self.image_sequence_length, self.image_height, self.image_width, self.image_channels))
        else:
            self.X_train = np.zeros(shape=(0, self.image_height, self.image_width, self.image_channels))
            self.X_valid = np.zeros(shape=(0, self.image_height, self.image_width, self.image_channels))
            self.X_test = np.zeros(shape=(0, self.image_height, self.image_width, self.image_channels))

        self.Y_train = np.zeros(shape=(0, self.number_of_classes))
        self.Y_valid = np.zeros(shape=(0, self.number_of_classes))
        self.Y_test = np.zeros(shape=(0, self.number_of_classes))
        #print("Got here")
        print("Y_train after init: ", self.Y_train.shape[1])

        folders = {'training', 'validation', 'testing'}

        # ************************************************************** #
        # FS: 13/07/2021
        # Y_train, Y_test, Y_validation, Obj_train, Obj_test, Obj_validation
        #image_sequence_length = (self.number_of_classes + 2 )/2

        self.Y_train, self.Y_test, self.Y_valid = LoadTraj.getTraj(self.number_of_classes, self.slide)
        print(self.Y_train)
        print(" ")
        print("Trajectory information Loaded (pre-processing)")
        print("-----------------------------")
        print("   Training sequence dim.     " + str(self.Y_train.shape[0])+' '+str(self.Y_train.shape[1]))
        print("   Test sequence dim.         " + str(self.Y_test.shape[0])+' '+str(self.Y_test.shape[1]))
        print("   Validation sequence dim.   " + str(self.Y_valid.shape[0])+' '+str(self.Y_valid.shape[1]))
        print(" ")

        # ************************************************************** #
        # TEST HERE
        # ************************************************************** #

        for f in folders:
            self.load[self.type](f)

        self.Y_train = self.Y_train.astype(float)
        self.Y_test = self.Y_test.astype(float)
        self.Y_valid = self.Y_valid.astype(float)


        # Ensure our y data is a sequence ahead e.g t + tau and our images are a sequence behind t - tau
        # We do this by slicing last row of our X data and first row of our Y data


        #if self.shuffle == True:
        #    self.X_train, self.Y_train = self.shuffler(self.X_train, self.Y_train)
        #    self.X_test, self.Y_test = self.shuffler(self.X_test, self.Y_test)
        #    self.X_valid, self.Y_valid = self.shuffler(self.X_valid, self.Y_valid)

        # Should just scale using training
        self.ymin = np.amin(self.Y_train)
        self.ymax = np.amax(self.Y_train)

        def normalizer(array, min, max):
           return (array - min)/ (max - min)

        self.Y_train = normalizer(self.Y_train, self.ymin, self.ymax)
        self.Y_test = normalizer(self.Y_test, self.ymin, self.ymax)
        self.Y_valid = normalizer(self.Y_valid, self.ymin, self.ymax)

        #print(unravel_index(self.Y_train.argmax(), self.Y_train.shape))
        #sys.exit()
        # We know 0 min and 255 max so no need to find max and min, just hardcode this
        self.X_train = normalizer(self.X_train, 0, 255)
        self.X_test = normalizer(self.X_test, 0, 255)
        self.X_valid = normalizer(self.X_valid, 0, 255)

        # to confirm everything makes sense output our standardized x
        xminstd = np.amin(self.X_train)
        xmaxstd = np.amax(self.X_train)

        pd.DataFrame(self.Y_train).to_csv("Y_train")
        pd.DataFrame(self.Y_valid).to_csv("Y_valid")

        print("Trajectory information Loaded (post-processing)")
        print("-----------------------------")
        print("                                 X       Y ")
        print("   Training sequence dim.     " + str(self.X_train.shape[0])+' '+str(self.X_train.shape[1]) + '   ' + str(self.Y_train.shape[0])+' '+str(self.Y_train.shape[1]))
        print("   Training seq. data type    " + str(self.Y_train.dtype))
        print("   Test Sequence dim.         " + str(self.X_test.shape[0])+' '+str(self.X_test.shape[1]) + '   ' + str(self.Y_test.shape[0])+' '+str(self.Y_test.shape[1]))
        print("   Test seq. data type        " + str(self.Y_test.dtype))
        print("   Validation Sequence dim.   " + str(self.X_valid.shape[0])+' '+str(self.X_valid.shape[1]) + '   '+ str(self.Y_valid.shape[0])+' '+str(self.Y_valid.shape[1]))
        print("   Validation seq. data type  " + str(self.Y_valid.dtype))
        print("   Min Y:" + str(self.ymin) + " Max X:" + str(self.ymax))
        print("   Std Min X:" + str(xminstd) + " STD Max X:" + str(xmaxstd))
        print(" ")

        # Early Exit for Testing purposes
        #sys.exit()
        self.data_was_loaded = True
        self.save_processed_data()

    def load_processed_data(self):
        path_to_data = './data_sets/' + self.type
        self.X_train = np.array(np.load(path_to_data + '/X_train.npy'), dtype=np.float32)
        self.X_valid = np.array(np.load(path_to_data + '/X_valid.npy'), dtype=np.float32)
        self.X_test = np.array(np.load(path_to_data + '/X_test.npy'), dtype=np.float32)
        self.Y_train = np.array(np.load(path_to_data + '/Y_train.npy'), dtype=np.float32)
        self.Y_valid = np.array(np.load(path_to_data + '/Y_valid.npy'), dtype=np.float32)
        self.Y_test = np.array(np.load(path_to_data + '/Y_test.npy'), dtype=np.float32)
        self.data_was_loaded = True

    def save_processed_data(self):
        path_where_to_save = './data_sets/' + self.type
        np.save(path_where_to_save + '/X_train', self.X_train)
        np.save(path_where_to_save + '/Y_train', self.Y_train)
        np.save(path_where_to_save + '/X_valid', self.X_valid)
        np.save(path_where_to_save + '/Y_valid', self.Y_valid)
        np.save(path_where_to_save + '/X_test', self.X_test)
        np.save(path_where_to_save + '/Y_test', self.Y_test)

    def create_data_sets(self):
        self.check_dir('./data_sets')
        self.check_dir('./data_sets/dgn')
        self.check_dir('./data_sets/conv3d')
        self.check_dir('./data_sets/lstm_bucketing')
        self.check_dir('./data_sets/lstm_sliding')

    @staticmethod
    def get_index_from_file_name(filename):
        end_of_index = filename.find('_')
        index = int(filename[:end_of_index])

        return index

    @staticmethod
    def get_timestamp_from_file_name(filename, path):
        # path has last '/' added to it when called...
        timestamp = os.path.getmtime(str(path + filename))

        return timestamp

    @staticmethod
    def check_dir(path):
        if not os.path.exists(path):
            os.makedirs(path)

    @staticmethod
    def shuffler(X, Y):
        assert len(X) == len(Y)
        # Easier to use indexing rather than trying to re-combine dimensionally mismatched arrays
        ix_perm = np.random.permutation(len(X))
        return X[ix_perm], Y[ix_perm]
