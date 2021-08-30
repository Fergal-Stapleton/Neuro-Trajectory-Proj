import numpy as np
import csv
import pygame
from PIL import Image
import os
import cv2
import sys
from datetime import datetime

def read_coords(fname):
    with open(fname) as csvFile:
        car_positions = []
        readCSV = csv.reader(csvFile, delimiter=',')
        for row in readCSV:
            car_position = [float(v) for v in row]
            car_positions.append(car_position)
        return car_positions


def read_traffic_data(fname):

    with open(fname) as csvFile:
        trajectories_data = []
        readCSV = csv.reader(csvFile, delimiter=',')
        for row in readCSV:
            trajectory_data = [v for v in row]
            trajectories_data.append(trajectory_data)
        return trajectories_data


def read_data_from_dataset(fname, desired_data):
    flag = check_valid_csv(fname)
    if flag == 'empty':
        raise ValueError('empty dataset csv')
    else:
        pass

    with open(fname) as refCsv:
        data = {}
        reader = csv.DictReader(refCsv, delimiter='\t')
        for row in reader:
            for header, value in row.items():
                try:
                    if header in desired_data:
                        data[header].append(value)
                except KeyError:
                    data[header] = [value]
        return data


def write_data(fname, *args):
    with open(fname, "a") as csvfile:
        writer = csv.writer(csvfile, lineterminator='\n')
        export_list = np.array([])
        for arg in args:
            export_list = np.append(export_list, arg)
        export_list = list(export_list)
        writer.writerow(export_list)


def check_valid_csv(fname):

    try:
        split_path = fname.rsplit('/', 1)[0]
        os.makedirs(split_path)
    except OSError:
        pass

    try:
        open(fname, 'x')
    except FileExistsError:
        pass

    with open(fname, 'r') as csvfile:
        reader = csv.reader(csvfile)
        try:
            first_line = next(reader)
            return 'not_empty'
        except StopIteration:
            return 'empty'


def write_state_buf(fname, args):
    # args[6] will be teh other car positions
    cars = args[7]
    car_headers = []

    for i in range(len(args[7])):
        car_headers.append('x'+str(i+1))
        car_headers.append('y'+str(i+1))

    if len(car_headers) > 0 and len(cars) == 0:
        raise AssertionError("Traffic car trajectories not recorded")
        sys.exit()



    #now = datetime.now()
    if(args[1]>40000):
        sys.exit()

    #current_time = now.strftime("%H:%M:%S")
    #print("Current Time =", current_time)
    #print(cars[0].position.x)
    #print(car_headers)

    results_dict = {'CarPositionX': args[0],
                     'CarPositionY': args[1],
                     'CarAngle': args[2],
                     'Acceleration': args[3],
                     'Velocity_long': args[4],
                     'Velocity_angular': args[5],
                     'ImageName': args[6]}

    for i in range(len(args[7])):
        results_dict[car_headers[i*2]] = cars[i].position.x
        results_dict[car_headers[i*2+1]] = cars[i].position.y


    fieldnames = ['CarPositionX', 'CarPositionY', 'CarAngle', 'Acceleration', 'Velocity_long', 'Velocity_angular','ImageName'] + car_headers
    flag = check_valid_csv(fname)
    if flag == 'empty':
        try:
            with open(fname, 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter='\t', lineterminator='\n')
                writer.writeheader()
                writer.writerow(results_dict)
        except :
            pass

    elif flag == 'not_empty':
        try:
            with open(fname, 'a') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter='\t', lineterminator='\n')
                writer.writerow(results_dict)
        except:
            pass


def save_frame(screen, frame_name, path):
    #print("got here")
    try:
        os.makedirs(path)
    except OSError:
        pass

    screendata = pygame.surfarray.array3d(screen)
    screendata = np.rot90(screendata, axes=(0, 1))
    screendata = np.flipud(screendata)
    img = Image.fromarray(screendata)
    img.save(path + '/' + frame_name)


def resize_image(image, size):
    image = pygame.surfarray.array3d(image)
    resized_image = cv2.resize(image, size)
    resized_image = pygame.surfarray.make_surface(resized_image)
    return resized_image


def write_distances_as_int(fname, distances_vector):
    flag = check_valid_csv(fname)
    # if type(distances_vector) != np.array:
    # distances_vector = np.asarray(distances_vector, dtype=np.float32)
    try:
        with open(fname, 'a') as csvfile:
            writer = csv.writer(csvfile, lineterminator='\n')
            distances_vector = distances_vector.astype(int)
            writer.writerow(distances_vector)
    except :
        raise OSError('path to distances.txt does not exists')


def read_distances(fname):
    try:
        with open(fname, 'r') as csvfile:
            distances_dataset = []
            reader = csv.reader(csvfile, lineterminator='\n')
            for row in reader:
                np_row = np.asarray(row)
                distances_dataset.append(np_row)
            distances_dataset = np.asanyarray(distances_dataset)
            return distances_dataset
    except:
        raise OSError('path to distances.txt does not exists')
