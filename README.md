# Neuro-Trajectory-Proj

Code here is based on and further developed from:
https://github.com/RovisLab/NeuroTrajectory


## Installation

The following instructions will help to run and update code

### Prerequisites

The packages needed for install can be found inside requirements.yml.
To avoid potential conflicts it is best to install the environment using load_conda.

```
conda env create -f environment.yml
```

To activate and deactivate GPU environement run the following commands

```
conda activate nnt-gpu
conda deactivate
```

If running on ICHEC don't activate the environment as this is handled in sbatch file, however the environment still needs to be set up and newest version of CUDA installed.
Insure you are in the nnt-gpu environment before running the following commands.

```
conda install -c conda-forge cudatoolkit-dev
conda install -c conda-forge cudnn=8.2.1
```

The cudatoolkit is quite large ~20gb so this may be an issue as the ICHEC home directory limit is 25gb

### Running the code

The script which runs the Neuroevolutionary code is

```
python main.py
```

This has been tested on Windows. 

Some properties are hardcoded that are not yet in the config. The most important of these are

load_data.py:

```
self.slide = True
self.shuffle = True
self.absolute_path_cond = True
self.large_data = True #(Warning: Only works on Windows, set to false if using Mac or Linux)
```
genetic_algorithm.py

```
self.population = 6
self.generations = 3
```

If collecting new data an updated version of GridSim has been included that records highway data.
First navigate to directory to run code

```
cd gridSim/GridSim_Scenarios
```

GridSim has a separate list of dependencies. If installing these on nnt-gpu it is best to try installing through conda to avoid conflicts. If not use pip.

```
cat requirements.txt
```

To run a simulation:

```
python car_kinematic_highway.py
```

This populates the folder and updated teh state_buf.txt. The state_buf.txt contains the positional output and needs to nbe moved to ./images/ folder (at the top directory) when training models.

### Training a model

Note: Though there is code for DGN, LSTM and Conv3D only the LSTM code has been updated. It has been hardcoded to run in main.py

data_types.py contains the configuration parameters. num_classes should match the number of coordinates:

'num_classes': 8,
'classes_name': ['x2', 'y2', 'x3', 'y3', 'x4', 'y4', 'x5', 'y5']

If loading new data remove .npy files from data_sets/lstm_sliding


At the end of each training, plots and .csv files are generated in ./train/(date_time)/

## Built with

* [SciKit Learn][https://scikit-learn.org/stable/] - Machine Learning with Python.
* [Tensorflow](https://www.tensorflow.org/) - An open source machine learning framework for everyone.
* [Numpy](http://www.numpy.org/) - NumPy is the fundamental package for scientific computing with Python.
