### Project Description


This project implements a model-free learning based controller 
to generate stable human-like hopping in a realistic simulation of an existing robotic leg: the GURO Hopper,
developed by Dr. Guoping Zhao in the Locomotion Laboratory of the Technical University of Darmstadt. 


The code was written within the frame of a Bachelor Thesis 
with the title “Generating human-like hopping with Deep Reinforcement Learning”,
submitted by Rustam Galljamov on 8th November 2018. 
**A PDF version of the thesis** with a detailed description can be found in the media folder. 

Here the most important

* The robot is simulated using the **Python Wrapper** of the **MuJoCo** physics engine (mujoco_py, version 1.50.1.56).  
* The simulation is included into a **custom environment for the OpenAI gym** (version 0.10.5). 
* We use the **PPO** algorithm from the OpenAI Baseline repository (version is included in the package). 
* The code is written in Python 3.6 and tested on an 64-bit Ubuntu 16.04 installation. 



### Installation Instructions


It is recommended to install the required packages (see *requirements.txt*) in the order they’re presented in this and the previous section. Most of the packages are very easy to install with just one or two console lines. For installing more sophisticated packages, links to instructions are given in this section. 


To install tensorflow follow the steps in the corresponding documentation: 
https://www.tensorflow.org/install/
 
The installation steps for mujoco_py are described here:
https://github.com/openai/mujoco-py


Installation instructions for the OpenAI Gym (gym) can be found under the following link:
https://gym.openai.com/docs/#installation


PLEASE NOTE:
The given code was not tested with other baseline versions then the one provided within the package (modification of the version from 10.09.2018). It is expected, that adjustments will be necessary in order to run the code with a newer baselines version. More information about the baseline repository is to be found in the corresponding documentation:
https://github.com/openai/baselines


### First Steps

After all requirements are installed, two more steps are required to be able to run the code:
* add the path to the main folder itself included, like '/path/to/thesis_galljamov18/', to your system’s PYTHONPATH and set the PATH_THESIS_FOLDER variable in python/settings.py.
* add the path to the baselines-folder to your PYTHONPATH




### Test Installation - First Run

After the installation is completed and first steps are executed, run python/training/guro_train.py. It should load an already trained model and render a simulation for 6000 steps (about 12 seconds), where the robotic leg is performing stable hopping in place.


________________


### Project Structure


In this section, folders, subfolders and files are shortly described to know their purpose in the project. A more detailed description can be found in the files themselves. 


The main folder contains several subfolders and files: 
* **mujoco/**: containing MuJoCo models of the robot and a model for parameter identification as well as the corresponding STL files
* **python/**: containing the learning algorithm, tools, identification experiments etc.
* **media/**: containing some videos of the simulated robot in action and the thesis as PDF
* **requirements.txt**, listing all the required packages in a format accepted by most python IDE’s.
 
python/ subfolders and files:
* **baselines/**: modification of the official baseline repository in its state from 10.09.2018
* **guro_gym_env**: The OpenAI Gym Environment for the GURO Hopper robot
* **identification/**: data from the identification experiment and two scripts to find simulation parameters most closely replicating the observed experiment.
* **muscle_model**: Package containing a muscle model implementation by Guoping Zhao (2018) and some usage examples
* **training/**: the hearth of the package, described in more detail in the next section
* **settings.py**: module containing project wide parameters like common paths and plot settings.
* **tools.py**: a collection of useful functions, especially to process and plot data


python/training folder:
* **human_hopping_data/**: data from a human hopping experiment and a script to extract reference trajectories for the agent’s training
* **ppo2_models/**: saved agent models that can be loaded in guro_train. The names reflect the training settings/parameters used to train this model. New models during training are by default also saved in this folder. The name can be set in guro_train.py.
* **sim_data/**: when an agent model is loaded, data can be collected for investigations. In this case it will be saved in this folder as an .npy-file. The names of the saved data files refers to the model, used to collect the data. A ‘P’ or ‘Prtbt’ at a name’s beginning shows that the file only contains perturbed data.
* **guro_env_test.py**: a script that loads the guro gym environment, sends motor commands and collects data of interest. Can be used to implement a non-learning based controller, test reference trajectories, tune PD Position Controllers etc.
* **guro_train.py**: The starting point! This script allows to specify most of the training settings (beside such in python/guro_gym_env/gym_guro/mujoco/guro_env.py), start training or load an already trained model and observe its behavior.
* **plot_results.py**: Loads simulation data from sim_data/ and plots the mean leg length over 100 hops and the force length relation


#### GURO Environment


The most important module is the guro_env.py. It loads the MuJoCo simulation and turns it into a OpenAI Gym environment, meeting all requirements of a full reinforcement learning scenario. It is located at: python/guro_gym_env/gym_guro/mujoco/.


This module allows setting additional parameters for training and model demonstration, listed as constants at the beginning of the script. It receives the agent’s outputs, applies them in the environment and determines the new state as well as a reward for taking this action in the given state. Here also Early Termination and Reference State Initialization and State Normalization are implemented. Finally, the script collects relevant data and combines it to meaningful plots to monitor the training progress. 


