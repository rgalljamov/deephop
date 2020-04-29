import numpy as np

# import Guro's Gym Environment, necessary to register the Guro-v0 environment
from thesis_galljamov18.python.guro_gym_env.gym_guro.mujoco import guro_env
from thesis_galljamov18.python.settings import PATH_THESIS_FOLDER

# import baselines
from thesis_galljamov18.python.baselines.baselines import run
from thesis_galljamov18.python.baselines.baselines.common import cmd_util


# --------------------------------
# SIMULATION SETTINGS
# --------------------------------

SIMULATION_TIMESTEP = 1/2000

RENDER_SIMULATION = True
RENDER_X_STEPS = 6000

# every Y steps during training, the training is rendered for X steps to observe the so far learned behavior
# during the rendered steps, relevant data is collected and plotted afterwards
CHECK_TRAINING_PROGRESS_EVERY_Y_STEPS = 1e6

# pause sim after startup to be able to change rendering speed or camera perspective
PAUSE_VIEWER_AT_BEGINNING = True and RENDER_SIMULATION

PLOT_DATA = True

# should be False during training, and True when a model was loaded.
# Is automatically set to True below, when a model was loaded
PLOT_ALSO_FLIGHT_PHASE_DATA = False


# --------------------------------
# SAVE AND LOAD AGENT SETTINGS
# --------------------------------

# Choose whether, where and how ofter you want to save your model during training.
SAVE_MODEL = True
save_filename = "test_steps{}_rew{}_eplen{}"
SAVE_PATH = PATH_THESIS_FOLDER +"python/training/ppo2_models/" + save_filename

# NOTICE: saving interval should always be a power of 2
SAVING_INTERVAL = np.power(2, 18) # 2^18 is about 250k steps

# Load a saved model to test it in simulation
# if set to True, the guro environment will be automatically loaded in DEMO mode
LOAD_MODEL = True
load_from_folder = PATH_THESIS_FOLDER +"python/training/ppo2_models/"
load_filename = "knHpET_wP5xwV_flKnee_10p20v_eplen800_64r_15208_splRew600_16LO_3M_LR3f5_s5767168_r1211_l375"
LOAD_PATH = load_from_folder + load_filename

if LOAD_MODEL:
    PLOT_ALSO_FLIGHT_PHASE_DATA = True


# --------------------------------
# TRAINING SETUP
# --------------------------------

ENVIRONMENT = "Guro-v0"
TRAIN_FOR_TIMESTEPS = 8e6
LEARNING_RATE = lambda f: 3e-5 * f**7 if (3e-5 * f**7) > 1e-7 else 1e-7

FLIGHT_PHASE_HIP_ANGLE = 17
FLIGHT_PHASE_KNEE_ANGLE = 142


def setupAndStartTraining():
    arg_parser = cmd_util.common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args()

    args.env = ENVIRONMENT
    args.alg = 'ppo2'
    args.num_timesteps = 0 if LOAD_MODEL else TRAIN_FOR_TIMESTEPS
    args.network = 'mlp'
    args.play = LOAD_MODEL

    run.main(args)
    return True


# set all settings and start the training
if __name__ == "__main__":
    setupAndStartTraining()

