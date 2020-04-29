import os, time, coloredlogs, logging, traceback
import numpy as np
from gym import utils, spaces
from gym.envs.mujoco import mujoco_env

from thesis_galljamov18.python.muscle_model import humanmuscle
from thesis_galljamov18.python import tools, settings
from thesis_galljamov18.python.training import guro_train as config


# -----------------------------------
# SIMULATION SETTINGS
# -----------------------------------

SIM_TIMESTEP = config.SIMULATION_TIMESTEP

DO_RENDER = config.RENDER_SIMULATION
PLOT_DATA = config.PLOT_DATA
# when only stance phase is trained, it sometimes make sence to only plot stance phase data
PLOT_ALSO_FLIGHT_DATA = config.PLOT_ALSO_FLIGHT_PHASE_DATA

# to enable plots required for the thesis
THESIS = settings.THESIS_PLOTS


# -----------------------------------
# TESTs SETTINGS
# -----------------------------------

# set the robot's joints to the values from the reference trajectories at every simulation step
# PD_POSITION_CONTROL should be False then
JUST_APPLY_KIN_DATA = False
# or let the robot follow the hip and joint angles from the reference trajectories with PD controllers
# JUST_APPLY_KIN_DATA has to also be True in this case
PD_POSITION_CONTROL = True

# True if you want the sledge position to stay in place (e.g. to test Position control)
HOLD_SLEDGE_IN_PLACE = False


# -----------------------------------
# TRAINING SETTINGS
# -----------------------------------

# use PD position control in flight phase and let the agent control the motors in stance phase only
TRAIN_ONLY_STANCE = True

# True: agent only controls the knee motor, hip motor torque is 0 during stance phase
# False: agent is controlling both motors during the stance phase
ONLY_ONE_MOTOR = False

# if True, a uni-articular knee extensor muscle is simulated
# the agent in this case outputs the activation signal for this muscle
# FULLY IMPLEMENTED BUT NO STABLE HOPPING YET
MUSCLES_ACTIVATED = False

# reduce the controller frequency by FRAME_SKIP compared to SIM_TIMESTEP
FRAME_SKIP = 1

# how many timesteps can the simulated stance phase be longer or shorter then the one from the ref hop
ALLOWED_STANCE_DURATION_DEVIATION_TIMESTEPS = 20

# set to true to run the named experiment (dropping height is specified in oneTimeInitialization())
GROUND_DROP_EXPERIMENT = False
drop_has_happend = False
# to calculate the leg stiffness on perturbation
COLLECT_ONLY_PERTURBED_HOP_DATA = False

# deactivate all actions not necessarily required to run a model demo or other tests
DEMO = False or GROUND_DROP_EXPERIMENT or config.LOAD_MODEL
# collect and save data for 100 hops
HOP_100_TIMES = (False and DEMO) or COLLECT_ONLY_PERTURBED_HOP_DATA


# -----------------------------------
# FINITE STATE CONDITIONS
# -----------------------------------

# lowest allowed vertical sledge position
FINITE_SLEDGE_HEIGHT_MM = 400
FINITE_HIP_ANGLE = -5
FINITE_FOOT_POS_DEVIATION_CM = 3
FINITE_SLEDGE_POS_DEVIATION_CM = 0.4
FINITE_SLEDGE_VEL_DEVIATION_CMperS = 15
FINITE_HIP_ANG_DEVIATION = 7
FINITE_KNEE_ANG_DEVIATION = 8
FINITE_HIP_ANG_VEL_DEVIATION = 3
FINITE_KNEE_ANG_VEL_DEVIATION = 4


# -----------------------------------
# CONSTANTS
# -----------------------------------

# AXIS indices
AXIS_SLEDGE = 0
AXIS_HIP = 1
AXIS_KNEE_PULLEY_ON_THE_HIP = 2
AXIS_KNEE = 3

# MOTOR indices
MOT_HIP = 0
MOT_KNEE = 1
MOT_VIRTUAL_KNEE = 2

# MUSCLE indices
MUS_VAS = 0

# Hopping PHASES
PHASE_COMPRESS = 0
PHASE_MIDSTANCE = 1
PHASE_EXTEND = 2
PHASE_FLIGHT = 3


# -----------------------------------
# HELPER FUNCTIONS
# -----------------------------------

def rad(degrees):
    return np.multiply(degrees,np.pi)/180

def deg(angles_in_rad):
    return np.divide(angles_in_rad,np.pi)*180

def say(text):
    """Used to inform the user during training, that the agent is ready to demonstrate it's current state"""
    os.system('spd-say "{}" --volume -1 --voice-type male2'.format(text))


# -----------------------------------
# GURO ENVIRONMENT
# -----------------------------------

class GuroEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    # -----------------------------------
    # ENVIRONMENT'S STATE INDICES
    # -----------------------------------

    # currently used states
    NUM_STATES = 6
    STATE_KNEE_MOT_ANG = 0
    STATE_HIP_ANG = 1
    STATE_KNEE_MOT_ANG_VEL = 2
    STATE_HIP_ANG_VEL = 3
    STATE_SLEDGE_VEL = 4
    STATE_SLEDGE_POS = 5

    # states possibly useful in future
    STATE_DES_FREQ = 50
    STATE_MOT_TOR_HIP = 60
    STATE_MOT_TOR_KNEE = 6
    STATE_VIRTUAL_MOT_TOR_KNEE_AXIS = 70
    STATE_GRFS = 6

    # for muscle activation outputting agent
    STATE_VAS_LEN = 50
    STATE_VAS_VEL = 51
    STATE_VAS_FORCE = 52
    STATE_BFSH_LEN = 55
    STATE_BFSH_VEL = 56
    STATE_BFSH_FORCE = 57


    # -----------------------------------
    # FLAGS AND OTHER GLOBAL VARS
    # -----------------------------------

    motor_input = np.zeros(3)

    # training step count
    step_count = 0

    # considering only the one current episode... get's resetted with the environment
    episode_step_count = 0

    # last initial position on desired trajectory
    rsi_position = 0

    # useful to slow down simulation before it starts or change point of view
    pauseViewerAtFirstStep = config.PAUSE_VIEWER_AT_BEGINNING

    # init joint angles
    initKneeAngle = None
    initHipAngle = rad(15)

    # init knee motor pulley vel
    initHipPulleyVel = None


    # -----------------------------------
    # TRAINING PROGRESS MONITORING
    # -----------------------------------

    all_rewards = []
    episode_rewards = []
    batch_mean_rewards = []
    episode_lengths = []
    batch_mean_episode_lengths = []

    # count how many states were initialized in flight phase
    countFlightInits = 0
    flight_inits_rsi_timesteps = []

    # to determine at which position of the trajectory the most episodes end
    episode_ends_timesteps = []
    # to see how often initialization appeared at a certain state
    episode_init_timesteps = []


    # -----------------------------------
    # COLLECTIONS
    # -----------------------------------

    # kinematics
    des_sledge_poss = []
    des_knee_angs = []
    des_hip_angs = []
    is_hip_pulley_angs = []

    is_rope_elongs = []
    is_foot_poss = []

    des_sledge_vels = []
    des_knee_ang_vels = []
    des_hip_ang_vels = []
    des_knee_accs = []
    des_hip_accs = []

    is_sledge_poss = []
    is_knee_angs = []
    is_hip_angs = []
    is_sledge_normed_poss = []
    is_knee_normed_angs = []
    is_hip_normed_angs = []
    is_knee_vel_normed = []
    is_hip_vel_normed = []
    is_grfs_normed = []
    is_sledge_run_mean = []
    is_sledge_run_std = []
    is_sledge_vels = []
    is_knee_ang_vels = []
    is_knee_agn_vels_fltrd = []
    is_hip_pulley_ang_vels = []
    is_hip_pulley_ang_vels_fltrd = []
    is_hip_ang_vels = []
    is_knee_accs = []
    is_hip_accs = []
    phases = []

    # kinetics
    grfs = []
    grfs_filtered = []
    agent_outputs_knee = []
    agent_outputs_hip = []
    knee_motor_torque_agent_outputs = []
    hip_motor_torque_agent_outputs = []
    knee_motor_torque_from_muscle_calculated_trajec = []
    knee_motor_des_tor_with_rope_transmission = []
    knee_motor_torque_measured = []
    hip_motor_torque_measured = []
    knee_axis_torque_des_trajec = []
    knee_axis_torque_actual_trajec = []
    rope_forces = []


    #####################################################################################
    #                                                                                   #
    #                         ENVIRONMENT INITIALIZATION                                #
    #                                                                                   #
    #####################################################################################

    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, settings.PATH_THESIS_FOLDER + 'mujoco/guro_mujoco.xml', 1)
        utils.EzPickle.__init__(self)

        self.model.opt.timestep = SIM_TIMESTEP

        # setup logger
        self.log = logging.getLogger(__name__)
        coloredlogs.install(level='DEBUG')

        # small usage simplification
        self.data = self.sim.data

        # set action space
        low = np.zeros(1 if (MUSCLES_ACTIVATED or ONLY_ONE_MOTOR) else 2)
        high = np.ones(1 if (MUSCLES_ACTIVATED or ONLY_ONE_MOTOR)  else 2)
        self.action_space = spaces.Box(low, high)

        # set observation space
        high = np.ones(self.NUM_STATES)
        low = -high
        self.observation_space = spaces.Box(low, high)

        # initialize simulation environment
        self.oneTimeInitializations()
        self.initParamsAndContainers()
        self.setInitialConditions()
        self._initMuscles()

        # disable ET together with some other features
        if DEMO:
            self.demonstration_on = True

        # current training settings (will be printed in the console on every demonstration...)
        self.settings = ("MUS - " if MUSCLES_ACTIVATED else ("1 MOT - " if ONLY_ONE_MOTOR else "2" + " MOT - ")) + \
                        "3ET + LOpun + ET all conditions + softKnHp - upTo25ET - rlSens + FS1\n" \
                        "softenET 2M -> 2*ET -> 3ET + {} LO steps + ET ({}, {}, 2M) + {}\n" \
                        "MaxEpLen20 - wP2wV1 + LR: 3e-5 * f**7 clipped to 1e-7".format(
                            ALLOWED_STANCE_DURATION_DEVIATION_TIMESTEPS, FINITE_SLEDGE_POS_DEVIATION_CM,
                            FINITE_SLEDGE_VEL_DEVIATION_CMperS, FINITE_FOOT_POS_DEVIATION_CM)

        # ... and at the beginning of a training
        self.log.info("---\nSETTINGS: {}!\n---\n".format(self.settings))


    def oneTimeInitializations(self):
        """ Loads ref trajecs and initializes several variables, that are not episode dependent."""

        # count flight phases for ground drop experiment
        self.flightPhaseCount = 0

        self.loadReferenceTrajectories()
        self.initOrClearDebuggingCollections()

        # count how many steps have been recorded for each phase
        self.phasesTimesteps = np.zeros(4)

        # save detected phases of last 10ms to avoid detecting flight phase during compression
        self.lastPhase = PHASE_FLIGHT

        # avoid determening the contact position several times when unchanged
        self.lastTimestepGetContactPositionWasCalled = -1

        # True when current models performance is rendered (disables several checks and logs)
        self.demonstration_on = DEMO

        # if hopping data should be collected
        if HOP_100_TIMES:
            self.nHopsData = {"grfs":[], "legLen":[], "legStif":[], "phases":[]}
            self.numOfSavedHops = 0

        # initialize class to compute running mean and variance for normalization
        self.run_stats = []
        for i in range(self.NUM_STATES):
            self.run_stats.append(tools.getRunningStats())


    def loadReferenceTrajectories(self):

        from thesis_galljamov18.python.training.human_hopping_data import process_human_hopping_data as humanData

        # also load only perturbed data, if ground drop experiment is performed
        if GROUND_DROP_EXPERIMENT:
            # set the ground plate position to the desired ground dropping height
            self.groundDropMeters = np.around(0.075 * 0.5366 / 1.0945287, 5)
            print("ground drop in meters is {}".format(self.groundDropMeters))
            self.drop_has_happend = False
            self.model.body_pos[6][2] = self.groundDropMeters - 0.002  # position is related to the plates COM

            self.allPerturbedHopsRefData, self.refPerturbedTouchdownConditions = \
                humanData.getReferenceTrajecsForRobot(groundDropTest=GROUND_DROP_EXPERIMENT,
                                                      perturbationHeight=self.groundDropMeters,
                                                      onlyPerturbedHops=True)
            self.totalNrOfPerturbedHops = np.size(self.allPerturbedHopsRefData[0])

        else:
            # set the ground plate position being below the ground
            self.model.body_pos[6][2] = -0.1

        self.allHopsRefData, self.refTouchdownConditions = \
            humanData.getReferenceTrajecsForRobot(groundDropTest=GROUND_DROP_EXPERIMENT,
                                                  perturbationHeight=self.groundDropMeters if GROUND_DROP_EXPERIMENT else 0)

        self.totalNrOfHops = np.size(self.allHopsRefData[0])


    def initParamsAndContainers(self):
        """
        Initialization of episode dependent variables, that has to be performed on each episode initialization.
        """

        # array saving previous state for comparison and reward calculation
        self.previousState = np.zeros(self.NUM_STATES)

        self.timestepsInFlight = 0
        # avoid determening the phase several times when unchanged
        self.lastTimestepGetPhaseWasCalled = -1

        # randomly choose one hop out of all in the reference data: kinematic trajecs to follow
        self.des_kin_trajecs = self.getOneRandomHopTrajecs()

        # determine the position on the desired trajectory to initialize the new episode in
        self.applyPhaseRSI()

        # counts the steps in current episode
        self.episode_step_count = 0

        # needed when more then one hop was achieved during training
        # in this case the episode_step_count will be bigger than the stance phase duration of the next hop
        # and we have to substract the length of the last stance phase before LO from the episode timestep when finding our pos on the new ref hop
        self.timestepsInPreviousHopsOfCurrentEpisode = 0

        # timesteps the LO is too late
        self.lateLOStepCount = 0

        self.rewards = []

        # pause the simulation and rendering when a new episode was initiated to test init states
        testInitialStates = False
        if testInitialStates:
            self._get_viewer()._paused = True


    def applyPhaseRSI(self):
        """
        To avoid overfitting, by having too much observations from a certain phase compared to the others,
        we divide the stance phase in three sub phases: compression, mid-stance, extension.

        The state observed in each of these phases are counted.
        On every episode initialization, we randomly choose a position on the desired trajectory
        from the phase with the least amount of observations so far.

        As the TD and TO are especially hard to learn, states from these areas of the ref trajectory are chosen
        with a higher probability. This higher probability is linearly decreasing during training time
        until all states in each phase are randomly chosen with the same probability.
        """

        # sample from phase with the minimum observations so far, with a favor for TD and TO area
        ADVANCED_STATE_RSI = True and not JUST_APPLY_KIN_DATA

        if ADVANCED_STATE_RSI and not DEMO:
            # choose state from the phase with the least amount of data sampled from so far
            phaseToSampleRsiStateFrom = np.argmin(self.phasesTimesteps[[PHASE_COMPRESS, PHASE_MIDSTANCE, PHASE_EXTEND]])

            # sample much more states in compression phase at the beginning and decrease it with time
            sampleFromMidstance = phaseToSampleRsiStateFrom == PHASE_MIDSTANCE

            if sampleFromMidstance:
                self.rsi_timestep = np.random.randint(self.refHopStancePhaseDuration * 0.375,
                                                      self.refHopStancePhaseDuration * 0.625)
            else:
                sampleFromCompression = phaseToSampleRsiStateFrom == PHASE_COMPRESS
                sampleTakeOffTimesteps = False

                if sampleFromCompression:
                    sampleFromTouchdownTimesteps = np.random.rand() < np.clip((0.5 - self.step_count / 2e6), 0.1, 1)
                else:
                    sampleTakeOffTimesteps = np.random.rand() < np.clip((0.5 - self.step_count / 2e6), 1 / 10, 1)

                self.rsi_timestep = np.random.randint(0,
                                                      self.refHopStancePhaseDuration - 1) if not JUST_APPLY_KIN_DATA else 0

                if sampleFromCompression:
                    self.rsi_timestep = np.clip(
                        int(self.rsi_timestep * (0.35 if not sampleFromTouchdownTimesteps else 0.1)),
                        0, self.refHopStancePhaseDuration / 2)
                elif sampleTakeOffTimesteps:
                    self.rsi_timestep = \
                        np.random.randint(self.refHopStancePhaseDuration * 9 / 10, self.refHopStancePhaseDuration - 1) \
                            if not JUST_APPLY_KIN_DATA else 0

            # reduce all counts by minimum timesteps count
            minTimesteps = self.phasesTimesteps[phaseToSampleRsiStateFrom]
            self.phasesTimesteps -= minTimesteps

        elif not DEMO:

            # sample much more states in compression phase at the beginning and decrease it with time
            sampleFromCompression = np.random.rand() < np.clip((0.95 - self.step_count / 3e6), 0.7, 1)
            if sampleFromCompression:
                sampleFromTouchdownTimesteps = np.random.rand() < np.clip((0.75 - self.step_count / 2e6), 0.1, 1)

            self.rsi_timestep = np.random.randint(0,
                                                  self.refHopStancePhaseDuration - 1) if not JUST_APPLY_KIN_DATA else 0

            if sampleFromCompression:
                self.rsi_timestep = np.clip(
                    int(self.rsi_timestep * (0.45 if not sampleFromTouchdownTimesteps else 0.1)),
                    0, self.refHopStancePhaseDuration - 1)

        self.rsi_timestep = int(self.rsi_timestep) if not DEMO and not JUST_APPLY_KIN_DATA else 0
        self.episode_init_timesteps.append(self.rsi_timestep)


    def getOneRandomHopTrajecs(self):
        """randomly choose one hop out the reference data and sort it to reuse previous code"""

        randHopNr = np.random.randint(0, self.totalNrOfHops)
        self.chosenHopNr = randHopNr
        refTrajecsOneHop = [trajec[randHopNr] for trajec in self.allHopsRefData]
        self.refHopStancePhaseDuration = np.size(refTrajecsOneHop[0])
        self.episode_step_count = 0

        # -------------------------------------------------------
        # sort the data for it to be usable with the old code
        # -------------------------------------------------------

        # previous ref trajecs order
        sledge_positions, sledge_vels, knee_angles, hip_angles, knee_ang_vels, hip_ang_vels, grfs, phases, \
        sledge_accs, knee_ang_accs, hip_ang_accs = range(11)

        # new ref trajecs order
        iSledgePos, iSledgeVel, iHipAngle, iHipAngVel, iKneeAngle, iKneeAngVel = range(6)

        sortedRefTrajecsOneHop = np.zeros([self.refHopStancePhaseDuration, 6])
        sortedRefTrajecsOneHop[:, sledge_positions] = refTrajecsOneHop[iSledgePos]
        sortedRefTrajecsOneHop[:, sledge_vels] = refTrajecsOneHop[iSledgeVel]
        sortedRefTrajecsOneHop[:, knee_angles] = refTrajecsOneHop[iKneeAngle]
        sortedRefTrajecsOneHop[:, hip_angles] = refTrajecsOneHop[iHipAngle]
        sortedRefTrajecsOneHop[:, knee_ang_vels] = refTrajecsOneHop[iKneeAngVel]
        sortedRefTrajecsOneHop[:, hip_ang_vels] = refTrajecsOneHop[iHipAngVel]

        return sortedRefTrajecsOneHop


    def _initMuscles(self):
        """ ATTENTION: should always be called after :func: setInitialConditions """
        humanmuscle.timestep = SIM_TIMESTEP
        knee_angle = (self.getKneeAng(), )
        self.musVAS = humanmuscle.KneeExtensorMuscle(knee_angle)
        self.musVasActBuffer = np.ones(7)*0.001


    def setInitialConditions(self):
        """
        Sets all joint positions and velocities
        to the values derived from the reference trajecs at the rsi timestep.
        """

        sledge_positions, sledge_vels, knee_angles, hip_angles, knee_ang_vels, hip_ang_vels, grfs, \
        sledge_accs, knee_ang_accs, hip_ang_accs = range(10)

        initKinematics  = self.des_kin_trajecs[int(self.rsi_timestep),:]

        # sledge
        self.data.qpos[AXIS_SLEDGE] = initKinematics[sledge_positions]
        self.data.qvel[AXIS_SLEDGE] = initKinematics[sledge_vels]

        # hip joint
        self.data.qpos[AXIS_HIP] = initKinematics[hip_angles]
        self.data.qvel[AXIS_HIP] = initKinematics[hip_ang_vels]
        self.initHipAngle = initKinematics[hip_angles]

        # knee joint
        self.data.qpos[AXIS_KNEE] = initKinematics[knee_angles]
        self.data.qvel[AXIS_KNEE] = initKinematics[knee_ang_vels]
        # reset the init Knee angle to get a knee angle with much better resolution on first step (for rope transmission)
        self.initKneeAngle = None
        # it is then set in simulateRopeTransmission with high precision: self.initKneeAngle = initKinematics[knee_angles]

        # hip pulley
        self.data.qpos[AXIS_KNEE_PULLEY_ON_THE_HIP] = 0
        self.data.qvel[AXIS_KNEE_PULLEY_ON_THE_HIP] = 0


    def getRefHopBasedOnTDConditions(self, simTdSledgePos, simTdSledgeVel):
        """
        When a TO was reached during a training episode, we need to choose a new reference hop after the TD.
        This is done by comparing the TD conditions (sledge pos and vel) in simulation and all reference hops.
        The reference hop with the most similar TD conditions to those in sim is then chosen.
        """
        # new ref trajecs order
        iSledgePos, iSledgeVel, iHipAngle, iHipAngVel, iKneeAngle, iKneeAngVel = range(6)

        # touchdownConditions shape is ([6, totalNrOfHops])
        tdConditionDeviations = []

        # if the ground drop experiment is performed, the ref hop after the ground drop has to be chosen
        # from a different pool of hops, only containing perturbed hops with the right ground drop height
        if GROUND_DROP_EXPERIMENT and self.drop_has_happend and self.flightPhaseCount == 3:
            for hopIndex in range(self.totalNrOfPerturbedHops):
                hop = self.refPerturbedTouchdownConditions[0:2,hopIndex]
                # sledge pos has a 10 times smaller range then vel, but vels are more important
                tdConditionDeviations.append(abs((hop[iSledgePos] - simTdSledgePos)*10 + 2*abs((hop[iSledgeVel] - simTdSledgeVel))))

            hopNr = tdConditionDeviations.index(min(tdConditionDeviations))
            self.chosenHopNr = hopNr
            refTrajecsOneHop = [trajec[hopNr] for trajec in self.allPerturbedHopsRefData]
            self.refHopStancePhaseDuration = np.size(refTrajecsOneHop[0])
            self.episode_step_count = 0
            tools.log("successfully sampled one of the perturbed hops!")

            # reduce all ref sledge positions back to normal height as they are sampled from the unperturbed data
            for hopIndex in range(len(self.allHopsRefData[0])):
                self.allHopsRefData[0][hopIndex] -= self.groundDropMeters
        else:
            if not COLLECT_ONLY_PERTURBED_HOP_DATA: self.saveHopDataToDict()

            for hopIndex in range(self.totalNrOfHops):
                hop = self.refTouchdownConditions[0:2,hopIndex]
                # sledge pos has a 10 times smaller range then vel, but vels are more important
                tdConditionDeviations.append(abs((hop[iSledgePos] - simTdSledgePos)*10 + 2*abs((hop[iSledgeVel] - simTdSledgeVel))))

            hopNr = tdConditionDeviations.index(min(tdConditionDeviations))
            self.chosenHopNr = hopNr
            refTrajecsOneHop = [trajec[hopNr] for trajec in self.allHopsRefData]
            self.refHopStancePhaseDuration = np.size(refTrajecsOneHop[0])
            self.episode_step_count = 0

        # -------------------------------------------------------
        # sort the data for it to be usable with the previous implementations
        # -------------------------------------------------------

        # previous ref trajecs order
        sledge_positions, sledge_vels, knee_angles, hip_angles, knee_ang_vels, hip_ang_vels, grfs, phases, \
        sledge_accs, knee_ang_accs, hip_ang_accs = range(11)

        sortedRefTrajecsOneHop = np.zeros([self.refHopStancePhaseDuration, 6])
        sortedRefTrajecsOneHop[:, sledge_positions] = refTrajecsOneHop[iSledgePos]
        sortedRefTrajecsOneHop[:, sledge_vels] = refTrajecsOneHop[iSledgeVel]
        sortedRefTrajecsOneHop[:, knee_angles] = refTrajecsOneHop[iKneeAngle]
        sortedRefTrajecsOneHop[:, hip_angles] = refTrajecsOneHop[iHipAngle]
        sortedRefTrajecsOneHop[:, knee_ang_vels] = refTrajecsOneHop[iKneeAngVel]
        sortedRefTrajecsOneHop[:, hip_ang_vels] = refTrajecsOneHop[iHipAngVel]

        # reset rsi_timestep as it was just interesting for the last hop
        self.rsi_timestep = 0

        return sortedRefTrajecsOneHop


    def reset_model(self):
        self.initParamsAndContainers()
        self.setInitialConditions()
        if MUSCLES_ACTIVATED: self._initMuscles()
        return self.determineNewState()


    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 1
        self.viewer.cam.elevation = -20
        self.viewer.cam.azimuth = 0


    def _get_obs(self):
        return self.determineNewState()


    #####################################################################################
    #                                                                                   #
    #                            REINFORCEMENT SCENARIO                                 #
    #                                                                                   #
    #####################################################################################

    # --------------------------------
    # PERFORM A SIMULATION STEP
    # --------------------------------

    def step(self, a):
        """
         Performs one simulation step by applying the agents output a to the environment and returning
         the next environment state and a reward signal.
         :param a: list containing desired motor torques like: [des_hip_motor_tor, des_knee_mot_tor]
         """

        # step function is called during environment initialization - ignore this step
        if np.array_equal(a,np.zeros(self.model.nu)):
            return np.zeros(self.NUM_STATES), 0, False, {}

        # check if the action and states are valid
        if not self.areActionsAndStateValid(a):
            return self.stopEpisode()

        self.step_count += 1

        # pause sim after startup to be able to change rendering speed or camera perspective
        if DO_RENDER and self.pauseViewerAtFirstStep:
            self._get_viewer()._paused = True
            self.pauseViewerAtFirstStep = False

        # check for LATE LO (liftoff) in simulation
        if self.episode_step_count >= self.refHopStancePhaseDuration-1:
            # count how many steps the agent is late for LO
            self.lateLOStepCount += 1
        else:
            self.lateLOStepCount = 0

        # fix sledge position (e.g. for flight phase position control tuning)
        if HOLD_SLEDGE_IN_PLACE:
            self.data.qpos[0] = 0.5356
            self.data.qvel[0] = 0
            self.data.qacc[0] = 0

        # own implementation of frameskip as otherwise we get problems with the rope mechanism
        for frame in range(FRAME_SKIP):

            self.episode_step_count += 1

            # simulate next step
            try:
                if JUST_APPLY_KIN_DATA:
                    self.do_simulation_by_applying_ref_trajecs()
                else:
                    motor_output = self.determineMotorCommandsFromAgentsOutputs(a, frame)
                    self.do_simulation(motor_output, 1)
            except Exception as ex:
                self.logException()
                pass

        # calculate mean reward for a fixed timesteps amount to better see improvements in learning
        # and reduce noise compared to tracking episodic mean reward
        if self.step_count % np.power(2,14) == 0:
            self.batch_mean_rewards.append(np.around(np.mean(self.all_rewards[-np.power(2, 14) - 1:]), 2))
            self.batch_mean_episode_lengths.append(np.around(np.mean(self.episode_lengths[-40:])))
            print(self.settings)

        modulo, steps_to_render = self.collectDebuggingDataAndRenderSim()

        # check for and handle flight phase and touchdown
        stop_episode = self.handleFlightPhaseAndTouchdown(modulo < steps_to_render)

        # stop episode if a terminal condition was detected
        if stop_episode:
            return self.stopEpisode()

        done, newState, reward = self.determineRLScenarioParams(modulo, steps_to_render)

        return newState,reward,done,{}


    def determineRLScenarioParams(self, modulo, steps_to_render):
        """
        Determines the next state, calculates the reward and checks, if a finite state was reached.
        """

        if not JUST_APPLY_KIN_DATA:
            newState = self.determineNewState()
            done = self.reachedTerminalState()
        else:
            done = False
            newState = self.previousState

        # if late LO was detected return only a small reward
        reward = self.calculateReward(self.episode_step_count) if self.lateLOStepCount == 0 else 1
        self.all_rewards.append(reward)
        self.rewards.append(reward)

        if done:
            episodeLen = self.episode_step_count + self.timestepsInPreviousHopsOfCurrentEpisode
            if not modulo < steps_to_render - 2:
                epRew = int(np.sum(self.all_rewards[-episodeLen:]))
                self.episode_rewards.append(tools.exponentialRunningSmoothing("eprew", epRew, 0.01))
            self.episode_lengths.append(episodeLen)
            self.episode_ends_timesteps.append(self.episode_step_count + self.rsi_timestep)

        return done, newState, reward

    # --------------------------------
    # GET STATE
    # --------------------------------

    def determineNewState(self):
        """
        Gets and returns the current normalized state of the environment.
        NOTE: As the state is already normalized by the environment,
        a normalization in the learning algorithm is not required.
        """

        # max allowed value after feature normalization
        clipRange = 5

        newState = np.clip(self.normalizeWithRefDataMeanAndVariance(),-clipRange, clipRange)
        newState = np.around(newState, 8)

        NORMALIZATION_ON = True
        if NORMALIZATION_ON and max(abs(newState)) > clipRange:
            self.log.error("New State had a value bigger {}:\n{}!".format(clipRange, newState))

        # count how many states were sampled in each of three phases
        self.phasesTimesteps[self.getPhase()] += 1 if self.getPhase() == PHASE_COMPRESS else 0.75
        # consider also midstance
        if self.refHopStancePhaseDuration * 0.4 >= self.episode_step_count <= self.refHopStancePhaseDuration * 0.6:
            self.phasesTimesteps[PHASE_MIDSTANCE] += 0.6

        self.previousState = newState
        return newState


    # --------------------------------
    # CALCULATE REWARD
    # --------------------------------

    def calculateReward(self, episode_timestep):
        return self.calculateImitationReward(episode_timestep)


    def calculateImitationReward(self, episode_timestep):
        """
        The reward consists of three different parts: trajectory reproduction, foot position and episode length.
        The reward signal is a weighted sum of these parts.
        """

        angle_dists, vel_dists, angles, vels, des_angles, des_vels = self.getKinematicsDistance(episode_timestep)

        # punish the agent taking off too early in simulation
        if self.lateLOStepCount > 0:
            self.lateLOStepCount -= 1
            return 1

        # -----------------------------------
        # TRAJECTORY REPRODUCTION (KINEMATIC REWARD)
        # -----------------------------------

        max_kinematic_reward = 15

        pos_difference_cm, vel_difference_cm_per_s = self.getSledgePosAndVelDeviations(episode_timestep)

        # the maximum allowed deviations for pos and or vel have been changed
        # rescale the new ranges to the initial range to reuse old reward function
        if FINITE_SLEDGE_POS_DEVIATION_CM < 5:
            pos_difference_cm = tools.linearScale(pos_difference_cm,
                                                  0, FINITE_SLEDGE_POS_DEVIATION_CM, 0, 5)
            vel_difference_cm_per_s = tools.linearScale(vel_difference_cm_per_s,
                                                        0, FINITE_SLEDGE_VEL_DEVIATION_CMperS, 0, 5)

        # position reward is between 0 and 5 for a max deviation of 5cm
        position_rew = -0.05624185 + 5.056242*np.exp(-0.8997435*pos_difference_cm)

        # velocity reward is between 0 and 5 for a max deviation of 20cm/s
        velocity_rew = -0.05624185 + 5.056242*np.exp(-0.8997435*vel_difference_cm_per_s)

        reward = 2.3*position_rew + 0.7*velocity_rew

        # -----------------------------------
        # EPISODE LENGTH REWARD
        # -----------------------------------

        max_eplen_rew = 20
        initRSI_Timestep = self.episode_init_timesteps[-1]
        eplen =  (initRSI_Timestep + episode_timestep+ self.timestepsInPreviousHopsOfCurrentEpisode) \
            if not JUST_APPLY_KIN_DATA else self.step_count

        # the reward itself is a linear function, clipped on a max values specified above
        eplen_rew = max_eplen_rew*eplen/600
        eplen_rew = np.clip(eplen_rew, 0, max_eplen_rew)
        reward += eplen_rew

        # -----------------------------------
        # FOOT POSITION REWARD
        # -----------------------------------

        max_foot_pos_rew = 8

        # check for GRFs, as otherwise foot position cannot be determined correctly
        if self.getGRF() > 0:
            # reward the foot position being close to z-axis
            foot_distance_cm = abs(np.around(abs(self.getGroundContactYPosition()) * 100, 2))
            # as ET allows only 2cm instead of 10cm deviation
            foot_distance_cm = tools.linearScale(foot_distance_cm, 0, FINITE_FOOT_POS_DEVIATION_CM / 2, 0, 10)
            # exponential function (5 when foot distance is 0, and 0 when scaled foot_distance reaches 10cm
            foot_pos_rew = -0.0634568 + 10.06346 * np.exp(-0.5066307 * foot_distance_cm)
            foot_pos_rew = tools.linearScale(foot_pos_rew, 0, 10, 0, max_foot_pos_rew)
            reward += foot_pos_rew

        # IDEA: not enough tested so far
        # if self.timestepsInPreviousHopsOfCurrentEpisode > 0 and self.episode_step_count > 20:
        #     # at least one successful takeoff was achieved and TD conditions were good
        #     reward *= 1.25 # increase the reward by 25% to show that perform several hops is good

        # scale reward to the range [0:10]
        rew_max_possible = max_kinematic_reward + max_foot_pos_rew + max_eplen_rew
        reward = tools.linearScale(reward, 0, rew_max_possible, 0, 10)

        return reward


    # --------------------------------
    # TERMINAL CONDITIONS
    # --------------------------------

    def reachedTerminalState(self):
        """
        Determines, wheter a terminal condition was met or not. In the first case, the episode will be terminated.

        Therefore, the sim trajectory is compared to the ref trajectory. A terminal condition is met,
        when the deviation between sim and ref trajectories exceed a certain limit.
        This limit is linearly increased with time to allow exploration.

        The episode is also terminated after a certain episode duration is reached.
        """

        # 1600 steps means at least three full hops which is enough to get stable hopping
        if self.episode_step_count >= 1600:
            return True

        # soften ET with time but first after 'startLinSoftAfterSteps' steps
        # soften linearly, which means increase the allowed deviation linearly with time.
        # clip the max allowed deviations to be maximum double compared to initial values
        startLinSoftAfterSteps = 0
        if self.step_count > startLinSoftAfterSteps:
            doubleAfterSteps = 2e6
            _FINITE_SLEDGE_POS_DEVIATION_CM = np.clip(
                (1+((self.step_count-startLinSoftAfterSteps)/doubleAfterSteps))*FINITE_SLEDGE_POS_DEVIATION_CM,
                0, 2*FINITE_SLEDGE_POS_DEVIATION_CM)

            _FINITE_SLEDGE_VEL_DEVIATION_CMperS = np.clip(
                (1+((self.step_count-startLinSoftAfterSteps)/doubleAfterSteps))*FINITE_SLEDGE_VEL_DEVIATION_CMperS,
                0, 2*FINITE_SLEDGE_VEL_DEVIATION_CMperS)
        else:
            _FINITE_SLEDGE_POS_DEVIATION_CM = FINITE_SLEDGE_POS_DEVIATION_CM
            _FINITE_SLEDGE_VEL_DEVIATION_CMperS = FINITE_SLEDGE_VEL_DEVIATION_CMperS

        if DEMO or JUST_APPLY_KIN_DATA:
            return False

        # if we are in a late LO there is no ref trajecs available for this timestep anymore
        # we give the agent 20 timesteps to reach flight phase, otherwise the episode will be terminated
        if self.lateLOStepCount > 0:
            return self.lateLOStepCount >= ALLOWED_STANCE_DURATION_DEVIATION_TIMESTEPS  # 32tsps = about 8% of the mean stance phase duration

        # foot should always stay close to the z-axis
        foot_too_far_away_from_guide = (self.getGRF() > 0 and
                                        abs(self.getGroundContactYPosition()*100) > (FINITE_FOOT_POS_DEVIATION_CM if not self.demonstration_on
                                                                                     else 2.5*FINITE_FOOT_POS_DEVIATION_CM))

        # check if the simulation trajectories have deviated too much from the ref trajecs
        angle_dists, vel_dists, angles, vels, des_angles, des_vels = self.getKinematicsDistance(self.episode_step_count)
        sledge_pos_difference_cm, sledge_vel_difference_cm_per_s = self.getSledgePosAndVelDeviations(self.episode_step_count)

        trajec_deviation_too_high = (sledge_pos_difference_cm > _FINITE_SLEDGE_POS_DEVIATION_CM \
                                    and sledge_vel_difference_cm_per_s > _FINITE_SLEDGE_VEL_DEVIATION_CMperS) \
                                    or (angle_dists[0] > FINITE_KNEE_ANG_DEVIATION and angle_dists[1] > FINITE_HIP_ANG_DEVIATION \
                                    and vel_dists[0] > FINITE_KNEE_ANG_DEVIATION and vel_dists[1] > FINITE_HIP_ANG_VEL_DEVIATION)

        # check for too low sledge position of too small hip angle
        sledge_too_low = self.getSledgePos() < FINITE_SLEDGE_HEIGHT_MM / 1000
        hip_angle_too_small = self.getHipAng() < rad(FINITE_HIP_ANGLE)

        # if the models performance is currently rendered, only stop if sledge is too low
        if self.demonstration_on:
            return sledge_too_low or hip_angle_too_small or foot_too_far_away_from_guide

        terminal_condition_reached = (trajec_deviation_too_high
                or (sledge_too_low)
                or (hip_angle_too_small)
                or foot_too_far_away_from_guide)

        if terminal_condition_reached:
            if sledge_too_low:
                print("Finite state reached because of sledge position being too low!")
            elif foot_too_far_away_from_guide:
                print("Finite state reached because foot was too far away from guide")

        return terminal_condition_reached and self.episode_step_count >= 40 * FRAME_SKIP


    #####################################################################################
    #                                                                                   #
    #                         GETTER and SETTER FUNCTIONS                               #
    #                                                                                   #
    #####################################################################################

    def getKneeAngAcc(self):
        return self.data.qacc[AXIS_KNEE]

    def getHipAngAcc(self):
        return self.data.qacc[AXIS_HIP]

    def getSledgeAcc(self):
        return self.data.qacc[AXIS_SLEDGE]

    def getKneeAngVel(self):
        vel = self.data.qvel[AXIS_KNEE]
        vel_filtered = tools.exponentialRunningSmoothing("knee_vel", vel, 0.75)
        self.data.qvel[AXIS_KNEE] = vel_filtered
        return vel_filtered

    def getHipAngVel(self):
        return np.around(self.data.qvel[AXIS_HIP], 2)

    def getHipPulleyAngVel(self):
        vel = self.data.qvel[AXIS_KNEE_PULLEY_ON_THE_HIP]
        vel_filtered = tools.exponentialRunningSmoothing("hip_pul_vel", vel, 0.75)
        self.data.qvel[AXIS_KNEE_PULLEY_ON_THE_HIP] = vel_filtered
        return vel_filtered

    def getKneeAng(self):
        return self.data.qpos[AXIS_KNEE]

    def getHipAng(self, inDegree=False):
        hip_ang = np.around(self.data.qpos[AXIS_HIP], 2)
        return deg(hip_ang) if inDegree else hip_ang

    def getHipPulleyAng(self):
        return self.data.qpos[AXIS_KNEE_PULLEY_ON_THE_HIP]

    def getSledgePos(self):
        sledgePos = np.around(self.data.qpos[0], 3)
        return sledgePos

    def getSledgeVel(self):
        return np.around(self.data.qvel[0], 3)

    def getGRF(self):
        """ :returns the vertical coordinate of the GRFs"""
        contactForces = self.data.cfrc_ext
        grf_z_coordinate = contactForces[5, 5]
        return np.around(grf_z_coordinate, 1)

    def getPhase(self, considerMidstance=False):
        """ Returns the hopping phase: compression, extension or flight."""

        # stance phase
        current_grfs = self.getGRF()
        if current_grfs > 0:
            phase = PHASE_EXTEND if self.getSledgeVel() >= 0 else PHASE_COMPRESS
            if considerMidstance:
                if self.refHopStancePhaseDuration * 0.35 >= self.episode_step_count <= self.refHopStancePhaseDuration * 0.65:
                    phase = PHASE_MIDSTANCE
            self.timestepsInFlight = 0
        # flight phase
        else:

            # getPhase is called several times during one timestep: increase timestepsInFlight only once per timestep
            if self.lastTimestepGetPhaseWasCalled != self.episode_step_count:
                self.timestepsInFlight += 1
                self.lastTimestepGetPhaseWasCalled = self.episode_step_count

            # wait longer if the flight phase was entered in compression mode
            flightDetectionDelay = (80 if self.lastPhase == PHASE_COMPRESS else 10) / 1000  # 25ms in s

            if self.timestepsInFlight * SIM_TIMESTEP >= flightDetectionDelay:
                phase = PHASE_FLIGHT
                # print("Guro is flying  {} steps in a row!".format(self.timestepsInFlight))
            else:
                phase = self.lastPhase

        self.lastPhase = phase

        return phase

    def getGroundContactYPosition(self):
        """ should be only called when GRFs are positive, otherwise data.contact[0] shows last detected contact,
        even when there is no contact at all"""

        if self.getGRF() <= 0:
            raise Exception("getGroundContactYPosition() should only be called during stance phase, "
                            "as data.contact shows last detected contacts when there are no contacts at all")

        ground_contact = self.data.contact[0]
        contact_y_pos = ground_contact.pos[1]

        if self.lastTimestepGetContactPositionWasCalled != self.episode_step_count:
            self.is_foot_poss.append(contact_y_pos)
            self.lastTimestepGetContactPositionWasCalled = self.episode_step_count

        return contact_y_pos

    def getHipMotorTorque(self):
        """ :returns hip flexion-extension torque"""
        return np.around(self.data.actuator_force[MOT_HIP], 4)

    def getKneeMotorTorque(self):
        """  :returns the torque generated by the knee motor located on the hip"""
        return np.around(self.data.actuator_force[MOT_KNEE], 4)

    def getKneeJointTorque(self):
        """ :returns the torque that is applied on the knee through the pulley"""
        return np.around(self.data.actuator_force[MOT_VIRTUAL_KNEE], 4)


    #####################################################################################
    #                                                                                   #
    #                               HELPER FUNCTIONS                                    #
    #                                                                                   #
    #####################################################################################

    def collectDebuggingDataAndRenderSim(self):
        """
        At the beginning and every Y steps during training,
        the training is rendered for X steps to observe the so far learned behavior
        during the rendered steps, relevant data is collected and plotted afterwards.

        X and Y can be set in guro_train.py.
        """
        steps_to_render = config.RENDER_X_STEPS if not HOP_100_TIMES else int(5e5)

        modulo = self.step_count % config.CHECK_TRAINING_PROGRESS_EVERY_Y_STEPS
        if not modulo < steps_to_render:
            # disable ET together with some other features
            self.demonstration_on = False
        else:
            self.saveDebuggingData()

            # trainings beginning: do not stop ET
            if self.step_count > steps_to_render and modulo < steps_to_render / 2:
                self.demonstration_on = True
            # demonstration during training after Y steps
            else:
                # run the demo half the time without ET and then with ET to see which trajectories deviation lead to ET
                self.demonstration_on = False

            # inform user that demo during training starts and pause simulation renderings
            if modulo == 0:
                print(self.settings)
                say("Attention please!")
                time.sleep(5)
                self._get_viewer()._paused = True

            # after training demo is finished, plot the collected data
            if modulo == steps_to_render - 1 or (DEMO and modulo >= steps_to_render):
                if PLOT_DATA:
                    self.plotCollectedData(steps_to_render)

            if DO_RENDER:
                self._get_viewer().render()

        return modulo, steps_to_render


    def areActionsAndStateValid(self, a):
        """
        Checks the initial state and the current action output and returns whether it is valid or not
        :returns True if actions and state are valid
        """
        if a is None:
            raise AssertionError("Agents output was None at timestep {}".format(self.step_count))

        if not DEMO and (np.isnan(a[0])):
            raise AssertionError("Agent outputted nan values for motor torques")

        if self.episode_step_count == 1 and not self.IsInitialStateValid():
            return False

        return True


    def IsInitialStateValid(self):
        """
        Check if a new state was initialized slightly above the ground
        try to correct it if so and stop episode if correction is not possible
        """
        stop_episode = True

        if self.episode_step_count == 1 and self.getGRF() == 0:
            self.countFlightInits += 1
            self.flight_inits_rsi_timesteps.append(self.rsi_timestep)

            # decrease sledge pos until the robot is touching the ground again
            while self.getGRF() < 4:
                self.data.qpos[AXIS_SLEDGE] -= 0.001
                self.sim.forward()

            # sometimes the foot is too far away after sledge pos correction
            copPosMm = int(self.getGroundContactYPosition() * 1000)
            if copPosMm * 10 > FINITE_FOOT_POS_DEVIATION_CM:
                stop_episode = False

        return stop_episode


    def handleFlightPhaseAndTouchdown(self, collectDebugData):
        """
        During the Flight phase, the training is paused and the robot's motors are controlled by PD Position controllers
        bringing the robot in a desired flight phase posture.

        After the flight phase ends (touchdown, TD), the training resumes. Therefore a new reference hop is chosen
        based on TD conditions (sledge pos and vel).

        :returns True, if the takeoff in simulation happened too early, else False
        """
        has_entered_flight_phase = False
        countFlightPhaseSteps = 0

        # use PD Position control in flight phase... the agent only has to learn the stance phase behavior.
        # on TD the training resumes and the next ref hop will be chosen based on TD conditions
        while (TRAIN_ONLY_STANCE and self.getPhase() == PHASE_FLIGHT and not HOLD_SLEDGE_IN_PLACE):
            # count flight phase steps
            countFlightPhaseSteps += 1

            if countFlightPhaseSteps == 1:
                has_entered_flight_phase = True

            # during training, the (training) step count stays the same, as the agents training is "paused"
            if DEMO: self.step_count += 1

            # wait a bit before controlling the flight phase posture to allow the robot to move upwards first
            # otherwise when LO was at a smaller hip angle the posture control will increase it and the foot will hit the ground
            if countFlightPhaseSteps > 4:
                self.controlMotorsInFlightStep()

            # wait for at least 10 steps in flight to avoid detecting very short flight phases
            if countFlightPhaseSteps == 10:
                self.flightPhaseCount += 1

                # if the takeoff in simulation happened too early, stop the episode
                earlyLOForXSteps = self.refHopStancePhaseDuration - self.episode_step_count
                if earlyLOForXSteps >= ALLOWED_STANCE_DURATION_DEVIATION_TIMESTEPS \
                        and not DEMO and not self.demonstration_on:
                    return True
                elif earlyLOForXSteps > 0:  # bit early, but not too early
                    self.lateLOStepCount = earlyLOForXSteps

                # drop the ground in third flight phase
                if GROUND_DROP_EXPERIMENT and self.flightPhaseCount == 3:
                    # by changing the position of the ground plate
                    self.model.body_pos[6][2] = -0.1
                    self.drop_has_happend = True

            if collectDebugData:
                if PLOT_ALSO_FLIGHT_DATA: self.saveDebuggingData(noneForRefTrajecs=True)
                if DO_RENDER: self._get_viewer().render()
                # slow down simulation to make flight phase renderings about the same speed as stance phase
                if not JUST_APPLY_KIN_DATA: time.sleep(0.002)

        # -----------------------------------
        # TOUCHDOWN
        # -----------------------------------

        # if True, flight phase was simulated in a while loop. Thereafter TD was reached.
        if has_entered_flight_phase:

            if MUSCLES_ACTIVATED: self._initMuscles()  # reinitialize the muscle

            # set a new hop as reference trajectory
            # only trust flight phase detection when at least 5 steps were counted in flight and the phase after the flight phase isn't extension
            if JUST_APPLY_KIN_DATA or (countFlightPhaseSteps > 20 and self.getPhase() == PHASE_COMPRESS):
                # logging.info("TD detected after {} flight steps on episode step nr. {}".format(countFlightPhaseSteps, self.episode_step_count))
                if GROUND_DROP_EXPERIMENT and COLLECT_ONLY_PERTURBED_HOP_DATA and self.flightPhaseCount == 4:
                    self.saveHopDataToDict()
                    self.flightPhaseCount = 0
                    self.drop_has_happend = False
                    self.model.body_pos[6][2] = self.groundDropMeters - 0.002
                    # increase all ref sledge positions back to normal height
                    for hopIndex in range(len(self.allHopsRefData[0])):
                        self.allHopsRefData[0][hopIndex] += self.groundDropMeters
                    self.reset_model()

                self.timestepsInPreviousHopsOfCurrentEpisode += self.episode_step_count
                self.des_kin_trajecs = self.getRefHopBasedOnTDConditions(self.getSledgePos(), self.getSledgeVel())

        return False


    def plotCollectedData(self, steps_to_render):

        timestepInMillions = np.around(self.step_count, -5) / 1e6
        lowPassFreq = 20

        # Monitor Phase RSI and ET
        tools.log("Displaying Histograms for settings\n{}".format(self.settings))
        if len(self.episode_init_timesteps) > 5:
            tools.overlayingHistograms([self.episode_init_timesteps, self.episode_ends_timesteps],
                                       "Episode Initializations, Episode Terminations",
                                       xLabel="Percentage of the Stance Phase duration",
                                       yLabel="Quantity []")

        # debug kinematics: trajectory comparison
        tools.log("Displaying trajec comparison for settings\n{}".format(self.settings))
        tools.overlayingPlots(
            [[self.des_sledge_poss[-steps_to_render:], self.is_sledge_poss[-steps_to_render:]],
             [self.des_sledge_vels[-steps_to_render:], self.is_sledge_vels[-steps_to_render:]]],
            "Vertical\nCOM Position [m], Vertical\nCOM velocity [m/s]",
            legend="human,simulation;human,simulation",
            xLabels=["Time [s]"], title=None, thesis=True, sampleRate=int(2000 / FRAME_SKIP),
            shadeFlightPhase=True)

        # debug kinetics
        tools.overlayingPlots(
            [[self.knee_motor_torque_agent_outputs[-steps_to_render:],
              tools.lowpassFilterData(self.knee_motor_torque_agent_outputs[-steps_to_render:], 2e3 / FRAME_SKIP,
                                      lowPassFreq, 2)],
             [self.hip_motor_torque_agent_outputs[-steps_to_render:],
              tools.lowpassFilterData(self.hip_motor_torque_agent_outputs[-steps_to_render:], 2e3 / FRAME_SKIP,
                                      lowPassFreq, 2)],
             [self.grfs[-steps_to_render:],
              tools.lowpassFilterData(self.grfs[-steps_to_render:], 2e3 / FRAME_SKIP, lowPassFreq, 2)]],
            "Knee Motor\nTorque [Nm], Hip Motor\nTorque [Nm], GRF [N]",
            legend="unfiltered,filtered*;unfiltered,filtered*;unfiltered,filtered*", shadeFlightPhase=True,
            title=None, thesis=True, sampleRate=int(2000 / FRAME_SKIP))

        # debug velocity trajectories
        tools.overlayingPlots(
            [[self.des_sledge_vels, self.is_sledge_vels], [self.des_knee_ang_vels, self.is_knee_ang_vels],
             [self.des_hip_ang_vels, self.is_hip_ang_vels],
             [self.knee_motor_torque_agent_outputs[-steps_to_render:],
              self.knee_motor_torque_measured[-steps_to_render:]],
             [self.hip_motor_torque_agent_outputs[-steps_to_render:],
              self.hip_motor_torque_measured[-steps_to_render:]],
             self.phases, tools.lowpassFilterData(self.grfs, 2e3, 400, 2),
             self.all_rewards[-steps_to_render + 1:]],
            "Sledge\nVelocities, Knee Angle\nVelocities, Hip Angle\nVelocities, Knee Motor\nTorque, Hip Motor\nTorque, "
            "Phases\n[cmprs; extnd; fly], GRFs, Rewards",
            title="DEBUG: Velocity Comparison",
            legend="desired,measured;desired,measured;desired,measured;agent output,measured;agent output,measured; ; ; ")

        # monitor learning process
        if not DEMO and np.size(self.batch_mean_rewards) > 0:
            tools.severalPlots(
                [self.batch_mean_rewards, self.episode_rewards, self.batch_mean_episode_lengths],
                yLabels=["Mean Batch Rewards\n(from last {} steps)".format(np.power(2, 14)),
                         "Episode Rewards", "Mean Batch Episode Lengths\n(from last 40 episodes)"],
                title="Learning curves after {}M steps\nSettings: {})".format(timestepInMillions,
                                                                              self.settings))
        # -----------------------------------
        # CONDITIONAL PLOTS
        # -----------------------------------

        # debug Knee Extensor Muscle
        if MUSCLES_ACTIVATED:
            tools.severalPlots(
                [self.is_sledge_poss[-steps_to_render + 1:], self.stims_VAS[-steps_to_render + 1:],
                 self.is_knee_angs[-steps_to_render + 1:],
                 self.lens_mtc_VAS[-steps_to_render + 1:], self.lens_ce_VAS[-steps_to_render + 1:],
                 self.vels_VAS[-steps_to_render + 1:],
                 self.tor_VAS[-steps_to_render + 1:]],
                ["Sledge Pos", "Stim", "Knee Ang", "MTC length", "CE length", "CE velocity",
                 "Knee torque\ngenerated by muscle"], shareX=True,
                title="DEBUG Knee Extensor Muscle")

            tools.overlayingPlots(
                [[self.des_sledge_poss, self.is_sledge_poss], [self.des_knee_angs, self.is_knee_angs],
                 [self.knee_motor_torque_agent_outputs[-steps_to_render:],
                  self.knee_motor_torque_measured[-steps_to_render:],
                  self.knee_motor_torque_from_muscle_calculated_trajec[-steps_to_render:]],
                 self.agent_outputs_knee,
                 self.stims_VAS[-steps_to_render + 1:],
                 self.all_rewards[-steps_to_render:]],
                "Sledge Positions, Knee Angle, Knee Motor\nTorque, Agent\nOutputs, Mus\nActivation, Step Rewards",
                legend="desired, measured;desired,measured;desired (rope), measured, desired (muscle); ; ;",
                title="DEBUG: MUSCLE ACTIVATION at timestep {}M\n{}".format(timestepInMillions,
                                                                            self.settings))

        # additional plots not required at every run
        if THESIS:
            # check impact of filtering and changing angular velocities in simulation
            tools.overlayingPlots([[self.is_knee_ang_vels, self.is_knee_agn_vels_fltrd],
                                   [self.is_hip_pulley_ang_vels, self.is_hip_pulley_ang_vels_fltrd],
                                   self.phases],
                                  labels_comma_separated_string="Knee Axis [rad/s], Knee Motor Pulley\non the hip [rad/s], Phases\n[Compress-Extend-Flight]",
                                  legend="applied, filtered; applied, filtered; ",
                                  title="Comparison of Knee- and Hip Pulley angular velocities\nAs Filter an exponential running average with factor = 0.75 was used")

            # debug the rope mechanism
            tools.overlayingPlots([self.is_hip_pulley_angs, self.is_knee_angs, self.phases,
                                   np.multiply(self.is_rope_elongs[-steps_to_render + 1:], 1000),
                                   [self.knee_motor_torque_agent_outputs[-steps_to_render:],
                                    self.knee_motor_torque_measured[-steps_to_render:],
                                    self.knee_motor_torque_from_muscle_calculated_trajec[
                                    -steps_to_render:]],
                                   self.knee_axis_torque_actual_trajec[-steps_to_render:]],
                                  labels_comma_separated_string="Hip Pulley\nAngle [°], Knee Pulley\nAngle [°], Phases, Rope Elong [mm], Hip Pulley -/\nKnee Motor\nTorque [Nm], Measured\nKnee Axis\nTorque [Nm]",
                                  legend=" ; ; ; ; caclulated (rope), measured, calculated (mus); ",
                                  title="Rope Transmission Monitoring\nkRope = {} - dRope = {} - dJoints = {} - frictionJoints = {}\nSettings: {}".format(
                                      self.rope_stiffness, self.rope_damping,
                                      self.model.dof_damping[AXIS_KNEE_PULLEY_ON_THE_HIP],
                                      self.model.dof_frictionloss[AXIS_KNEE_PULLEY_ON_THE_HIP],
                                      self.settings))

            # Min Max Normalization Debugging
            tools.overlayingPlots(
                [self.is_sledge_normed_poss, [self.is_knee_normed_angs, self.is_hip_normed_angs],
                 [self.is_knee_vel_normed, self.is_hip_vel_normed], self.is_grfs_normed],
                labels_comma_separated_string="Sledge Positions, Joint Angles, Joint Angle Velocities, GRFs",
                legend=" ; knee, hip; knee, hip; ",
                title="Min Max Normalization Debugging")

            # learning curves for thesis
            tools.log("Displaying Learning curves for settings\n{}".format(self.settings))
            tools.severalPlots(
                [self.episode_rewards, self.batch_mean_episode_lengths],
                yLabels=["Episode Rewards []\n(filtered*)", "Mean Batch Episode\nLengths [timesteps]"],
                xLabels=["Episode Number []", "Number of 40 episodes batches []"],
                title=None, thesis=True)

        # reset the collected data
        self.initOrClearDebuggingCollections()

        # stop simulation, when a saved model was demonstrated during rendering
        if config.LOAD_MODEL:
            exit(0)


    def determineMotorCommandsFromAgentsOutputs(self, agents_action, frame):
        """
        Dependent on the current training settings, the agent can output different actions:
            - one or two motor torques
            - one or more muscle activations
        Given these actions, this function calculates the desired torques for three motors.
        :param agents_action: a list of one to three actions, being motor torques or muscle activations
        :param frame: current frame for own frameskip implementation
        :return: desired torques for the robot's motors in simulation, including the virtual motor on the knee axis
        """
        # agent outputs DESIRED MOTOR TORQUES for both or one of the motors
        if not MUSCLES_ACTIVATED:

            # if agent outputs only one single motor current, set the hip motor current to 0
            if len(agents_action) == 1:
                old_a = agents_action
                new_a = np.zeros(3)
                # set hip actuation to 0
                new_a[0] = 0
                new_a[1] = old_a[0]
                agents_action = new_a

            # clip agent's output to max. torque ranges
            agents_action[MOT_KNEE] = np.clip(agents_action[MOT_KNEE], -5, 5)
            agents_action[MOT_HIP] = np.clip(agents_action[MOT_HIP], -5, 5)

            if frame == 0 and not JUST_APPLY_KIN_DATA:
                self.knee_motor_torque_agent_outputs.append(agents_action[MOT_KNEE])
                self.hip_motor_torque_agent_outputs.append(agents_action[MOT_HIP])

            # agent outputs up to two desired motor torques (hip and knee motor)
            # here the rope transmission is simulated to also get the torque on the knee axis
            if len(agents_action) < 3: agents_action = np.append(agents_action, 0)
            agents_action[MOT_KNEE], agents_action[MOT_VIRTUAL_KNEE] = self.simulateRopeTransmission(agents_action[MOT_KNEE])
            if frame == 0:
                self.knee_motor_des_tor_with_rope_transmission.append(agents_action[MOT_KNEE])
                self.knee_axis_torque_des_trajec.append(agents_action[MOT_VIRTUAL_KNEE])

            agents_action = np.around(agents_action, 3)
            self.motor_input = agents_action
            motor_output = agents_action

        # agent outputs MUSCLE ACTIVATION
        else:

            agents_action[MUS_VAS] = np.clip(agents_action[MUS_VAS], -5, 5)
            delayMuscleStims = False

            # scale agent output with its range of [-5:5] to the possible muscle activation
            muscleStimulation = tools.linearScale(agents_action[MUS_VAS], -5 if not DEMO else 0, 5, 0, 1)

            if delayMuscleStims:
                # simulate muscle sensory information delay
                agents_action[MUS_VAS] = self.musVasActBuffer[-1]
                self.musVasActBuffer = np.roll(self.musVasActBuffer, 1)
                self.musVasActBuffer[0] = np.clip(muscleStimulation, 0.001, 1)
            else:
                agents_action[MUS_VAS] = muscleStimulation

            if agents_action[MUS_VAS] > 1 or agents_action[MUS_VAS] < 0:
                raise AssertionError(
                    "Muscles were activated, but activation was bigger 1 or smaller 0: {}".format(agents_action[MOT_KNEE]))

            # given the activation of the simulated muscle, determine the desired torques in the motors
            motor_output = np.zeros(3)
            motor_output[MOT_HIP] = 0
            motor_output[MOT_KNEE] = self.getAppliedKneeTorqueFromMuscleStims(agents_action[MUS_VAS])
            if frame == 0:
                self.knee_motor_torque_from_muscle_calculated_trajec.append(motor_output[MOT_KNEE])
            motor_output[MOT_KNEE], motor_output[MOT_VIRTUAL_KNEE] = \
                self.simulateRopeTransmission(motor_output[MOT_KNEE])
        return motor_output


    def logException(self):
        self.printTraceback()
        say("Exception was catched!")
        print("------------\n Exception catched during simulation step at timestep {}".format(self.step_count))
        print("Motor input: {}".format(self.motor_input))
        print("current state: {}".format(self.previousState))
        print("Last 10 rewards: {}".format(self.all_rewards[-10:]))


    def do_simulation_by_applying_ref_trajecs(self):
        """
        Useful to test if reference trajectories can be reproduced in simulation and result in the desired motion.
        At every timestep, the robots kinematics are just set to the values from ref trajecs
        or are followed by PD Position Controllers
        """
        follow_ref_trajec_with_PD_Controllers = PD_POSITION_CONTROL

        if self.episode_step_count == self.refHopStancePhaseDuration - 1:
            self.initParamsAndContainers()

        if not follow_ref_trajec_with_PD_Controllers:
            self.setKinematics(self.episode_step_count)
            self.do_simulation([0, 0, 0], 1)
        else:
            des_kinematics = self.getDesiredKinematicsForTimestep(self.episode_step_count)
            ref_knee_ang = des_kinematics[2]
            ref_hip_ang = des_kinematics[3]
            des_hip_motor_torque, des_knee_motor_torque = \
                tools.getDesiredTorquesFromPositionPID(deg(ref_hip_ang), deg(ref_knee_ang),
                                                       deg(self.getHipAng()), deg(self.getKneeAng()),
                                                       in_flight=False)
            des_knee_motor_torque, knee_axis_torque = self.simulateRopeTransmission(des_knee_motor_torque)
            self.do_simulation([des_hip_motor_torque, des_knee_motor_torque, knee_axis_torque], 1)


    def initOrClearDebuggingCollections(self):
        self.des_knee_angs, self.des_hip_angs, self.des_knee_ang_vels, self.des_hip_ang_vels, self.is_knee_angs, self.is_hip_angs, self.is_knee_ang_vels, self.is_hip_ang_vels = [], [], [], [], [], [], [], []
        self.des_knee_accs, self.des_hip_accs, self.is_knee_accs, self.is_hip_accs, self.phases, self.is_foot_poss = [], [], [], [], [], []
        self.agent_outputs_knee, self.agent_outputs_hip, self.is_sledge_poss, self.is_sledge_vels, self.des_sledge_poss, self.des_sledge_vels = [], [], [], [], [], []
        self.is_rope_elongs, self.knee_motor_torque_agent_outputs, self.knee_motor_torque_measured, self.knee_axis_torque_des_trajec, self.knee_axis_torque_actual_trajec = [], [], [], [], []
        self.is_hip_pulley_angs, self.is_hip_pulley_ang_vels, self.is_hip_pulley_ang_vels_fltrd, self.is_knee_agn_vels_fltrd = [], [], [], []
        self.knee_motor_torque_agent_outputs, self.knee_motor_torque_from_muscle_calculated_trajec, self.knee_motor_torque_measured, self.knee_axis_torque_des_trajec, self.knee_axis_torque_actual_trajec = [], [], [], [], []
        self.is_sledge_normed_poss, self.is_knee_normed_angs, self.is_hip_normed_angs, self.is_knee_vel_normed, self.is_hip_vel_normed, self.is_grfs_normed, self.grfs = [], [], [], [], [], [], []
        self.hip_motor_torque_agent_outputs, self.hip_motor_torque_measured = [], []

        if MUSCLES_ACTIVATED:
            self.stims_VAS, self.lens_mtc_VAS, self.lens_ce_VAS, self.vels_VAS, self.frcs_VAS, self.tor_VAS = [],[],[],[],[],[]


    def printTraceback(self):
        traceback.print_stack()


    def stopEpisode(self):
        return self.determineNewState(), 0, True, {}


    def controlMotorsInFlightStep(self):
        """
        Uses PD Position controllers to bring the robotic leg in a desired flight phase posture.
        The posture is defined by the flight phase hip and knee angles.
        """
        des_hip_motor_torque, des_knee_motor_torque = \
            tools.getDesiredTorquesFromPositionPID(config.FLIGHT_PHASE_HIP_ANGLE, config.FLIGHT_PHASE_KNEE_ANGLE,
                                                   deg(self.getHipAng()), deg(self.getKneeAng()))
        des_knee_motor_torque, knee_axis_torque = self.simulateRopeTransmission(des_knee_motor_torque)
        self.do_simulation([des_hip_motor_torque, des_knee_motor_torque, knee_axis_torque], 1)


    def saveDebuggingData(self, noneForRefTrajecs = False):
        """
        Collects relevant data to be plotted to monitor the training progress.
        :param noneForRefTrajecs: set to True, to save None's for the desired trajectories during flight phase.
                                  This way, the ref trajecs are not plotted during the flight phase
        """

        if self.getPhase()==PHASE_FLIGHT:
            self.hip_motor_torque_agent_outputs.append(0)
            self.knee_motor_torque_agent_outputs.append(0)

        # desired / reference kinematics
        des_kinematics = self.getDesiredKinematicsForTimestep(self.episode_step_count)
        i_sledge_posits, i_sledge_vels, i_knee_angs, i_hip_angs, i_knee_ang_vels, i_hip_ang_vels, i_grfs = range(7)
        self.des_sledge_poss.append(None if noneForRefTrajecs else des_kinematics[i_sledge_posits])
        self.des_knee_angs.append(None if noneForRefTrajecs else deg(des_kinematics[i_knee_angs]))
        self.des_hip_angs.append(None if noneForRefTrajecs else deg(des_kinematics[i_hip_angs]))
        self.des_sledge_vels.append(None if noneForRefTrajecs else des_kinematics[i_sledge_vels])
        self.des_knee_ang_vels.append(None if noneForRefTrajecs else des_kinematics[i_knee_ang_vels])
        self.des_hip_ang_vels.append(None if noneForRefTrajecs else des_kinematics[i_hip_ang_vels])

        # measured / current kinematics
        self.is_sledge_poss.append(self.getSledgePos())
        self.is_sledge_vels.append(self.getSledgeVel())
        self.is_knee_angs.append(deg(self.getKneeAng()))
        self.is_knee_ang_vels.append(self.getKneeAngVel())
        self.is_hip_angs.append(deg(self.getHipAng()))
        self.is_hip_ang_vels.append(self.getHipAngVel())
        self.is_hip_pulley_angs.append(deg(self.getHipPulleyAng()))

        # kinetics and others
        self.phases.append(self.getPhase())
        self.knee_motor_torque_measured.append(self.getKneeMotorTorque())
        self.hip_motor_torque_measured.append(self.getHipMotorTorque())
        self.knee_axis_torque_actual_trajec.append(self.getKneeJointTorque())
        self.grfs.append(self.getGRF())
        if self.getGRF() == 0: self.is_foot_poss.append(-1e-2)

        if MUSCLES_ACTIVATED: self.saveMuscleDebuggingData()


    def stop_rendering(self):
        global DO_RENDER
        DO_RENDER = False


    def toggleMuscles(self, activate):
        global MUSCLES_ACTIVATED
        MUSCLES_ACTIVATED = activate


    def demoModeOn(self):
        global DEMO
        DEMO = True


    def saveHopDataToDict(self):
        """
        If HOP_100_TIMES is True, collects the GRFs, legLen and Phase informations for 100 consecutive hops
        and saves then to a dictionary in the folder "sim_data".
        """
        fileName = "NEW148_knHpET_wP5xwV_flKnee_10p20v_eplen800_64r_15208_splRew600_16LO_3M_LR3f5_s2097_r96_l31"
        skipFirstXSteps = 0
        if HOP_100_TIMES and self.episode_step_count > skipFirstXSteps + 1:
            grfs = self.grfs[
                   -self.episode_step_count + skipFirstXSteps:]  # tools.lowpassFilterData(self.grfs[-self.episode_step_count:], 2e3, 40, 2)
            # tools.plot(grfs, title="saved grfs from current hop")
            legLens = self.is_sledge_poss[-self.episode_step_count + skipFirstXSteps:]
            legStifs = [(grfs[i + 1] - grfs[i]) / (legLens[i + 1] - legLens[i] + 1e-7) for i in
                        range(self.episode_step_count - skipFirstXSteps - 1)]
            phases = self.phases[-self.episode_step_count:]

            self.nHopsData["grfs"].append(grfs)
            self.nHopsData["legLen"].append(legLens)
            self.nHopsData["legStif"].append(legStifs)
            self.nHopsData["phases"].append(phases)
            countSavedHops = len(self.nHopsData["grfs"])
            print("\nSaved {} hop with length {}".format(countSavedHops, self.episode_step_count))
            if countSavedHops == 50:
                tools.overlayingPlots([self.nHopsData["grfs"]], "grfs recorded from sim")
                savePath = settings.PATH_THESIS_FOLDER + "python/training/sim_data"
                fileName = ("Prtbt_" if COLLECT_ONLY_PERTURBED_HOP_DATA else "") + \
                           fileName + "_{}".format(np.random.randint(100))
                np.save(savePath + fileName, self.nHopsData)
                exit(33)


    def normalizeDataMinMax(self):
        """
        Normalization by using Min and Max Values for every state.
        This kind of normalization turned out to be less beneficial than the Mean and Std normalization.

        NOT USED IN THE MOMENT!
        """

        USE_KNEE_MOTOR_INSTEAD_KNEE_AXIS = True

        # get new state values
        sledge_pos = self.getSledgePos()
        sledge_vel = self.getSledgeVel()
        knee_ang = self.getHipPulleyAng() if USE_KNEE_MOTOR_INSTEAD_KNEE_AXIS else self.getKneeAng()
        hip_ang = self.getHipAng()
        knee_ang_vel = self.getHipPulleyAngVel() if USE_KNEE_MOTOR_INSTEAD_KNEE_AXIS else self.getKneeAngVel()
        hip_ang_vel = self.getHipAngVel()
        grf = self.getGRF()

        newState = self.previousState
        newState[self.STATE_SLEDGE_VEL] = tools.normalizeMinMax(sledge_vel, -1, 1)
        newState[self.STATE_SLEDGE_POS] = tools.normalizeMinMax(sledge_pos, 0.4, 0.55)
        newState[self.STATE_KNEE_MOT_ANG] = tools.normalizeMinMax(knee_ang, rad(-100 if USE_KNEE_MOTOR_INSTEAD_KNEE_AXIS else 110),
                                                                  rad(30 if USE_KNEE_MOTOR_INSTEAD_KNEE_AXIS else 165))
        newState[self.STATE_KNEE_MOT_ANG_VEL] = tools.normalizeMinMax(knee_ang_vel, -25 if USE_KNEE_MOTOR_INSTEAD_KNEE_AXIS else -5,
                                                                      25 if USE_KNEE_MOTOR_INSTEAD_KNEE_AXIS else 5)
        newState[self.STATE_HIP_ANG] = tools.normalizeMinMax(hip_ang, rad(10), rad(35))
        newState[self.STATE_HIP_ANG_VEL] = tools.normalizeMinMax(hip_ang_vel, -3, 3)

        self.is_sledge_normed_poss.append(newState[self.STATE_SLEDGE_POS])
        self.is_knee_normed_angs.append(newState[self.STATE_KNEE_MOT_ANG])
        self.is_hip_normed_angs.append(newState[self.STATE_HIP_ANG])
        self.is_knee_vel_normed.append(newState[self.STATE_KNEE_MOT_ANG_VEL])
        self.is_hip_vel_normed.append(newState[self.STATE_HIP_ANG_VEL])

        return newState


    def normalizeWithRefDataMeanAndVariance(self):
        """
        Normalizes the sensor information by using their mean and variance derived from the reference trajectories.
        In the case of the hip pulley kinematics reference trajectories were collected from experiment,
        where GURPO hopped by following the hip and knee ref trajectories by using PD position controllers.
        """

        # get new state values
        sledge_pos = self.getSledgePos()
        # correct the sledge position if GURO is hopping on the ground plattform
        if GROUND_DROP_EXPERIMENT and not self.drop_has_happend:
            sledge_pos -= self.groundDropMeters
        sledge_vel = self.getSledgeVel()
        knee_ang = np.around(self.getHipPulleyAng(),2)
        hip_ang = self.getHipAng()
        knee_ang_vel = np.around(self.getHipPulleyAngVel(),2)
        hip_ang_vel = self.getHipAngVel()

        # derived from reference trajectories
        means = [-0.8399072558985972, 0.39362465587763634, -0.02569200933877392, -0.11370739423088674, 0.01929697539211253, 0.5066570016460371]
        stds = [0.49933613195808546, 0.05892244400925175, 13.857572358222596, 1.883898540073142, 0.40281297721754417, 0.013244542063715208]

        newState = self.previousState

        # normalize the inputs by using means and standard deviations derived from reference trajectories
        newState[self.STATE_SLEDGE_POS] = (sledge_pos - means[self.STATE_SLEDGE_POS]) / stds[self.STATE_SLEDGE_POS]
        newState[self.STATE_SLEDGE_VEL] = (sledge_vel - means[self.STATE_SLEDGE_VEL]) / stds[self.STATE_SLEDGE_VEL]
        newState[self.STATE_KNEE_MOT_ANG] = (knee_ang - means[self.STATE_KNEE_MOT_ANG]) / stds[self.STATE_KNEE_MOT_ANG]
        newState[self.STATE_KNEE_MOT_ANG_VEL] = (knee_ang_vel - means[self.STATE_KNEE_MOT_ANG_VEL]) / stds[self.STATE_KNEE_MOT_ANG_VEL]
        newState[self.STATE_HIP_ANG] = (hip_ang - means[self.STATE_HIP_ANG]) / stds[self.STATE_HIP_ANG]
        newState[self.STATE_HIP_ANG_VEL] = (hip_ang_vel - means[self.STATE_HIP_ANG_VEL]) / stds[self.STATE_HIP_ANG_VEL]

        return newState


    def setKinematics(self, timestep, all_trajecs=None, also_set_accs = False):
        """
        Simply sets all joint kinematics to the values from the reference trajectory.
        Useful to check, whether ref trajectories are correct and result in the desired motion.
        """

        sledge_positions, sledge_vels, knee_angles, hip_angles, knee_ang_vels, hip_ang_vels, grfs = range(7)

        # get vector of desired kinematic values
        des_kinematics = self.getDesiredKinematicsForTimestep(timestep)

        # hip joint
        self.data.qpos[AXIS_HIP] = des_kinematics[hip_angles]
        self.data.qvel[AXIS_HIP] = des_kinematics[hip_ang_vels]
        if also_set_accs: self.data.qacc[AXIS_HIP] = des_kinematics[9]

        # knee joint
        self.data.qpos[AXIS_KNEE] = des_kinematics[knee_angles]
        self.data.qvel[AXIS_KNEE] = des_kinematics[knee_ang_vels]
        if also_set_accs: self.data.qacc[AXIS_KNEE] = des_kinematics[8]

        if HOLD_SLEDGE_IN_PLACE:
            self.data.qpos[0] = 0.65
            self.data.qvel[0] = 0
            self.data.qacc[0] = 0
        else:
            self.data.qpos[AXIS_SLEDGE] = des_kinematics[sledge_positions]
            self.data.qvel[AXIS_SLEDGE] = des_kinematics[sledge_vels]
            if also_set_accs: self.data.qacc[AXIS_SLEDGE] = des_kinematics[7]



    def getSledgePosAndVelDeviations(self, episode_timestep):
        """
        Calculates the deviation between the sledge position and velocity
        in simulation and reference trajectories.
        """
        ref_kinematics = self.getDesiredKinematicsForTimestep(episode_timestep)
        ref_sledge_pos, ref_sledge_vel = ref_kinematics[[0, 1]]
        is_sledge_pos = self.getSledgePos()
        is_sledge_vel = self.getSledgeVel()
        pos_difference_cm = abs(ref_sledge_pos - is_sledge_pos) * 100
        vel_difference_cm_per_s = abs(ref_sledge_vel - is_sledge_vel) * 100
        return pos_difference_cm, vel_difference_cm_per_s


    def getKinematicsDistance(self, episode_timestep):
        """
        Calculates the deviations in velocity and position for knee and hip joints
        """
        sledge_positions, sledge_vels, knee_angles, hip_angles, knee_ang_vels, hip_ang_vels, grfs = range(7)
        angle_indices = [knee_angles, hip_angles]
        vel_indices = [knee_ang_vels, hip_ang_vels]
        # get vector of desired kinematic values
        des_kinematics = self.getDesiredKinematicsForTimestep(episode_timestep)
        des_angles = deg(des_kinematics[angle_indices])
        des_vels = des_kinematics[vel_indices]
        # get vector of corresponding actual values
        angles = deg([self.getKneeAng(), self.getHipAng()])
        vels = [self.getKneeAngVel(), self.getHipAngVel()]
        # calculate pos and vel reward
        angle_dists = np.abs(np.subtract(des_angles, angles))
        vel_dists = np.abs(np.subtract(des_vels, vels))
        return angle_dists, vel_dists, angles, vels, des_angles, des_vels


    def getDesiredKinematicsForTimestep(self, episode_timestep):
        """
        :returns the sledge and joint position and velocities from the reference trajectories
                 for the given episode_timestep.
        """
        trajec_len  = np.shape(self.des_kin_trajecs)[0]
        # do not add rsi_timesteps when agent is already in the second hop of an episode
        episode_timestep += self.rsi_timestep
        if episode_timestep > trajec_len-1:
            episode_timestep = trajec_len-1
        return self.des_kin_trajecs[episode_timestep, :]


    #####################################################################################
    #                                                                                   #
    #                                   MODELING                                        #
    #                                                                                   #
    #####################################################################################

    def simulateRopeTransmission(self, set_knee_motor_torq):
        # pulley radii in m
        radHip = 20e-3/2
        radKnee = 80e-3/2

        # rope stiffness and damping
        self.rope_stiffness = 1e7
        self.rope_damping = 1e2

        # set initial knee angle with high accuracy as it is important for rope transmission
        if self.initKneeAngle is None:
            self.initKneeAngle = self.getKneeAng()

        # angle differences in relation to initial angles
        deltaPhiHip = self.getHipPulleyAng()
        deltaPhiKnee = self.initKneeAngle - self.getKneeAng()

        # angular velocities
        hip_ang_vel = self.getHipPulleyAngVel()
        knee_ang_vel = self.getKneeAngVel()

        # how much rope is wound up on both pulleys
        upper_rope_on_hip = deltaPhiHip * radHip
        upper_rope_on_knee = deltaPhiKnee * radKnee
        lower_rope_on_hip = - upper_rope_on_hip
        lower_rope_on_knee = - upper_rope_on_knee

        upper_rope_elong = upper_rope_on_hip + upper_rope_on_knee
        upper_rope_elong_vel = hip_ang_vel * radHip - knee_ang_vel * radKnee

        # only collect rope elongation data in stance phase
        if self.getPhase() != PHASE_FLIGHT:
            self.is_rope_elongs.append(upper_rope_elong)

        # calculate rope force from elongation and its velocity
        # note: the force in the lower rope is always the same as in the upper with an opposite sign
        upper_rope_force = upper_rope_elong * self.rope_stiffness + upper_rope_elong_vel * self.rope_damping
        self.rope_forces.append(upper_rope_force)


        knee_motor_torq = set_knee_motor_torq - upper_rope_force * radHip
        virtual_knee_motor_torq = upper_rope_force * radKnee

        # get motor torques
        if abs(upper_rope_force) > 100:
            stop = True

        return knee_motor_torq, virtual_knee_motor_torq


    def simulateElectricalMotorDynamics(self, motor_index, desired_torque):
        """ Simple simulation where the desired motor torque is low pass filtered by a smoothing average."""
        return tools.exponentialRunningSmoothing(motor_index, desired_torque, 0.22)


    def getAppliedKneeTorqueFromMuscleStims(self, stims, plot=False):
        stims = np.clip(stims, 0.001, 1)
        self.musVAS.stim = stims
        #self.musBFSH.stim = stims[1]

        knee_ang = self.getKneeAng()
        self.musVAS.stepUpdateState(np.array((knee_ang,)))
        # self.musBFSH.stepUpdateState(np.array((knee_ang,)))

        # found by calculating max torque from muscle and comparing it to max motor torque
        # 1/72 comes from T_mot_max/F_max_iso*N; 3.2 is found by hand to achieve hopping
        muscle_force_scaling = 1 # 1/72 * 3.2

        knee_torque = self.musVAS.frcmtc * self.musVAS.levelArm * muscle_force_scaling
        # - self.musBFSH.frcmtc * self.musBFSH.levelArm * muscle_force_scaling

        if knee_torque > 10:
            debug = True

        self.saveMuscleDebuggingData()

        if (plot or len(self.stims_VAS) == 1505) and PLOT_DATA and MUSCLES_ACTIVATED:

            tools.severalPlots([self.is_sledge_poss[-1500:], self.stims_VAS[-1500:], self.agent_outputs_knee[-1500:], self.is_knee_angs[-1500:],
                                self.lens_mtc_VAS[-1500:], self.lens_ce_VAS[-1500:], self.vels_VAS[-1500:], self.tor_VAS[-1500:]],
                               ["Sledge Pos", "Stim", "Agents\nOutput", "Knee Ang", "MTC length", "CE length", "CE velocity", "Knee torque\ngenerated by muscle"], shareX=True,
                               title="DEBUG VASTUS Muscle")

        return knee_torque


    def saveMuscleDebuggingData(self):
        self.stims_VAS.append(self.musVAS.stim)
        self.lens_mtc_VAS.append(self.musVAS.lmtc)
        self.lens_ce_VAS.append(self.musVAS.lce)
        self.vels_VAS.append(self.musVAS.vce)
        self.frcs_VAS.append(self.musVAS.frcmtc)
        knee_torque_from_muscle = self.musVAS.frcmtc * self.musVAS.levelArm
        self.tor_VAS.append(knee_torque_from_muscle)