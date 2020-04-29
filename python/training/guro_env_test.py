import gym
import time
import numpy as np

# necessary to use GuroEnv
from thesis_galljamov18.python.guro_gym_env import gym_guro
from thesis_galljamov18.python import tools


# -----------------------
# SIMULATION CONTROL
# -----------------------

NUM_TIMESTEPS = 6000
RENDER_SIM = True
PLOT_DATA = True

THESIS = False


# -----------------------
# AGENT PARAMS
# -----------------------

guro_env = None
step_count = 0
rewards = []


# -----------------------
# KINEMATICS data
# -----------------------

knee_angles = np.zeros(NUM_TIMESTEPS)
hip_angles = np.zeros(NUM_TIMESTEPS)
sledge_positions = np.zeros(NUM_TIMESTEPS)

knee_ang_vels = np.zeros(NUM_TIMESTEPS)
hip_ang_vels = np.zeros(NUM_TIMESTEPS)
sledge_vels = np.zeros(NUM_TIMESTEPS)

knee_ang_accs = np.zeros(NUM_TIMESTEPS)
hip_ang_accs = np.zeros(NUM_TIMESTEPS)
sledge_accs = np.zeros(NUM_TIMESTEPS)

set_hip_angs = []
set_knee_angs = []

is_knee_angs = []
is_hip_angs = []
is_phases = []


# -----------------------
# KINETICS data
# -----------------------

grfs = np.zeros(NUM_TIMESTEPS)
phases = np.zeros(NUM_TIMESTEPS)

hip_motor_torque_trajec = np.zeros(NUM_TIMESTEPS)
hip_motor_angle_trajec = np.zeros(NUM_TIMESTEPS)

# torque of the real knee motor located at the hip
knee_motor_torque_trajec = np.zeros(NUM_TIMESTEPS)
des_knee_motor_torque_trajec = np.zeros(NUM_TIMESTEPS)

# torque from the virtual motor on the knee,
# calculated from the real knee motor on the hip considering transmission ratio, inertia and damping
knee_axis_torque_trajec = np.zeros(NUM_TIMESTEPS)

is_hip_motor_torques = []
is_knee_motor_torques = []


# -----------------------
# HELPER METHODS
# -----------------------

def rad(degrees):
    return degrees*np.pi/180

def deg(angles_in_rad):
    if isinstance(angles_in_rad,list):
        angles_in_rad = np.array(angles_in_rad)
    return angles_in_rad/np.pi*180


def saveTrajecs():
    """saves trajectories of all kinetic and kinematic properties
    to be used as a reference trajectory during training.
    The trajectories are saved in a .txt file in a import friendly format."""

    trajecs = [sledge_positions, sledge_vels, knee_angles, hip_angles, knee_ang_vels, hip_ang_vels, grfs,
               phases, sledge_accs, knee_ang_accs, hip_ang_accs, des_knee_motor_torque_trajec]

    all_trajecs = np.zeros([NUM_TIMESTEPS, len(trajecs)])

    if PLOT_DATA:
        tools.severalPlots( [sledge_positions, sledge_vels, knee_angles, hip_angles, knee_ang_vels, hip_ang_vels, grfs],
                            yLabels=["sledge pos; sledge vels; knee ang; hip ang; knee ang vel; hip ang vel; grfs"],
                            title="Trajectories recorded from the simulatioin as the imitation target")

        tools.severalPlots([sledge_positions, grfs, phases], yLabels=["sledge pos; GRFs; Phases"], title="Phases check")

    for i in range(len(trajecs)):
        all_trajecs[:,i] = trajecs[i]

    np.savetxt('guro_STANCE_ref_trajecs.txt', all_trajecs, delimiter=',', fmt='%f')
    return True


def loadTrajecs(test=False):
    data = np.loadtxt('guros_workout_trajecs.txt', delimiter=',', skiprows=1)
    if test:
        tools.severalPlots([data[:, 0], data[:, 1], data[:, 2], data[:, 4], data[:, 3], data[:, 5]],
                       yLabels=["Sledge Pos", "Sledge Vel", "Knee Ang", "Knee Ang Vel", "Hip Ang", "Hip Ang Vel"],
                       title="Desired Kinematic Trajectories")


def loadTrajecsAndFilter(test=False):
    sledge_positions, sledge_vels, knee_angles, hip_angles, knee_ang_vels, hip_ang_vels, grfs = range(7)

    path = settings.PATH_THESIS_FOLDER + 'python/training/guros_workout_trajecs.txt'
    data = np.loadtxt(path, delimiter=',', skiprows=1)

    # crop trajectory to get exactly three hops
    # -------------------------------------------

    # find first apex:
    start_index = np.argmax(data[2000:3000, sledge_positions]) + 2000
    end_index = np.argmax(data[6700:7200, sledge_positions]) + 6700

    data = data[start_index:end_index, :]

    fltrd_data = np.zeros([np.shape(data)[0], 2])
    fltrd_data[:, 0] = tools.lowpassFilterData(data[:, knee_ang_vels], 2e3, 12, 1)
    fltrd_data[:, 1] = tools.lowpassFilterData(data[:, hip_ang_vels], 2e3, 12, 1)

    # filter velocity data
    data[:, knee_ang_vels] = fltrd_data[:, 0]
    data[:, hip_ang_vels] = fltrd_data[:, 1]

    return data


# --------------------------------------------
# Hand tuned NOT-learning-based CONTROL APPROACHES
# --------------------------------------------

from thesis_galljamov18.python.tools import exponentialRunningSmoothing as smoothen
SMOOTH_FACTOR = 0.01

apex_point_reached = False
touchdown_happend = False
touchdown_timesteps = 0

already_was_in_flight = False
already_was_on_ground = False


def setDesiredTorques():
    """ simple phase dependent bang bang control
    :returns list of two set torques for the hip and the knee motor [tor_hip, tor_knee]"""

    global touchdown_happend, apex_point_reached, desired_torque_knee_motor, desired_hip_torq, touchdown_timesteps, already_was_in_flight, already_was_on_ground

    # Quick and Dirty!
    # grfs.append(env.getGRF())
    touchdown_happend = touchdown_happend or grfs[-1] > 20
    # agent output parameters (knee and hip motor torque)
    if touchdown_happend:
        already_was_on_ground = True
        touchdown_timesteps += 1
        if not apex_point_reached:
            sledge_pos = guro_env.getSledgePos()
            apex_point_reached = (sledge_pos <= 0.35)

        if apex_point_reached:
            desired_torque_knee_motor = tools.exponentialRunningSmoothing(1, 7, SMOOTH_FACTOR)
            desired_hip_torq = tools.exponentialRunningSmoothing(0, 0, SMOOTH_FACTOR)

        elif guro_env.getGRF() > 0:
            # compression phase
            if not already_was_in_flight and deg(guro_env.getHipAng()) > 45:
                desired_torque_knee_motor = smoothen(1, -1, SMOOTH_FACTOR)
                desired_hip_torq = smoothen(0, 1, SMOOTH_FACTOR)
            elif already_was_in_flight:
                desired_torque_knee_motor = smoothen(1, -1, SMOOTH_FACTOR)
                desired_hip_torq = smoothen(0, 1, SMOOTH_FACTOR)
            else:
                desired_torque_knee_motor = smoothen(1, 0.5, SMOOTH_FACTOR)

        else:
            desired_torque_knee_motor = smoothen(1, 7, SMOOTH_FACTOR)
            desired_hip_torq = smoothen(0, 0, SMOOTH_FACTOR)

        # LO detected
        if guro_env.getGRF() == 0:
            apex_point_reached = False
            touchdown_happend = False
            touchdown_timesteps = 0
    else:
        # flight phase
        if not already_was_on_ground:
            return [0, 0]
        already_was_in_flight = True
        desired_hip_torq = smoothen(0, 0.5, 30 * SMOOTH_FACTOR)
        desired_torque_knee_motor = smoothen(1, -0.45, 30 * SMOOTH_FACTOR)
        apex_point_reached = False
    # desired_angle_hip = rad(5) if grfs[-1] > 50 else rad(5)
    return [desired_hip_torq, desired_torque_knee_motor]


# initialize PIDs for angle control
hip_pos_pid = tools.getPID(0.42, 0, 8.6, -5, 5, 25e-4)
knee_pos_pid = tools.getPID(0.44, 0, 2.2, -5, 5, 25e-4)


def setDesiredTorquesForPosition(des_hip_angle, des_knee_angle):
    """Get Torques to reach desired hip and knee angles."""
    guro_env.toggleMuscles(False)

    hip_pos_pid.setpoint = des_hip_angle
    knee_pos_pid.setpoint = des_knee_angle

    set_hip_angs.append(des_hip_angle)
    set_knee_angs.append(des_knee_angle)

    hip_motor_tor = hip_pos_pid(guro_env.getHipAng())
    knee_motor_tor = knee_pos_pid(guro_env.getKneeAng())

    return [hip_motor_tor, knee_motor_tor]


# ------------------
# RUN SIMULATION
# -----------------

def main():
    global step_count
    global guro_env

    guro_env = gym.make('Guro-v0')

    if not RENDER_SIM:
        guro_env.stop_rendering()

    # disables logs only required during training
    guro_env.demoModeOn()

    for _ in range(NUM_TIMESTEPS):

        # simulate step
        observation, reward, done, info = guro_env.step(setDesiredTorquesForPosition(2, 175))

        if done:
            guro_env.reset()
            continue

        if PLOT_DATA:
            # kinematics
            knee_angles[step_count] = guro_env.getKneeAng()
            hip_angles[step_count] = guro_env.getHipAng()
            sledge_positions[step_count] = guro_env.getSledgePos()
            knee_ang_vels[step_count] = guro_env.getKneeAngVel()
            hip_ang_vels[step_count] = guro_env.getHipAngVel()
            sledge_vels[step_count] = guro_env.getSledgeVel()
            knee_ang_accs[step_count] = guro_env.getKneeAngAcc()
            hip_ang_accs[step_count] = guro_env.getHipAngAcc()
            sledge_accs[step_count] = guro_env.getSledgeAcc()

            # kinetics
            grfs[step_count] = guro_env.getGRF()
            phases[step_count] = guro_env.getPhase()
            knee_axis_torque_trajec[step_count] = guro_env.getKneeJointTorque()
            # des_knee_motor_torque_trajec[step_count] = desired_torque_knee_motor
            # des_hip_angle_trajec[step_count] = desired_hip_torq

        if RENDER_SIM:
            guro_env.render()

        step_count += 1


    # plot PID control results
    if PLOT_DATA and THESIS:
        # print("Raising time of the Knee PID is: {}ms".format(np.round(np.where(np.array(is_knee_angs)<80)[0][0]*5e-1, 2)))
        tools.overlayingPlots([[set_hip_angs, is_hip_angs], is_hip_motor_torques, [set_knee_angs, is_knee_angs], is_knee_motor_torques],
                              labels_comma_separated_string="Hip Angle, Hip Motor Torque, Knee Angle, Knee Motor Torque",
                              title="PID Position Control\nHip Pos Params: Kp = {}, Ki = {}, Kd = {}"
                                    "\nKnee Pos Params: Kp = {}, Ki = {}, Kd = {}".format(hip_pos_pid.Kp, hip_pos_pid.Ki, hip_pos_pid.Kd,
                                                                                          knee_pos_pid.Kp, knee_pos_pid.Ki, knee_pos_pid.Kd),
                              legend="Set Angle, Measured Angle; ; Set Angle, Measured Angle;")

    # save trajectories collected from hand tuned controllers to be used as a reference for training
    assert saveTrajecs()


if __name__ == '__main__':
    main()