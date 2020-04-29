"""
To identify the sledge's frictional forces, an experiment was conducted.
Here the sledge was dropped 5 times from different heights and it's vertical position and the GRFs were collected.

This script:
    - Loads the sledge trajectories collected from the the experiment
    - reproduces the same experiment in an adjusted mujoco environment
    - uses Bayesian Optimization (BO) to minimize the differences between the experiment and simulation trajectories
      and thus find the optimal friction and damping parameters for the vertical linear guide

"""
import mujoco_py, json
import numpy as np
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization as BO
from thesis_galljamov18.python import settings
from thesis_galljamov18.python import tools


sim = None
model = None
viewer = None
data = None

# data for BO: to avoid loading it at every iteration
global AllInitVels
global AllInitPos
global AllExpFlightTrajecsWithMargin
global AllFlightTimes
global AllGroundHeights


# -----------------------------------
# SIMULATION PARAMS
# -----------------------------------

DO_RENDER = False
NUM_FALLING_TIMESTEPS = 500


# -----------------------------------
# LOAD and PROCESS the DATA
# -----------------------------------

global currentExperimentSledgePosTrajec

global timestepsAfterDrop
global timestepsBeforeTD

timestepsAfterDrop = 50
timestepsBeforeTD = 0


def loadAndCalculateDataForBO():
    """
    IMPORTANT: should be called first to load all the experiment data and process it for further work
    Loads experiment data, crops it to the relevant part and extracts important parameters
    """

    global AllInitVels
    global AllInitPos
    global AllFlightTimes
    global AllExpFlightTrajecsWithMargin
    global AllGroundHeights

    AllInitVels = np.zeros(20)
    AllInitPos = np.zeros(20)
    AllFlightTimes = np.zeros(20)
    AllExpFlightTrajecsWithMargin = np.zeros([NUM_FALLING_TIMESTEPS, 20])
    AllGroundHeights = np.zeros(20)

    # load full trajecs
    AllExpVels = loadVelocityData()
    AllExpPos = np.loadtxt(settings.PATH_THESIS_FOLDER + 'python/identification/'
                        'experiment_data/processed/sledge_identification_exp_pos_trajecs_averaged.mat')

    # crop position trajectories and get important params like initial velocities and groundHeights
    for i in range(AllExpPos.shape[1]):
        dropTimestep, preTouchdownTimestep, flightTime = getDropAndTDTimestepsFromPosTrajecFromExp(AllExpPos[:, i])
        AllInitVels[i] = AllExpVels[dropTimestep-1,i]/1000 # from mm to m
        AllInitPos[i] = AllExpPos[dropTimestep-1, i]/1000 # from mm/s to m/s
        AllFlightTimes[i] = flightTime

        # to avoid modeling the influence of the dynamics when sledge was dropped as well as ground contact dynamics
        # the trajectories from experiment have to be cut on both sides (beginning and the end)
        groundHeight = AllExpPos[preTouchdownTimestep, i]
        AllExpFlightTrajecsWithMargin[:, i] = np.append(AllExpPos[dropTimestep:preTouchdownTimestep, i],
                                                        np.ones([(NUM_FALLING_TIMESTEPS - flightTime + timestepsAfterDrop), 1]) * groundHeight)

        AllGroundHeights[i] = groundHeight/1000


def loadVelocityData() -> np.ndarray:
    vels = np.loadtxt(settings.PATH_THESIS_FOLDER + 'python/identification/'
                     'experiment_data/processed/sledge_identification_exp_vel_trajecs_fltrd_averaged.mat')
    return vels


def getDropAndTDTimestepsFromPosTrajecFromExp(posTrajec):
    """ extract falling trajectory and cut this once more at beginning and end to only get the free falling part
    without the influence of the hand contact at the beginning and GRF Plate contact at the end of the flight phase"""

    # trajec is in mm
    posTrajec = np.array(posTrajec)

    # get the mean of starting height and ground height to filter away the noise
    startingHeight = np.mean(posTrajec[0:200])
    groundHeight = np.mean(posTrajec[-500:])

    # - 1 to avoid detecting drop due to noise in position data
    dropTimestep = np.where(posTrajec < (startingHeight-1))[0][0] # was -1 and then starting points were the same when adding 0.0145 to initial position
    # take as timestep one shortly before touchdown (when there are less then 3mm until the ground left
    touchdownTimestep = np.where(posTrajec <= groundHeight+3)[0][0]

    # time of the fall after it is being cut
    remainingFlightTime = touchdownTimestep - dropTimestep

    return dropTimestep + timestepsAfterDrop, touchdownTimestep, remainingFlightTime


def getFallingTrajecFromExpWithMargins(expFallingTrajec:np.array):
    return expFallingTrajec[timestepsAfterDrop:-timestepsBeforeTD]


def getFallingTrajecFromSimWithMargins(verticalPos:np.array, preTouchdownHeight):

    try:
        preTouchdownTimestep = np.where(verticalPos <= preTouchdownHeight)[0][0]
    except:
        preTouchdownTimestep=NUM_FALLING_TIMESTEPS

    fallingTrajec = verticalPos[0:preTouchdownTimestep]
    fallingTrajec = np.append(fallingTrajec, np.ones(NUM_FALLING_TIMESTEPS-fallingTrajec.size)*preTouchdownHeight)

    return fallingTrajec


# -----------------------------------
# RUN EXPERIMENT
# AND COLLECT DATA FOR OPTIMIZATION
# -----------------------------------

def calculateError(expFallingTrajec:np.array=np.array([]), simFallingTrajec:np.array=np.array([]), logErrors = False):

    if len(expFallingTrajec) == 0 or len(simFallingTrajec) == 0:
        raise ValueError("Empty trajectories passed to error calculation!")

    summedSquaredError = int(np.sum(np.square(expFallingTrajec - simFallingTrajec))) / 10
    if(logErrors):
        print("\nSummed squared Error between Trajectories: {}".format(summedSquaredError))

    expTD = np.where(expFallingTrajec == expFallingTrajec[-1])[0][0]
    simTD = np.where(simFallingTrajec == simFallingTrajec[-1])[0][0]

    # additionally penalize the touchdown being different in sim from exp
    touchDownError = np.power(expTD - simTD, 4)

    # touchDownError = abs(expTD-simTD)
    if logErrors:
        print("Difference in TouchdownTimestep between Sim and Exp to the power of 4: {}".format(touchDownError))

    sumError = summedSquaredError + touchDownError

    return -int(sumError)


def runAll40ExperimentsAndGetCost(damping, friction):

    scaledParams = unscaleDampingAndFriction(damping, friction)

    costFromAll40Exp = []

    for i in range(AllExpFlightTrajecsWithMargin.shape[1]):
        costFromAll40Exp.append(runExperimentAndGetCost(initPos=AllInitPos[i], initVel=AllInitVels[i], expFallingTrajec=AllExpFlightTrajecsWithMargin[:,i],
                                                        groundHeight=AllGroundHeights[i], damping=scaledParams[0], friction=scaledParams[1], plotResults=False, doRender=DO_RENDER))

    return np.sum(costFromAll40Exp)



def runExperimentAndGetCost(initPos=-333, initVel=333, expFallingTrajec=None, groundHeight = 0.0879,
                            damping=-1, friction=-1, doRender = DO_RENDER, plotResults=False, scaledValues=False):

    global currentExperimentSledgePosTrajec

    if(scaledValues):
        damping, friction = unscaleDampingAndFriction(damping, friction)

    sim.reset()
    vertical_pos = []
    grf_trajec = []

    step = 0
    countTimeSteps = NUM_FALLING_TIMESTEPS;

    if (damping >= 0):
        model.dof_damping[0] = damping
    if (friction >= 0):
        model.dof_frictionloss[0] = friction
    if initPos != -333:
        data.qpos[0] = initPos # - groundHeight # + 0.00145
    if initVel != 333:
        data.qvel[0] = initVel

    for _ in range(countTimeSteps):

        sim.step()

        if doRender:
            viewer.render()

        step += 1

        # get vertical pos and GRFs
        vertical_pos.append(data.qpos[0])
        contactForces = sim.data.cfrc_ext
        grf_forces = contactForces[2]
        grf_z_force = grf_forces[5]
        grf_trajec.append(grf_z_force)

    # scale and cut sim trajectory
    vertical_pos = np.array(vertical_pos)
    vertical_pos = vertical_pos*1000 # from m to mm

    vertical_pos = getFallingTrajecFromSimWithMargins(vertical_pos, expFallingTrajec[-1])
    currentExperimentSledgePosTrajec = vertical_pos

    # calculate error
    error = calculateError(expFallingTrajec, vertical_pos)

    compareSimWithExp = True

    if(compareSimWithExp):
        # Plot simulated and experiment-trajectory in one plot
        allTrajects = np.zeros([NUM_FALLING_TIMESTEPS,2])
        allTrajects[:,0] = vertical_pos
        # allTrajects[:,1] = goalTrajec


    if plotResults:
        print('Cost: {}'.format(error))

    return error


def log_cost_to_history(sumError):
    # save error for later validation
    costHistory = np.loadtxt('costHistory.txt')
    countEntries = costHistory.size
    costHistory = np.append(costHistory, sumError)
    if countEntries < costHistory.size:
        np.savetxt('costHistory.txt', costHistory, delimiter=',', fmt='%f')


# -----------------------------------
# RUN OPTIMIZATION
# -----------------------------------

def unscaleDampingAndFriction(damping, friction):
    """To speed up optimization with BO it is better to let it output values between -1 and 1,
    which we then have to scale to the ranges we expect the best parameters to come from"""
    min_damp, max_damp, min_fric, max_fric = [0, 2, 0, 2]
    damping = tools.linearScale(damping, -1, 1, min_damp, max_damp)
    friction = tools.linearScale(friction, -1, 1, min_fric, max_fric)
    return damping, friction


def optimizeSledgeParams():
    """ Run Bayesian Optimization with normalized inputs to find best parameters
        to replicate the free falling trajetories of the sledge
    """
    optimization = BO(runAll40ExperimentsAndGetCost, {'damping': (-1, 1), 'friction': (-1, 1)})

    # initialize optimization with so far best found parameters to improve the results
    optimization.initialize({
        'target': [-1198.0, -1300.0, -1197.0],
        'damping': [-0.6826810335539988, -0.8658977050860612, -0.6818051907866487],
        'friction': [0.5014436638411311, 0.7335613457613679, 0.49986569844586715]
    })

    optimization.maximize(n_iter=55, acq='ucb')

    print(optimization.res)
    # uncomment to save the results for monitoring
    # with open('BO_sledge_params/BO_all40_45iters_0-8_0-1.txt', 'w') as file:
    #     file.write(json.dumps(optimization.res))  # use `json.loads` to do the reverse

    results = np.array(optimization.res['all']['values'])
    results = results[results > (optimization.res['max']['max_val'] * 10)]

    bestParams = optimization.res['max']['max_params']
    unscaledDamping, unscaledFriction = bestParams['damping'], bestParams['friction']
    bestDamping, bestFriction = unscaleDampingAndFriction(unscaledDamping, unscaledFriction)
    print("Best Cost of {} was achieved with damping = {} and friction = {}"
          .format(max(results), bestDamping, bestFriction))
    tools.plot(results, title="Bayesian Optimization convergence", ylabel="Cost []", xlabel="Optimization step []")


def optimizeOptimization():
    """
    Run Bayesian Optimization several times with different configurations (f.ex. different acquisition functions).
        - save the best results from each optimization iteration and use then to initialize the next optimization
    """
    initDict = {'target': [], 'damping': [], 'friction': []}
    acquisitionFunctions = ['ei', 'ucb', 'ei', 'ucb', 'ei', 'ucb']

    for i in range(6):
        optimization = BO(runAll40ExperimentsAndGetCost, {'damping': (0, 1), 'friction': (0, 1)})
        try:
            optimization.initialize(initDict)
            optimization.maximize(n_iter=50, acq=acquisitionFunctions[i])
            res = optimization.res['max']

            if res['max_val'] not in initDict['target']:
                initDict['target'].append(res['max_val'])
                maxParams = res['max_params']
                initDict['damping'].append(maxParams['damping'])
                initDict['friction'].append(maxParams['friction'])
                print(initDict)

                with open('BO_sledge_params/6xBO_all40_60iters_0-8_0-4.txt', 'w') as file:
                    file.write(json.dumps(initDict))  # use `json.loads` to do the reverse

        except Exception as ex:
            print(ex)
            continue


# -----------------------------------
# TEST OPTIMIZATION RESULTS
# -----------------------------------

def testFoundParamsWithAll4Heights(damping, friction, valuesAreNormalized):
    """
    Perform one drop experiment for each height and compare the results from simulation to experiment results.
    Use the following parameters for the comparison.
    :param damping: vertical damping in the sledge's slide joint
    :param friction: frictionloss of the sledge's slide joint
    :param valuesAreNormalized: True if damping and friction are normalized to [-1:1]
    :return:
    """
    if valuesAreNormalized:
        damping, friction = unscaleDampingAndFriction(damping, friction)

    cost = 0

    refereceTrajecs = np.zeros([AllExpFlightTrajecsWithMargin.shape[0], 4])
    simTrajecs = np.zeros_like(refereceTrajecs)
    groundHeights = np.zeros(simTrajecs.shape[1])

    for i in range(4):
        expNrToTest = i * 5 + 1

        cost += runExperimentAndGetCost(initPos=AllInitPos[expNrToTest], initVel=AllInitVels[expNrToTest],
                                        expFallingTrajec=AllExpFlightTrajecsWithMargin[:, expNrToTest],
                                        groundHeight=AllGroundHeights[expNrToTest], doRender=DO_RENDER, plotResults=True,
                                        damping=damping, friction=friction, scaledValues=False)

        refereceTrajecs[:,i] = AllExpFlightTrajecsWithMargin[:, expNrToTest]
        simTrajecs[:,i] = currentExperimentSledgePosTrajec
        groundHeights[i] = AllGroundHeights[expNrToTest]
        # print("current ground height = {}".format(groundHeights[i]))

    print("Cummulative costs for all 4 heights are: {}".format(cost))

    correctedSimTrajecs = simTrajecs + groundHeights[0]
    tools.overlayingPlots([[refereceTrajecs[:,3]/1000,refereceTrajecs[:,2]/1000,refereceTrajecs[:,1]/1000,refereceTrajecs[:,0]/1000,
                            correctedSimTrajecs[:,3]/1000,correctedSimTrajecs[:,2]/1000,correctedSimTrajecs[:,1]/1000,correctedSimTrajecs[:,0]/1000]],
                          "Vertical Sledge Position [m]", legend="Experiment," * 4 + "Simuation," * 3 + "Simulation",
                          thesis=True, sampleRate=500, title=None if True else "Optimized Sledge Parameters Test\ndamping = {} Ns/m, friction = {} N".format(damping, friction), )


# -----------------------------------
# INIT SIM, MODEL and DATA
# -----------------------------------

model = mujoco_py.load_model_from_path(settings.PATH_THESIS_FOLDER +
            "mujoco/param_identification/sledge_param_identification.xml")

sim = mujoco_py.MjSim(model)
data = sim.data

model.opt.timestep=0.002

if DO_RENDER:
    viewer = mujoco_py.MjViewer(sim)
    viewer.render()


# -----------------------------------
# MAIN function
# -----------------------------------

def main():

    # True if you want to compare the falling trajectories of a parameter set with the experiment trajectories
    TEST_PARAMS = True

    # useful to slow down simulation before it starts or change point of view
    pauseViewerAtFirstStep = False

    # pause sim to be able to slow it down or change camera perspective
    if DO_RENDER and pauseViewerAtFirstStep:
        viewer._paused = True
        pauseViewerAtFirstStep = False

    loadAndCalculateDataForBO()

    if TEST_PARAMS:
        # {'max_val': -1197.0, 'max_params': {'damping': -0.6818051907866487, 'friction': 0.49986569844586715}}
        test_damping_normed = -0.6818051907866487
        test_friction_normed = 0.49986569844586715
        testFoundParamsWithAll4Heights(test_damping_normed, test_friction_normed, True)
        exit(33)

    # uncomment to run optimizations again
    # optimizeOptimization()
    # optimizeSledgeParams()


if __name__ == '__main__':
    main()



