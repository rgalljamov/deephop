import numpy as np
from matplotlib import pyplot as plt
import mujoco_py, json
from bayes_opt import BayesianOptimization as BO
from thesis_galljamov18.python import tools, settings


sim = None
model = None
viewer = None
data = None

global allHeights

# data for BO: to avoid loading it at every iteration
global AllExpGroundForceTrajects
global AllExpPosTrajecs
global AllExpCroppedPosTrajecs
global AllInitPositions
global AllInitVelocities
# derived from GRF EXP data
global AllExpBouncingTimes


# -----------------------------------
# SIMULATION PARAMS
# -----------------------------------

DO_RENDER = True
PLOT = False
EPISODE_TIMESTEPS = 600

waitTimestepsAfterTouchdown = 1
timestepsAfterTouchdown = 300


# -----------------------------------
# LOAD and PROCESS the DATA
# -----------------------------------

def loadAndCalculateDataForBO(test = False):
    """
    IMPORTANT: should be called first to load all the experiment data and process it for further work
    Loads experiment data, crops it to the relevant part and extracts important parameters
    """

    global AllExpGroundForceTrajects
    global AllInitPositions
    global AllInitVelocities
    global AllExpBouncingTimes
    global AllExpPosTrajecs
    global AllExpCroppedPosTrajecs

    AllExpGroundForceTrajects = correctOffset(loadGroundForceData())
    AllExpPosTrajecs = np.loadtxt(settings.PATH_THESIS_FOLDER + 'python/identification/'
                               'experiment_data/processed/sledge_identification_exp_pos_trajecs_averaged.mat')

    AllExpVelTrajecs = loadVelocityData()

    if test:
        tools.plot(AllExpVelTrajecs, title="All Position Trajecs")
        tools.severalPlots([AllExpPosTrajecs, AllExpVelTrajecs], yLabels=["pos", "vel"], title="experiment kinematics")

    totalNumOfExperiments = AllExpGroundForceTrajects.shape[1]

    AllExpTouchdownTimestepsFromPos = np.zeros([3, totalNumOfExperiments])
    AllExpBouncingTimes = np.zeros([2, totalNumOfExperiments])
    AllInitPositions = np.zeros(totalNumOfExperiments)
    AllInitVelocities = np.zeros(totalNumOfExperiments)
    AllExpCroppedPosTrajecs = np.zeros([EPISODE_TIMESTEPS, totalNumOfExperiments])

    timestepsBeforeTouchdown = 25

    for exp in range(totalNumOfExperiments):
        # get touchdownTimesteps from exp POS data
        touchdownTimesteps = findTouchdownTimestepsFromPosTrajec(AllExpPosTrajecs[:, exp])

        # get all bouncing times from exp
        bouncingTimes = getBounceTimes(touchdownTimesteps=touchdownTimesteps)
        AllExpBouncingTimes[:, exp] = bouncingTimes

        # get initial pos and vel (!CAUTION! GRF is sampled with 1kHz and pos with 0.5kHz
        initTimestep = touchdownTimesteps[0] - timestepsBeforeTouchdown
        initPos = AllExpPosTrajecs[(initTimestep), exp]

        groundHeight = np.mean(AllExpPosTrajecs[-500:, exp])
        AllInitPositions[exp] = initPos - groundHeight + 3.5  # correct offset due to GRF plate's height
        AllInitVelocities[exp] = AllExpVelTrajecs[initTimestep, exp]
        AllExpCroppedPosTrajecs[:, exp] = AllExpPosTrajecs[initTimestep:initTimestep+EPISODE_TIMESTEPS, exp] - groundHeight
        if exp == (totalNumOfExperiments-1):
            print("All kinematic and kinetic data successfully loaded!")


def loadGroundForceData() -> np.ndarray:
    """ :return: GRF data from Experiment as a matrix of shape [num_timesteps, num_experiments]"""

    AllZGRFData = np.zeros([4900, 20])

    for numExp in range(1,21):

        filename = settings.PATH_THESIS_FOLDER + 'python/identification/experiment_data/raw/force-data' \
                   '/frc {:03d}.txt'.format(numExp)

        # the first 19 rows are header-info - the next 1000 are just zero
        data = np.loadtxt(filename, delimiter='\t', skiprows=19)

        AllZGRFData[:, numExp-1] = data[0:4900,3]

    # tools.plot(AllZGRFData[:,:], title="ALL GRF DATA from NEW Experiment")
    # exit(33)

    return AllZGRFData


def loadVelocityData() -> np.ndarray:
    vel = np.loadtxt(settings.PATH_THESIS_FOLDER + 'python/identification/'
                     'experiment_data/processed/sledge_identification_exp_vel_trajecs_fltrd_averaged.mat')
    return vel


def correctOffset(AllZGRFData: np.ndarray) -> np.ndarray:
    """GRF data was oscillating around an offset... we have to calculate and correct it"""

    # calculate offset by averaging over first 500 timesteps
    firstTimesteps = AllZGRFData[:500,:]
    offsets = np.mean(firstTimesteps,0)

    correctedZGRFData = np.zeros(AllZGRFData.shape)
    for i in range(offsets.size):
        correctedZGRFData[:,i] = AllZGRFData[:,i] - offsets[i]

    return correctedZGRFData


def getRunningAverage(data:np.array, steps_to_average_over:int=10) -> np.array:
    runAverage = np.zeros(data.size-steps_to_average_over*2)

    for i in range(steps_to_average_over, data.size-steps_to_average_over):
        runAverage[i-steps_to_average_over] = np.mean(data[(i-steps_to_average_over):i])

    return runAverage


def findTouchdownTimestepsFromPosTrajec(posTraject:np.array, isSimData=False):
    """ returns a list of timesteps where Touchdown was detected and uses position data for that"""

    if posTraject.ndim > 1:
        raise ValueError("Position Trajectory with ndim == 1 expected!")


    groundHeight = 0 if isSimData else np.mean(posTraject[-500:])

    touchdowns = []

    for i in range(1, posTraject.size-1):
        if posTraject[i+1] < groundHeight and posTraject[i] > groundHeight and (posTraject[i] - posTraject[i+1]) > 0.0005:
            touchdowns.append(i+1)
            if len(touchdowns) == 3:
                break

    if not isSimData and not len(touchdowns) == 3:
        tools.plot(posTraject, title="PositionTrajec with less then 3 touchdowns detected!")
        raise AssertionError("Less then 3 Touchdowns detected in position trajectory!")

    if isSimData and False:
        touchdowns = np.array(touchdowns)
        if touchdowns.size > 1:
            print("{} TDs found in sim trajectory with following bouncingTimes {}".format(touchdowns.size, touchdowns))
        tools.plot(posTraject, title="Position Trajectory to get TD timesteps from", doBlock=False)
        plt.scatter(touchdowns, np.ones([len(touchdowns), 1])*groundHeight, c='r', s=10 ** 2)
        plt.show()

    return touchdowns


def findTouchdownTimestepsFromEXPTraject(forceTraject:np.array, listToAttachTDTimestepsTo:np.array=None):
    """ returns a list of timesteps where Touchdown was detected OR
        OR attaches the timesteps to listToAttachTDTimestepsTo and return this list."""
    raise NotImplementedError("Use findTouchdownTimestepsFromPosTrajec(posTraject:np.array) instead!")

    if forceTraject.ndim > 1:
        raise ValueError("Force Trajectory with ndim == 1 expected!")

    threshold_heigh = 20
    threshold_low = 3
    slope = 12
    touchdownTimesteps = []
    minTimeBetweenTouchdowns = 20

    for i in range(1, forceTraject.size-3):
        # avoid detecting touchdowns too close to each other
        if len(touchdownTimesteps) > 0:
            distToLastTD = 1000 # i - touchdownTimesteps[-1]
        else:
            distToLastTD = 1000

        if(distToLastTD > minTimeBetweenTouchdowns and (forceTraject[i] > threshold_heigh or forceTraject[i+2] > threshold_heigh) and forceTraject[i-1] < threshold_low):
            touchdownTimesteps.append(i+1)

    if len(touchdownTimesteps) < 3:
        raise ValueError("Less then three touchdownTimesteps!")

    # workaround: avoid detecting takeoff because of strong oscillations by ignoring timesteps too close to each other
    min_timesteps_between_touchdowns = 30
    realTouchdownTimesteps = []
    realTouchdownTimesteps.append(touchdownTimesteps[0])
    for i in range(len(touchdownTimesteps)-1):
        if touchdownTimesteps[i+1] - touchdownTimesteps[i] > min_timesteps_between_touchdowns:
            realTouchdownTimesteps.append(touchdownTimesteps[i+1])
            if len(realTouchdownTimesteps) == 3:
                break

    return np.array(realTouchdownTimesteps)


def testTouchDownDetection(numOfPlots=40):
    GRFdata = correctOffset(loadGroundForceData())
    allTouchDownTimesteps = []
    bouncingTime = []

    for i in range(min(numOfPlots, GRFdata.shape[1])):
        timestepsOfIthTraject = findTouchdownTimestepsFromEXPTraject(GRFdata[:,i])
        bouncingTime.append(timestepsOfIthTraject[1] - timestepsOfIthTraject[0])

        numSteps = len(timestepsOfIthTraject)
        if (numSteps != 3):
            print("{} TDs detected in data Nr. {}".format(numSteps, i))

        for timestep in timestepsOfIthTraject:
            allTouchDownTimesteps.append(timestep)

    print("Num of touchdownTimesteps is {}".format(len(allTouchDownTimesteps)))

    #tools.plot(bouncingTime, "Timesteps between first and second touchdown", "Time between two touchdowns", "Experiment Nr.")
    return np.array(allTouchDownTimesteps)


def getBounceTimes(posTrajec: np.ndarray=None, touchdownTimesteps=[], isSimTrajec = False):
    """ :returns [firstBounceTime, secondBounceTime]: the count of timesteps between first and second, second and third touchdown
        : params provide either POS trajec or touchdownTimesteps """

    if len(touchdownTimesteps) == 0 and posTrajec is not None:
        touchdownTimesteps = findTouchdownTimestepsFromPosTrajec(posTrajec, isSimTrajec)

    if not isSimTrajec and not len(touchdownTimesteps) == 3:
        if posTrajec is not None: tools.plot(posTrajec, title="ERROR: Trajectory with only {} touchdowns. 3 expected.".format(len(touchdownTimesteps)))
        raise AssertionError("not enough touchdown timesteps detected in experiment trajectory. "
                             "Detected {} instead of 3.".format(touchdownTimesteps.size))

    numTDs = len(touchdownTimesteps)
    if numTDs < 2:
        return []
    elif numTDs == 2:
        return [touchdownTimesteps[1]-touchdownTimesteps[0]]

    bouncingTimes = [touchdownTimesteps[1] - touchdownTimesteps[0], touchdownTimesteps[2] - touchdownTimesteps[1]]
    return np.asarray(bouncingTimes)


def getFallingHeights(plot=False)->np.ndarray:
    raise NotImplementedError("load other data for falling heights")

    # Load Experiment Data in an numpy array
    matData = np.loadtxt('heightAllmark1.mat')
    print(matData.shape)
    if plot:
        plt.plot(matData)
        plt.title('Falling Trajectories from EXPERIMENT')
        plt.show()
    return matData[1,:]


def getAllTrajectsOfInterest(AllZGRFData: np.ndarray=None) -> np.ndarray:

    if AllZGRFData is None:
        AllZGRFData = loadGroundForceData()
        AllZGRFData = correctOffset(AllZGRFData)

    TrajectsOfInterest = np.zeros([timestepsAfterTouchdown, AllZGRFData.shape[1]])

    numberOfExperiments = AllZGRFData.shape[1]

    for i in range(numberOfExperiments):
        # find first timestep where GRF was higher 5N
        touchdownTimestep = np.where(AllZGRFData[:,i]>5)[0][0]

        TrajectsOfInterest[:,i] = AllZGRFData[
                                  (touchdownTimestep+waitTimestepsAfterTouchdown):
                                  (touchdownTimestep+waitTimestepsAfterTouchdown+timestepsAfterTouchdown)
                                  , i]

    return TrajectsOfInterest


def fromExperimentGetTrajectOfInterest(grfZTraject: np.ndarray) -> np.ndarray:
    touchdownTimestep = np.where(grfZTraject[:]>5)[0][0]
    trajectOfInterest = grfZTraject[touchdownTimestep: touchdownTimestep + timestepsAfterTouchdown]
    return trajectOfInterest


# -----------------------------------
# RUN EXPERIMENT
# AND COLLECT DATA FOR OPTIMIZATION
# -----------------------------------

def runExperimentAndGetCost(expCroppedPosTrajec=[], init_pos=0.7, init_vel=0, bouncingTimesExp=[],
                            solref1=-1, solref2=-1, solimp1=-1, solimp2=-1, solimp3=-1,
                            doRender = DO_RENDER, plotResults=False, areParamsNormed=True):

    sim.reset()

    grf_z_trajectory = []
    vertical_vel = []
    vertical_pos = []

    numTimesteps = EPISODE_TIMESTEPS

    # set sim params
    if solimp1 != -1 and solref1 != -1:

        if areParamsNormed:
            solref1, solref2, solimp1, solimp2, solimp3 = unscaleGroundContactParams(solref1, solref2, solimp1, solimp2, solimp3)

        model.geom_solref[0] = [solref1, solref2]
        model.geom_solimp[0] = [solimp1, solimp2, solimp3]

    # set initial conditions (pos and vel) - convert mm to m
    data.qpos[0] = init_pos/1000
    data.qvel[0] = init_vel/1000

    step = 0

    # run sim and collect data
    for _ in range(numTimesteps):
        sim.step()

        if doRender:
            viewer.render()

        step += 1

        # get GRFs
        contactForces = sim.data.cfrc_ext
        grf_forces = contactForces[2]
        grf_z_force = grf_forces[5]
        grf_z_trajectory.append(grf_z_force)

        vertical_vel.append(data.qvel[0])
        vertical_pos.append(data.qpos[0])

    grf_z_trajectory = np.array(grf_z_trajectory)
    vertical_vel = np.array(vertical_vel)
    vertical_pos = np.array(vertical_pos)

    # get bouncing times from sim
    bouncingTimesSim = getBounceTimes(vertical_pos, isSimTrajec=True)

    # calculate error
    sumError = calculateCostByComparingTrajecs(expCroppedPosTrajec, vertical_pos*1000)

    # Plot simulated and experiment-trajectory in one plot
    if plotResults:
        tools.overlayingPlots([[expCroppedPosTrajec / 1000 + 0.001, vertical_pos]],
                              labels_comma_separated_string="Vertical Sledge Positions [m]",
                              legend="Experiment, Simulation", thesis=True, sampleRate=500,
                              title=None if True else "Ground Contact Model Identifiaction")
        print('Cost: {}'.format(sumError))

    return sumError


def runAll40ExperimentsAndGetCost(solref1=-1, solref2=-1, solimp1=-1, solimp2=-1, solimp3=-1, doRender = DO_RENDER, plotResults=False, shortTest=False):
    """ run loadAndCalculateDataForBO() before this function"""

    if AllExpGroundForceTrajects is None or len(AllExpGroundForceTrajects) == 0:
        raise ValueError("Please run 'loadAndCalculateDataForBO()' at the beginning of your program!")

    if AllExpGroundForceTrajects is None:
        raise ValueError("Call loadAndCalculateDataForBO() before executing runAll40ExperimentsAndGetCost()!")

    allCosts = []

    for i in range(AllExpGroundForceTrajects.shape[1]) if not shortTest else np.arange(0,4)*5+1:
        allCosts.append(runExperimentAndGetCost(AllExpCroppedPosTrajecs[:, i], AllInitPositions[i], AllInitVelocities[i], AllExpBouncingTimes[:, i],
                                                solref1, solref2, solimp1, solimp2, solimp3, doRender, plotResults=plotResults, areParamsNormed=not shortTest))

    if shortTest: print("Cumulative Costs for 4 Experiments were {}".format(np.sum(allCosts)))

    # print("Cost after 40 Experiments is {}".format(np.sum(allCosts)))
    return np.sum(allCosts)


def calculateCost(bouncingTimesSim, bouncingTimesExp):
    """
    The first cost calculation was based on calculating the difference in the bouncing times.
    When the sledge hit the ground, it rebounded up to 4 times. The time it was in the air between two rebounds
    we defined as the bouncing time.

    Later we found out, that comparing the vertical sledge position trajectories after TD leads to better results.
    """
    raise NotImplementedError("Use calculateCostByComparingTrajecs() instead")

    # if less or more bounces were present in the simulation:
    additionalPenalty = 500

    # if only one touchdown was detected, penalize params: max bouncing time is 154 for first and 65 for second timesteps
    noTakeoffPenalty = ((154+65+additionalPenalty/5)**2)

    countBounces = len(bouncingTimesSim)

    if countBounces == 0:
        # print("No takeoffs were detected within simulation.")
        return -noTakeoffPenalty

    # for two touchdowns penalize the third touchdown not being existant
    if(countBounces == 1):
        return -((65)**2 + (bouncingTimesSim[0]-bouncingTimesExp[0]+additionalPenalty/2)**2)

    # at least three touchdowns / two bounces

    cost = 0

    # cost for two bounces
    cost -= (bouncingTimesSim[0] - bouncingTimesExp[0])**2 + (bouncingTimesSim[1]-bouncingTimesExp[1])**2

    # increase cost if more then four bounces were present in the simulation, as we had four
    if countBounces > 4:
        for i in range(4, countBounces):
            cost -= abs((10*bouncingTimesSim[i])**2)

    # print("Takeoff times from EXP: {}\nTakeoff times from SIM: {}".format(bouncingTimesExp, bouncingTimesSim))

    return cost


def calculateCostByComparingTrajecs(expPosTrajec, simPosTrajec):
    """
    Calculate the cost as the summed square error for every timestep
    between the experiment and sim vertical sledge position trajectories
    """
    return - int(np.sum(np.square(expPosTrajec - simPosTrajec))) / 10


def testCostCalculation():
    bouncingTimesExp = [[150, 65], [150, 65], [150, 65], [150, 65], [150, 65], [150, 65]]
    bouncingTimesSim = [[145, 60], [120, 40], [100, 10], [80], [145, 60, 30, 25, 15, 5, 1], []]
    cost = []
    for i in range(len(bouncingTimesExp)):
        cost.append(calculateCost(bouncingTimesSim[i], bouncingTimesExp[i]))
    tools.plot(cost, title="Costs for the provided bouncingTimes")
    exit()


# -----------------------------------
# RUN OPTIMIZATION
# -----------------------------------

def unscaleGroundContactParams(solref1, solref2, solimp1, solimp2, solimp3):
    """
    To speed up optimization with BO it is better to let it output values between -1 and 1,
    which we then have to scale to the ranges we expect the best parameters to come from
    """

    # default: solref : "0.02 1"  solimp : "0.9 0.95 0.001"
    inputs = [solref1, solref2, solimp1, solimp2, solimp3]
    ranges = [[0, 0.4], [0, 1], [0, 40], [0, 1], [0, 0.4]]

    scaledInputs = []

    for i in range(len(inputs)):
        curRange = ranges[i]
        newRange = curRange[1] - curRange[0]
        scaledInputs.append(
            inputs[i] * newRange + curRange[0])

    return scaledInputs


def optimizeSolverParams():
    """ Run Bayesian Optimization with normalized inputs to find best parameters
            to replicate the vertical sledge position trajetories after the touchdown
        """

    optimization = BO(runAll40ExperimentsAndGetCost, {'solref1': (0, 1), 'solref2': (0, 1), 'solimp1':(0, 1), 'solimp2':(0,1), 'solimp3':(0,1)})

    # initialize optimization with so far best found parameters to improve the results
    optimization.initialize({
        'target': [-30668.7, -38917.8],
        'solref1': [ 0.10667760273187854,0.7917295903960275],
        'solref2': [0,0],
        'solimp1': [0.6323304436894982,0.9999981685561232],
        'solimp2': [0.0, 0.0],
        'solimp3': [0.9029115238081952, 1.0]
    })


    print("BO settings: n_iter = 250 and acq=UCB")
    optimization.maximize(n_iter=75, acq='ei')

    print("Optimization Results:\n" + optimization.res)
    results = np.array(optimization.res['all']['values'])
    results = results[results > optimization.res['max']['max_val']*100]

    bestParams = optimization.res['max']['max_params']
    solref1, solref2, solimp1, solimp2, solimp3 = bestParams['solref1'], bestParams['solref2'], bestParams['solimp1'], bestParams['solimp2'], bestParams['solimp3']
    solref1, solref2, solimp1, solimp2, solimp3 = unscaleGroundContactParams(solref1, solref2, solimp1, solimp2, solimp3)
    print("Best Cost of {} was achieved with {}, {}, {}, {}, {} params".format(max(results), solref1, solref2,solimp1, solimp2, solimp3))

    tools.plot(results, title="Optimization Results from setting: {}".format(settings.replace(" ", "_")))

    # uncomment to save the results for monitoring
    # with open('BO_sledge_params/{}.txt'.format(settings), 'w') as file:
    #     file.write(json.dumps(optimization.res['max']))  # use `json.loads` to do the reverse


# -----------------------------------
# TEST OPTIMIZATION RESULTS
# -----------------------------------

def testFoundParams(params=list(np.ones(5)*(-1)), scaled=False, testParamsOfMuJoCoModel=False):
    """:parameter params: a list of scaled or unscaled parameters for solref and solimp (len(params) = 5)
    This list can be empty, if :param:testParamsOfMuJoCoModel is set to True.
    In this case the simulation is just run with the values specified in the xml file!"""
    solref1, solref2, solimp1, solimp2, solimp3 = params

    if not testParamsOfMuJoCoModel:
        solref1, solref2, solimp1, solimp2, solimp3 = unscaleGroundContactParams(solref1, solref2, solimp1, solimp2, solimp3)
        print("Tested Parameters after scaling to the right range: {}, {}, {}, {}, {} params".format(solref1, solref2,solimp1, solimp2, solimp3))

    runAll40ExperimentsAndGetCost(solref1, solref2, solimp1, solimp2, solimp3, DO_RENDER, True, shortTest=True)


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

    loadAndCalculateDataForBO()
    bestParamsDict = {'solref1': 0.2743789584688579, 'solref2': 0.0, 'solimp1': 0.6323304436894982, 'solimp2': 0.0, 'solimp3': 0.9029115238081952}
    bestParamsListUnscaled = list(bestParamsDict.values())
    bestParamsList = unscaleGroundContactParams(bestParamsListUnscaled[0], bestParamsListUnscaled[1], bestParamsListUnscaled[2], bestParamsListUnscaled[3], bestParamsListUnscaled[4])
    print("Best Parameters:" + str(bestParamsList))

    if TEST_PARAMS:
        testFoundParams(testParamsOfMuJoCoModel=True)
        exit(33)

    optimizeSolverParams()



if __name__ == '__main__':
    main()