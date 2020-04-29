import numpy as np
from thesis_galljamov18.python import tools, settings
# enables additional plots
THESIS = False

# -------------------------
# helper functions
# -------------------------

def rad(degrees):
    return np.multiply(degrees,np.pi)/180

def deg(angles_in_rad):
    return np.divide(angles_in_rad,np.pi)*180

def loadData():
    # load the data
    # subject info: weight array([[643.3]]), array([[1.79]]), array([[33]]
    path = settings.PATH_THESIS_FOLDER + 'python/training/human_hopping_data/'
    filename = 'comcopfrc.mat'
    import scipy.io as sio
    data = sio.loadmat(path + filename)
    return data


# -------------------------
# MAIN function
# -------------------------

def getReferenceTrajecsForRobot(plotData = False, groundDropTest=False, perturbationHeight=0,
                                onlyPerturbedHops = False):
    """ :returns reference data for training including vertical sledge and joint kinematics
        data order is: sledge position, sledge velocity, hip angle, hip ang vel, knee angle, knee ang vel"""

    global THESIS

    data = loadData()

    # get legLengths of all unperturbed hops
    # ---------------------------------------------------

    legLen = data['lenLegAllAll']
    legLenAllPerturbations = legLen[0,:] if not onlyPerturbedHops else legLen[0,5]

    # get all trials in one list
    legLenAllTrialsList = []

    # get all unperturbed hops in one list
    legLenAllHops240HzList = []
    unperturbedHopsIndices = np.arange(0, 3) if not onlyPerturbedHops else [3]

    if not onlyPerturbedHops:
        for perturbation in legLenAllPerturbations:
            for trial in range(8):
                legLenAllTrialsList.append(perturbation[trial,:])

        for trial in legLenAllTrialsList:
            for index in unperturbedHopsIndices:
                legLenAllHops240HzList.append(trial[index][0, :])
    else:
        # get only the perturbed hop in each of the trials
        for trial in legLenAllPerturbations:
            legLenAllHops240HzList.append(trial[3][0])


    # get GRFs of all unperturbed hops
    # ---------------------------------------------------

    grfs = data['grfTotAllAll']
    grfsAllPerturbations = grfs[0,:] if not onlyPerturbedHops else grfs[0,5]

    # get all trials in one list
    grfsAllTrialsList = []

    # get all unperturbed hops in one list
    grfsAllHops240HzList = []

    if not onlyPerturbedHops:
        for perturbation in grfsAllPerturbations:
            for trial in range(8):
                grfsAllTrialsList.append(perturbation[trial,:])

        for trial in grfsAllTrialsList:
            for index in unperturbedHopsIndices:
                grfsAllHops240HzList.append(trial[index][2,:])
    else:
        # get only the perturbed hop in each of the trials
        for trial in grfsAllPerturbations:
            grfsAllHops240HzList.append(trial[3][2])

    if THESIS:
        tools.overlayingPlots([legLenAllHops240HzList], "Leg Lengths [m]",
                              title="Leg lengths of all {} hops".format(len(legLenAllHops240HzList)),
                              xLabels=["Time [1/240s]"])
        # tools.log("Mean Takeoff velocity is: {}".format(np.mean([hop[-1] for hop in legLenAllHops240HzList])))

    # set to True to plot the human LengthForce Relationship
    PLOT_HUMAN_LEN_FORCE_CURVE = False
    if PLOT_HUMAN_LEN_FORCE_CURVE:
        # tools.overlayingPlots([grfsAllHops240HzList], "leglens")
        tools.plotForceLengthCurve(grfsAllHops240HzList, legLenAllHops240HzList, 240, False)
        exit(33)

    if THESIS:
        tools.plotMeanStdOverSeveralHops(grfsAllHops240HzList)
        tools.overlayingPlots([grfsAllHops240HzList], "Leg Lengths [m]",
                              title="Vertical GRFs of all {} hops".format(len(legLenAllHops240HzList)),
                              sampleRate=240)


    # --------------------------------------------------------
    # SCALE DOWN human leg length to the robot dimensions
    # CLEAN THE DATA before that
    # --------------------------------------------------------

    # get human segment length from halving the leg length, which is estimated by the mean of TD leg lens of all hops
    # but first clean the data (not required for groundDropExperiment)
    if not onlyPerturbedHops:
        allTouchdownLegLengths = [hop[0] for hop in legLenAllHops240HzList]
        meanTDLegLength = np.mean(allTouchdownLegLengths)
        stdTDLegLength = np.std(allTouchdownLegLengths)
        allLiftOffLegLengths = [hop[-1] for hop in legLenAllHops240HzList]
        meanTOLegLength = np.mean(allLiftOffLegLengths)
        stdTOLegLength = np.std(allLiftOffLegLengths)

        # delete leg length trajectories where TD and LO leg length is deviating too much from the mean
        # ---------------------------------------------------------------------------------------------
        legLenAllHops240HzList = [hop for hop in legLenAllHops240HzList
                                      if (abs(hop[0]-meanTDLegLength) < stdTDLegLength)
                                      and (abs(hop[-1]-meanTOLegLength) < stdTOLegLength)]

        # remove ref data with stance phase durations deviating too much from the mean duration
        stancePhaseDurationsList240Hz = np.array([np.size(hop) for hop in legLenAllHops240HzList])
        meanStancePhaseDuration = np.mean(stancePhaseDurationsList240Hz)
        # tools.log("Mean stance phase duration is: "+str(meanStancePhaseDuration))
        stdStancePhaseDuration = np.std(stancePhaseDurationsList240Hz)
        legLenAllHops240HzList = [hop for hop in legLenAllHops240HzList
                                  if abs(np.size(hop) - meanStancePhaseDuration) < stdStancePhaseDuration]


    if THESIS:
        tools.overlayingPlots([legLenAllHops240HzList], "Leg Lengths [m]",
                              title="Leg lengths of all {} hops after cleanup\n(Removed hops with TD and LO leg lengths differing from the mean more then std)".format(len(legLenAllHops240HzList)),
                              sampleRate=240)

    # SCALE LENGHTS TO ROBOT
    allCleanedTouchdownLegLengths = [hop[0] for hop in legLenAllHops240HzList]
    allCleanedLiftoffLegLengths = [hop[-1] for hop in legLenAllHops240HzList]

    legLengthHuman = np.max(allCleanedTouchdownLegLengths)#+allCleanedLiftoffLegLengths)
    # print("Human Leg Length - estimated from max cleaned TD leg lens: {}".format(legLengthHuman))
    refLegLengthRobot = 0.5356 # rest leg length / sledge position at touchdown posture (14°, 148°)
    refLegLengthRobot += 0.001 # to avoid ground penetration due to simulation angles deviating from desired ones
    # as the shank position always changes
    # the segment length of 0.27m was manually adjusted to prevent ground penetration on the initialization step
    segmentLengthRobot = 0.279

    scalingFactor = refLegLengthRobot / legLengthHuman

    # scale the human leg length to the robot leg length
    robotsLegLengthsList240Hz = [hop*scalingFactor for hop in legLenAllHops240HzList]
    # maxRobotoLegLen = np.max([hop[0] for hop in robotsLegLengthsList240Hz]+[hop[-1] for hop in robotsLegLengthsList240Hz])


    # ---------------------------------------------------------------------------------
    # extract the HIP and KNEE ANGLES from the robots leg length for each individual hop
    # ---------------------------------------------------------------------------------

    hipAnglesAllHopsList240Hz = []
    kneeAnglesAllHopsList240Hz = []
    for robotHopLegLen in robotsLegLengthsList240Hz:
        arcusArguments = robotHopLegLen / (2*segmentLengthRobot)
        maxArg = np.max(arcusArguments)
        hipAngles = np.arccos(arcusArguments)
        kneeAngles = 2 * np.arcsin(arcusArguments)
        hipAnglesAllHopsList240Hz.append(hipAngles)
        kneeAnglesAllHopsList240Hz.append(kneeAngles)

    if THESIS:
        # get mean TD hip and knee angles to use it as desired values for the flight phase posture
        allHipTDAngles = deg([hop[0] for hop in hipAnglesAllHopsList240Hz])
        allKneeTDAngles = deg([hop[0] for hop in kneeAnglesAllHopsList240Hz])
        meanHipTDAngle = np.mean(allHipTDAngles)
        meanKneeTDAngle = np.mean(allKneeTDAngles)
        print("Mean TD conditions are {}° hip angle and {}° knee angle!".format(meanHipTDAngle, meanKneeTDAngle))
        # tools.severalPlots([allHipTDAngles, allKneeTDAngles], ["Hip TD Angles", "Knee TD Angles"])


    # -----------------------------------------------------------------------------------
    # get leg len VELOCITIES as well as joint angular velocities by taking the derivative
    # -----------------------------------------------------------------------------------

    robotLegSpeedsAllHops240HzList = [np.gradient(hop, 1 / 240) for hop in robotsLegLengthsList240Hz]
    hipAngVelsAllHops240HzList = [np.gradient(hop, 1 / 240) for hop in hipAnglesAllHopsList240Hz]
    kneeAngVelsAllHops240HzList = [np.gradient(hop, 1 / 240) for hop in kneeAnglesAllHopsList240Hz]


    # ---------------------------------
    # further CLEANING of ref data
    # ---------------------------------

    # as we have now several list (one for each trajec), it is better to first collect all the nr. of bad hops
    # and then delete them from all lists at once
    badHopsNrs = set()

    # remove hops with deviating LO hip angles as there were some problems observed during training with that
    liftoffHipAnglesAllHopsList = [hop[-1] for hop in hipAnglesAllHopsList240Hz]
    meanLOHipAngle = np.mean(liftoffHipAnglesAllHopsList)
    stdLOHipAngle = np.std(liftoffHipAnglesAllHopsList)
    for hopNr in range(len(liftoffHipAnglesAllHopsList)):
        if (liftoffHipAnglesAllHopsList[hopNr] - meanLOHipAngle) > stdLOHipAngle*1.25:
            badHopsNrs.add(hopNr)

    # print("{} bad hops detected from comparing LO hip angles!".format(len(badHopsNrs)))

    hopNr = 1

    if THESIS:
        tools.severalPlots([robotsLegLengthsList240Hz[hopNr], robotLegSpeedsAllHops240HzList[hopNr],
                            deg(hipAnglesAllHopsList240Hz[hopNr]), hipAngVelsAllHops240HzList[hopNr],
                            deg(kneeAnglesAllHopsList240Hz[hopNr]), kneeAngVelsAllHops240HzList[hopNr]],
                           yLabels=["Leg Length [m]", "Leg Length\nDerivative [m/s]", "Hip Angles [°]", "Hip Angle\nVelocities [rad/s]",
                                    "Knee Angles [°]", "Knee Angle\nVelocities [rad/s]"],
                           title="Test Reference Data Before Interpolation")


    # ----------------------------------------------------------------------
    # INTERPOLATE ALL REF TRAJECS for each individual hop at 240Hz to get 2kHz data out of it
    # ----------------------------------------------------------------------

    desiredFrequency = 2e3

    # sledge pos, sledge vel, hip angles, hip angVels, knee angles, knee angVels
    iSledgePos, iSledgeVel, iHipAngle, iHipAngVel, iKneeAngle, iKneeAngVel = range(6)
    refTrajecsAllHops240Hz = [robotsLegLengthsList240Hz, robotLegSpeedsAllHops240HzList,
                              hipAnglesAllHopsList240Hz, hipAngVelsAllHops240HzList,
                              kneeAnglesAllHopsList240Hz, kneeAngVelsAllHops240HzList]

    refTrajecsAllHops2kHz = [[],[],[],[],[],[]]

    for trajecIndex in range(len(refTrajecsAllHops240Hz)):
        for hopIndex in range(len(refTrajecsAllHops240Hz[trajecIndex])):
            hop = refTrajecsAllHops240Hz[trajecIndex][hopIndex]
            currentXAxis = np.arange(0, np.size(hop), 1)
            newXAxis = np.arange(0, np.size(hop), 240 / desiredFrequency)
            # cut the last 8 points as they will be all equal to the last point from the origin data
            interpolatedHop = np.interp(newXAxis, currentXAxis, hop)[:-8]
            refTrajecsAllHops2kHz[trajecIndex].append(interpolatedHop)

    # tools.plotMeanStdOverSeveralHops(refTrajecsAllHops2kHz[iSledgePos])
    # exit(33)

    # get each hops STANCE PHASE DURATION
    stancePhaseDurationsList = np.array([np.size(hop) for hop in refTrajecsAllHops2kHz[iSledgePos]])

    # get all the TOUCHDOWN CONDITIONS
    totalNrOfHops = len(stancePhaseDurationsList)
    touchdownConditions = np.zeros([6, totalNrOfHops])

    for trajecIndex in range(len(refTrajecsAllHops2kHz)):
        for hopIndex in range(len(refTrajecsAllHops2kHz[trajecIndex])):
            oneTrajecForOneHop = refTrajecsAllHops2kHz[trajecIndex][hopIndex]
            touchdownConditions[trajecIndex, hopIndex] = oneTrajecForOneHop[0]

    # set to True to get modified ref trajectories for training
    RESCALE_DATA_IN_TIME = False
    if RESCALE_DATA_IN_TIME:
        # scale the data horizontally by 20% to get a new reference trajectory
        for trajecIndex in range(len(refTrajecsAllHops2kHz)):
            for hopIndex in range(len(refTrajecsAllHops2kHz[trajecIndex])):
                refTrajecsAllHops2kHz[trajecIndex][hopIndex] = tools.rescaleInTime(
                    refTrajecsAllHops2kHz[trajecIndex][hopIndex], 1.2)

    # correct the data a bit (as we do not have the correct segment lengths due to the shank length changing with time)
    # the foot position should be about 1cm behind: reduce hip angle and lift sledge a bit up
    # the data is also only used for initialization
    hipAngleDecrease = rad(2)
    for hopIndex in range(len(refTrajecsAllHops2kHz[iHipAngle])):
        refTrajecsAllHops2kHz[iHipAngle][hopIndex] -= hipAngleDecrease

    if THESIS:
        tools.plot(deg(touchdownConditions[iKneeAngle, :]),
                   ylabel="Touchdown Knee Angles for all Hops [°]",
                   xlabel="Nr. of Hop []")

        tools.plot(stancePhaseDurationsList*1/desiredFrequency,
                   ylabel="Stance Phase Duration [s]", xlabel="Nr. of Hop []")

    if groundDropTest and not onlyPerturbedHops:
        # lift all sledge Pos by groundDropHeight
        for hopIndex in range(len(refTrajecsAllHops2kHz[iSledgePos])):
            for i in range(len(refTrajecsAllHops2kHz[iSledgePos][hopIndex])):
                refTrajecsAllHops2kHz[iSledgePos][hopIndex][i] += perturbationHeight

    if THESIS or __name__ == "__main__":
        tools.severalPlots([refTrajecsAllHops2kHz[iSledgePos][hopNr], refTrajecsAllHops2kHz[iSledgeVel][hopNr],
                            deg(refTrajecsAllHops2kHz[iHipAngle][hopNr]), refTrajecsAllHops2kHz[iHipAngVel][hopNr],
                            deg(refTrajecsAllHops2kHz[iKneeAngle][hopNr]), refTrajecsAllHops2kHz[iKneeAngVel][hopNr]],
                           yLabels=["Leg Length [m]", "Leg Length\nDerivative [m/s]",
                                    "Hip Angles [°]", "Hip Angle\nVelocities [rad/s]",
                                    "Knee Angles [°]", "Knee Angle\nVelocities [rad/s]"],
                           xLabels=["Time [1/2000s]"], thesis=True,
                           title="Interpolated Reference Data for one single hop's stance phase")

    # to return only ref trajecs for one single hop
    RETURN_ONLY_ONE_HOP = False

    if RETURN_ONLY_ONE_HOP:
        refTrajecOneHop = [trajec[0] for trajec in refTrajecsAllHops2kHz]
        # tools.severalPlots(refTrajecOneHop, yLabels=["sth"] * 6)
        return refTrajecOneHop
    else:
        return refTrajecsAllHops2kHz, touchdownConditions



if __name__ == "__main__":
    getReferenceTrajecsForRobot(groundDropTest=False, onlyPerturbedHops=False)
    # uncomment to get only perturbed hops to make the ground drop experiment
    # getReferenceTrajecsForRobot(groundDropTest=True, onlyPerturbedHops=True)
