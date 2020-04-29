"""
This module contains lot's of useful functions, that are used all across the project.
A big part concentrates on processing and plotting data.

"""

import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
from simple_pid import PID
import os, inspect
from thesis_galljamov18.python import settings

# -----------------------------------
# Local collections and flags
# -----------------------------------

# used for exponential smoothing
_exp_weighted_averages = {}

# used for running mean
_running_means = {}

# for shading flight phase in graphs
_listOfFlightPhaseAreasToShade = None


#####################################################################################
#                                                                                   #
#                                 PLOT DATA                                         #
#                                                                                   #
#####################################################################################

# -----------------------------------
# Plot settings
# -----------------------------------

SHADE_COLOR = (0.3, 0.3, 0.3, 0.1)
THESIS_FONT_SIZE = settings.PLOTS_FONT_SIZE
LINE_WIDTH = settings.PLOTS_LINE_WIDTH
FIGURE_FORMAT = settings.PLOT_FIGURE_SAVE_FORMAT


def log(text: str):
    """
    Plots the desired text in the console standing out from the other text
    by having '---' and line breaks below and above:
    """
    print("\n------------\n"+text+"\n------------\n")


def plot(data, title=None, ylabel="", xlabel = "", legend=None, doBlock:bool=True):
    """
    Simply plots any 1D collection with labels and title by calling one single line
    """
    plt.plot(data)
    if title:
        plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    if legend is not None:
        plt.legend(legend)
    plt.show(block=doBlock)


def histogram(data, xLabel="", yLabel="", title=None,  minValue = 0, maxValue=500, width=1):
    """
    Counts the occurrences of values between minValue and maxValue in data and plot a simple histrogram
    """
    bins = np.linspace(minValue, maxValue, maxValue / width)

    plt.hist(data, bins)
    if title:
        plt.title(title)
    plt.ylabel(yLabel)
    plt.xlabel(xLabel)
    plt.show()


def severalPlots(listOfDataToPlot: list, yLabels: list, overlayPlots: bool=False, xLabels: list=["Timesteps []"],
                 title=None, legend: list = [], sampleRate = 2000,
                 doBlock: bool=True, shareX: bool=False, thesis=False):
    """
    Displays several plots above each other, each in it's individual subplot. The lists can have different sizes.
    :param listOfDataToPlot: [[x1, x2, ..., xn], [y1, y2, ..., ym], [z1, z2, z3]]
    """

    if thesis:
        plt.rcParams.update({'font.size': 12})
        plt.rcParams.update({'lines.linewidth': 2})

    numberOfPlots = len(listOfDataToPlot)

    if len(yLabels) == 1 and numberOfPlots > 1:
        yLabels = str.split(yLabels[0], ";")

    plt.figure(1)

    if title:
        plt.suptitle(title)

    onlyOneXLabel = xLabels and len(xLabels) == 1
    subplot = None
    for i in range(numberOfPlots):
        try:
            if not overlayPlots:
                if subplot is not None and shareX:
                    subplot = plt.subplot(numberOfPlots*100 + 10 + i + 1, sharex=subplot)
                else: subplot = plt.subplot(numberOfPlots*100 + 10 + i + 1)
                plt.ylabel(yLabels[i])
            else: plt.ylabel(yLabels[0])
            plt.plot(listOfDataToPlot[i])
            if xLabels and not onlyOneXLabel:
                plt.xlabel(xLabels[i])
            if(len(legend)>0):
                plt.legend(legend)
            elif overlayPlots:
                plt.legend(yLabels)
        except:
            import traceback
            traceback.print_stack()
            raise ValueError("Exception when plotting several plots. Please check collection Nr. {}".format(i+1))

    if onlyOneXLabel:
        plt.xlabel(xLabels[0])

    plt.show(block=doBlock)


def overlayingHistograms(list_of_data, labelsCommaSeparated ="", minValue = 0, maxValue=500, width=10,
                         xLabel=None, yLabel=None):
    """
    Plots several historgrams in the same subplot, all with different colors and an alpha = 0.5.
    :param list_of_data: [[hist1data],[hist2data],...,[histNdata]]
    """

    bins = np.linspace(minValue, maxValue, maxValue/width)
    y_labels = str.split(labelsCommaSeparated, ',')

    plt.rcParams.update({'font.size': THESIS_FONT_SIZE})

    subplot = plt.subplot()

    for i in range(len(list_of_data)):
        subplot.hist(list_of_data[i], bins, alpha=0.5, label=y_labels[i])

    if xLabel: subplot.set_xlabel(xLabel)
    if yLabel: subplot.set_ylabel(yLabel)

    from matplotlib.ticker import PercentFormatter
    subplot.xaxis.set_major_formatter(PercentFormatter(xmax=maxValue))

    plt.legend()
    plt.show()


def overlayingPlots(list_of_lists_or_values, labels_comma_separated_string: str = "", xLabels: list=["Time [s]"],
                    legend: str= '', title=None, thesis=False, sampleRate = 2000, shadeFlightPhase=False):
    """
    Plots several graphs where more then one graphs can be in one subplot
    :parameter list_of_lists_or_values: for example [[subplot1graph1, subplot1graph2], [subplot2graph1, subplot2graph2], [subplot3]]
    :parameter labels_comma_separated_string: Exactly one label per subplot is required! Example: 'subplot1label, subplot2label, subplot3label'
    :parameter legend: string, where legends are separated by ; and individual legend entries by ,!
    In case a plot doesn't need a legend, just put a ' ;' inside. In above example: 'legend11, legend12; legend21, legend22; '
    """

    if thesis:
        plt.rcParams.update({'font.size': THESIS_FONT_SIZE})
        plt.rcParams.update({'lines.linewidth': LINE_WIDTH})
        plt.rcParams.update({'savefig.format': 'eps'})

    numberOfPlots = len(list_of_lists_or_values)
    y_labels = str.split(labels_comma_separated_string,',')

    assert numberOfPlots == len(y_labels), "Mismatch between number of plots ({}) and number of y_labels ({}).\n" \
                                           "Exactly one label per subplot is required!".format(numberOfPlots, len(y_labels))

    legends = stringToListOfLists(legend)

    assert numberOfPlots == len(legends), "Mismatch between number of plots ({}) and number of legends ({}).\n" \
                                          "Exactly one legend per subplot have to be named, even a legend is not necessary!".format(numberOfPlots, len(legends))

    figure, subplots = plt.subplots(numberOfPlots, sharex=True, sharey=False)
    # hotfix to allow to plot one sinlge overlayed subplot
    if numberOfPlots == 1:
        subplots = [subplots]

    if title: plt.suptitle(title)

    onlyOneXLabel = xLabels and len(xLabels) == 1

    global _listOfFlightPhaseAreasToShade

    for i in range(numberOfPlots):
        data_to_plot = list_of_lists_or_values[i]
        if isinstance(data_to_plot[0],list) or isinstance(data_to_plot[0], np.ndarray): # means we have a list of lists instead of a list of data to plot
            numOfPlots = len(data_to_plot)
            for graphIndex in range(numOfPlots):
                actualSubplot = data_to_plot[graphIndex]
                plotSampleLen = np.size(actualSubplot)
                subplots[i].plot(actualSubplot, linewidth=2 if thesis else 1.5)
                sampledTimeInSecs = (plotSampleLen / sampleRate)
                x_axis_steps = plotSampleLen / 5
                subplots[i].set_xticks(np.arange(0, plotSampleLen * 1.2, x_axis_steps))
                subplots[i].set_xticklabels(np.round(np.arange(0, sampledTimeInSecs*1.2, sampledTimeInSecs/5), 2))
                subplots[i].set_rasterized(True)

                if shadeFlightPhase:
                    # get x areas to shade
                    if _listOfFlightPhaseAreasToShade is None:
                        xAreasToShade = [index for index in range(len(actualSubplot)) if actualSubplot[index] is None]
                        # we need the beginning and end of each shaded area
                        # therefore divide all areas to list of areas
                        listOfAreas = []
                        listOfAreas.append([])
                        listOfAreasIndex = 0
                        for xIndex in range(len(xAreasToShade)-1):
                            listOfAreas[listOfAreasIndex].append(xAreasToShade[xIndex])
                            # when the next value more then just one bigger than the previous
                            if xAreasToShade[xIndex]+1 < xAreasToShade[xIndex+1]:
                                listOfAreasIndex += 1
                                listOfAreas.append([])
                        _listOfFlightPhaseAreasToShade = listOfAreas
                    for area in _listOfFlightPhaseAreasToShade:
                        if len(area) < 10:
                            continue
                        subplots[i].axvspan(min(area), max(area), color=SHADE_COLOR)

            if len(legends) > 0: subplots[i].legend(legends[i], loc='upper right')
        else:
            subplots[i].plot(data_to_plot,linewidth=2 if thesis else 1.5)
            if shadeFlightPhase:  # and None in actualSubplot:
                # get x areas to shade
                if _listOfFlightPhaseAreasToShade is None:
                    xAreasToShade = [index for index in range(len(actualSubplot)) if actualSubplot[index] is None]
                    # we need the beginning and end of each shaded area
                    # therefore divide all areas to list of areas
                    listOfAreas = []
                    listOfAreas.append([])
                    listOfAreasIndex = 0
                    for xIndex in range(len(xAreasToShade) - 1):
                        listOfAreas[listOfAreasIndex].append(xAreasToShade[xIndex])
                        # when the next value more then just one bigger than the previous
                        if xAreasToShade[xIndex] + 1 < xAreasToShade[xIndex + 1]:
                            listOfAreasIndex += 1
                            listOfAreas.append([])
                    _listOfFlightPhaseAreasToShade = listOfAreas
                for area in _listOfFlightPhaseAreasToShade:
                    if len(area) < 10:
                        continue
                    subplots[i].axvspan(min(area), max(area), color=SHADE_COLOR)

        font = {'size': THESIS_FONT_SIZE+2 if thesis else 12}

        subplots[i].set_ylabel(y_labels[i], fontdict=font)

        if xLabels and not onlyOneXLabel:
            subplots[i].set_xlabel(xLabels[i], fontdict=font)

    if onlyOneXLabel:
        plt.xlabel(xLabels[0], fontsize=THESIS_FONT_SIZE+2 if thesis else 12)

    plt.show()

def plotMeanStdOverSeveralHops(data, title=""):
    """
    Plots the mean and std of different params over several hops
    :param data: data should be a list of trajectories, which are again lists like [[hop1],[hop2],...,[hopN]]
    """

    plt.rcParams.update({'font.size': THESIS_FONT_SIZE})
    plt.rcParams.update({'lines.linewidth': 2})
    plt.rcParams.update({'savefig.format': FIGURE_FORMAT})

    log("Plotting mean and std of {} hops".format(np.size(data)))
    hopLens = [len(hop) for hop in data]

    dataSameLens = [hop[0:min(hopLens) - 1] for hop in data]
    dataSameLens = np.array(dataSameLens)

    dataMean = np.mean(dataSameLens, 0)
    dataStd = np.std(dataSameLens, 0)

    countSamples = len(dataMean)
    xAxisLegLen = np.linspace(0, countSamples, countSamples)

    fig, subPlot = plt.subplots()
    subPlot.plot(xAxisLegLen, dataMean)
    plt.fill_between(xAxisLegLen,
                     dataMean - dataStd, dataMean + dataStd,
                     facecolor=(0., 0., 1., 0.15))
    from matplotlib.ticker import PercentFormatter
    subPlot.set_xticks(np.arange(0, 472, 472 / 5))

    subPlot.set_xlabel("Percentage of stance phase duration")
    subPlot.set_ylabel("Vertical COM position [m]")

    plt.locator_params(axis='y', nbins=6)
    subPlot.xaxis.set_major_formatter(PercentFormatter(xmax=countSamples - 5))
    subPlot.set_rasterized(True)
    plt.title(title + "\nMean and Std over {} hops".format(len(data)))
    plt.show()


def plotForceLengthCurve(grfsAllHops, legLensAllHops, sampleRate, sim=True, cutFirstXTimestes=0, perturbed=False):
    """
    Plots the Force Length Relationship after some sophisticated postprocessing of the inputted data, being
    :param grfsAllHops: a list of grfs for several hops [[grfHop1], [grfHop2], ..., [grfHopN]]
    :param legLensAllHops: analogous to grfsAllHops
    """
    # zero padding (right padding is already introduced by the flight phase detection delay)
    flightZeros = [0]*18
    if sim:
        grfsAllHops = [flightZeros + hop[cutFirstXTimestes:] for hop in grfsAllHops]

    # filter grfs
    # overlayingPlots([grfsAllHops], "grfs unfiltered")
    grfsAllHopsFilt = [lowpassFilterData(hop, sampleRate, 20, 2) for hop in grfsAllHops]
    # overlayingPlots([grfsAllHops], "grfs filtered")

    if sim:
        # undo the zero paddings again (the padding on the right side is due to the flight phase detection delay)
        grfsAllHopsFilt = [hop[cutFirstXTimestes+18:-18] for hop in grfsAllHopsFilt]

    for i in range(2):
        # get mean and std of hop duration
        # and repeat after outliers have been removed
        allHopsDurations = [len(hop) for hop in legLensAllHops]
        meanHopDuration = np.mean(allHopsDurations)
        stdHopDuration = np.std(allHopsDurations)

        if i == 0:
            # remove all "hard" outliers from grfs and legLen data
            grfsAllHops = [hop for hop in grfsAllHops if abs(len(hop)-meanHopDuration) < 2*stdHopDuration]
            legLenAllHops = [hop for hop in legLensAllHops if abs(len(hop)-meanHopDuration) < 2*stdHopDuration]

    # scale all signals to the same hop length
    grfsAllHopsFilt = [rescaleInTime(hop, meanHopDuration/len(hop)) for hop in grfsAllHopsFilt]
    legLenAllHops = [rescaleInTime(hop, meanHopDuration/len(hop)) for hop in legLenAllHops]

    # convert both lists into numpy arrays
    grfsAllHopsFilt = np.array(grfsAllHopsFilt)
    legLenAllHops = np.array(legLenAllHops)

    # scale GRFs with body weight
    grfsAllHopsBodyWeighScaled = grfsAllHopsFilt / (23.29 if sim else 643.3)

    # get leg compressions
    tdLegLensAllHops = [hop[0] for hop in legLenAllHops]
    meanTDLegLen = np.mean(tdLegLensAllHops)
    legCompressionAllHops = meanTDLegLen - legLenAllHops

    # get leg compression and grfs mean
    meanLegCompression = np.mean(legCompressionAllHops,0)
    meanGRFs = np.mean(grfsAllHopsBodyWeighScaled,0)

    # plot the curve
    plt.rcParams.update({'font.size': THESIS_FONT_SIZE})
    plt.rcParams.update({'lines.linewidth': 2})
    plt.rcParams.update({'savefig.format': 'eps'})

    fontDict = {'fontsize':THESIS_FONT_SIZE}
    subplot = plt.subplot()
    subplot.plot(meanLegCompression, meanGRFs)
    subplot.set_xlabel(r"$\Delta l_{leg} / l_{leg}$", fontDict)
    subplot.set_ylabel(r"$F_{GRF} / F_{bodyweight}$", fontDict)

    from matplotlib.ticker import PercentFormatter, FormatStrFormatter
    subplot.xaxis.set_major_formatter(PercentFormatter(xmax=meanTDLegLen))
    plt.locator_params(axis='x', nbins=6)

    def linFunc(x,m,b):
        return m*x + b

    m, b = fitCurveToData(linFunc, meanLegCompression, meanGRFs)
    p = subplot.plot(meanLegCompression, linFunc(meanLegCompression, m, b), linestyle= ":", linewidth=3)
    color = p[-1].get_color()
    plt.annotate("stiffness = {}N/m".format(np.around(m,1)),
                                            xy=(0.5, 0.06), xycoords='axes fraction', color=color)

    subplot.plot(meanLegCompression[0], meanGRFs[0], 's', markersize=10)
    subplot.plot(meanLegCompression[-1], meanGRFs[-1], '^', markersize=10)

    plt.show()


#####################################################################################
#                                                                                   #
#                              HELPER FUNCTIONS                                     #
#                                                                                   #
#####################################################################################


def getCurrentPathAsString():
    currentDir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    return currentDir


def getCurrentParentFolderPathAsStrings():
    currentdir = getCurrentPathAsString()
    parentdir = os.path.dirname(currentdir)
    return parentdir


#####################################################################################
#                                                                                   #
#                           PROCESS AND MODIFY DATA                                 #
#                                                                                   #
#####################################################################################


def scalePercentToRange(inputs:list, ranges:list):
    """:parameter inputs: list of inputs between 0 and 1 (100%)
       :parameter ranges: list of ranges as lists of two-elemented lists [min, max]"""
    scaledInputs = []

    for i in range(len(inputs)):
        curRange = ranges[i]
        newRange = curRange[1] - curRange[0]
        scaledInputs.append(
            inputs[i] * newRange + curRange[0])

    return scaledInputs


def linearScale(input, min_in, max_in, min_out, max_out):
    """scales an input with range [min_in: max_in] to range [min_out:max_out]"""
    return min_out + ((max_out - min_out) / (max_in - min_in)) * (input-min_in)


def normalizeMinMax(input, input_min, input_max):

    # simple normalization
    mean = (input_max + input_min) / 2
    max_deviation = input_max - input_min

    # without (*0.75) the values will be normalized to the range [-0.5:0.5], with it to a slightly bigger range
    normedInput = (input - mean) / (max_deviation * 0.75)

    if abs(normedInput) > 3:
        print("normalizeMinMax: normedValue bigger 3:\n"
              "input was {}, with ranges [{}:{}]".format(input, input_min,input_max))

    return normedInput


def lowpassFilterData(data, sample_rate, cutoff_freq, order=1):
    """
    Uses a butterworth filter to filter data in both directions without causing any delay!
    """

    # prepare input data
    # The length of the input vector data must be at least padlen, which is 6:
    count_data_points = len(data)
    if count_data_points == 0:
        return 0
    elif count_data_points < 7:
        passed_data = data
        data = np.zeros(7)
        for i in range(count_data_points):
            data[7-count_data_points+i] = passed_data[i]

    nyquist_freq = sample_rate/2
    norm_cutoff_freq = cutoff_freq/nyquist_freq

    b, a = signal.butter(order, norm_cutoff_freq, 'low')
    fltrd_data = signal.filtfilt(b, a, data)

    return fltrd_data


def averageFilterData(data):
    fltrd_data = []
    fltrd_data.append(data[0])
    fltrd_data.append((data[0]+data[1])/2)
    for i in range(len(data)-2):
        fltrd_data.append((fltrd_data[i]+data[i+1]+data[i+2])/3)
    return fltrd_data


def movingAverage(data, num_of_points_to_average_over):
    return np.mean(data[-min(num_of_points_to_average_over, len(data)):])


def runningMean(new_value, label: str):
    """
    Computes the running mean, given a new value.
    Several running means can be monitored parallely by providing different labels.
    :param label: give your running mean a name.
                  Will be used as a dict key to save current running mean value.
    :return: current running mean value for the provided label
    """

    if label in  _running_means:
        old_mean, num_values = _running_means[label]
        new_mean = (old_mean * num_values + new_value) / (num_values + 1)
        _running_means[label] = [new_mean, num_values + 1]
    else:
        # first value for the running mean
        _running_means[label] = [new_value, 1]

    return _running_means[label][0]


def exponentialRunningSmoothing(label, new_data_point, smoothing_factor):
    """
    Implements an exponential running smoothing filter.
    Several inputs can be filtered parallely by providing different labels.
    :param label: give your filtered data a name.
                  Will be used as a dict key to save current filtered value.
    :return: current filtered value for the provided label
    """
    global _exp_weighted_averages

    if str(label) == str(1): # knee motor torque
        debug = True

    if label not in _exp_weighted_averages:
        _exp_weighted_averages[label] = 0

    new_average = smoothing_factor * new_data_point + (1-smoothing_factor) * _exp_weighted_averages[label]
    _exp_weighted_averages[label] = new_average

    if str(label) == str(1) and new_data_point == -5:
        debug = True

    return new_average


def resetExponentialRunningSmoothing(label):
    """
    Sets the current value of the exponential running smoothing identified by the label to zero.
    """
    global _exp_weighted_averages
    _exp_weighted_averages[label] = 0
    return True


def rescaleInTime(data, scalingFactor):
    """adopted from https://stackoverflow.com/questions/38820132/how-to-scale-a-signal-in-python-on-the-x-axis"""
    lenData = len(data)
    return np.interp(np.linspace(0, lenData, scalingFactor * lenData + 1), np.arange(lenData), data)


def getRunningStats():
    class RunningStats:
        """Based on http://www.johndcook.com/standard_deviation.html
        copied from https://github.com/liyanage/python-modules/blob/master/running_stats.py"""

        def __init__(self):
            self.n = 0
            self.old_m = 0
            self.new_m = 0
            self.old_s = 0
            self.new_s = 0

        def clear(self):
            self.n = 0

        def push(self, x):
            self.n += 1

            if self.n == 1:
                self.old_m = self.new_m = x
                self.old_s = 0
            else:
                self.new_m = self.old_m + (x - self.old_m) / self.n
                self.new_s = self.old_s + (x - self.old_m) * (x - self.new_m)

                self.old_m = self.new_m
                self.old_s = self.new_s

        def mean(self):
            return self.new_m if self.n else 0.0

        def variance(self):
            return self.new_s / (self.n - 1) if self.n > 1 else 0.0

        def standard_deviation(self):
            return np.sqrt(self.variance())

    return RunningStats()


#####################################################################################
#                                                                                   #
#                                COLLECTIONS                                        #
#                                                                                   #
#####################################################################################


def fitCurveToData(funcToFitToData, xData, yData, plot=False):
    """adopted from: https://www.scipy-lectures.org/intro/scipy/auto_examples/plot_curve_fit.html
       :param funcToFitToData: should be like func(x_axis_data, param1, param2,...)
       :returns parameters for passed function as array"""
    from scipy import optimize
    params, paramVar = optimize.curve_fit(funcToFitToData, xData, yData)
    if plot:
        plt.figure()
        plt.scatter(xData, yData, label='Data')
        plt.plot(xData, funcToFitToData(xData, params[0], params[1]),
                 label='Fitted function')
    return params


def stringToListOfLists(string, separatorsList: list = [';',',']):
    """
    converts string to list considering two separations/splittings
    :param string: string like 'a,b;c,d'
    :param separatorsList: list of separators beginning from outer to inner like [';',',']
    :return: list of lists llike [[a,b],[c,d]]
    """
    list_of_lists = []
    list_of_splitted_strings = str.split(string, separatorsList[0])
    for el in list_of_splitted_strings:
        list_of_lists.append(str.split(el, separatorsList[1]))
    return list_of_lists


#####################################################################################
#                                                                                   #
#                                PID Control                                        #
#                                                                                   #
#####################################################################################

def getPID(Kp, Ki, Kd, min_output, max_output, sample_time):
    pid = PID(Kp, Ki, Kd, sample_time=sample_time, output_limits=(min_output, max_output))
    return pid


# initialize PIDs for angle control in flight phase
frequency = 5e-4
hip_pos_pid_flight = getPID(0.26, 0, 9.6, -5, 5, frequency)
knee_pos_pid_flight = getPID(0.18, 0, 2.8, -5, 5, frequency)

# initialize PIDs for angle control in stance phase
hip_pos_pid_stance = getPID(0.48, 0, 8.6, -5, 5, frequency)
knee_pos_pid_stance = getPID(0.52, 0, 2.2, -5, 5, frequency)


def getDesiredTorquesFromPositionPID(des_hip_angle, des_knee_angle, meas_hip_angle, meas_knee_angle, in_flight=True):
    """Get Torques to reach desired hip and knee angles."""

    if in_flight:
        hip_pos_pid_flight.setpoint = des_hip_angle
        knee_pos_pid_flight.setpoint = des_knee_angle

        hip_motor_tor = hip_pos_pid_flight(meas_hip_angle)
        knee_motor_tor = knee_pos_pid_flight(meas_knee_angle)
    else:
        # use different PD Parameters for stance phase
        hip_pos_pid_stance.setpoint = des_hip_angle
        knee_pos_pid_stance.setpoint = des_knee_angle

        hip_motor_tor = hip_pos_pid_stance(meas_hip_angle)
        knee_motor_tor = knee_pos_pid_stance(meas_knee_angle)

    return [hip_motor_tor, knee_motor_tor]