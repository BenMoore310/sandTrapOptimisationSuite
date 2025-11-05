import numpy as np
from scipy.stats import qmc
import argparse
import matplotlib.pyplot as plt
import json
import subprocess
import sandTrapCatmull as STC
import HVKG.HVKGClassifier as HVKGOpt
import math
import multiprocessing
from multiprocessing import Pool
import shutil
import tempfile
import os
import torch
from scipy.spatial.distance import cdist
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename="HVKG.log", encoding="utf-8", filemode="w", level=logging.DEBUG
)
logging.getLogger("matplotlib.font_manager").disabled = True
# plt.style.use(["science", "notebook"])

# Some functions for handling the fact that x coords get ordered before simulation:
# for each simulation based on n pairs of x,y coordinates, the one function value returned
# gives the result for n! feasible designs.


# cwdPath = "/home/bm424/Projects/sandTrapShapeOpt"

# def run_on_server(server, cwdPath):
#     print(f"Running on server: {server}")
#     subprocess.run(
#         ["bash", "allRun", server],
#         cwd=cwdPath + "/runDirectory",
#         check=True
#     )


def classifierDummyEvalBool(x):
    # print(x.shape)
    # print('28')
    for i in range(x.shape[0] - 1):
        # print('30')
        # print('i=', i, x[i,0], x[i+1,0])
        if x[i, 0] > x[i + 1, 0] + 1:
            # print('33')
            # print('not sorted')
            return False
    # print('va')
    # print('37')
    return True


def classifierDummyEval(x):
    # print(x.shape)
    # print('28')
    for i in range(x.shape[0] - 1):
        # print('30')
        # print('i=', i, x[i,0], x[i+1,0])
        if x[i, 0] > x[i + 1, 0] + 1:
            # print('33')
            # print('not sorted')
            return 0
    # print('va')
    # print('37')
    return 1


def greedyMaximin(x, nSamples):
    distMatrix = cdist(x, x)
    np.fill_diagonal(distMatrix, np.inf)
    initial = np.argmax(distMatrix.min(axis=1))
    selected = [initial]

    while len(selected) < nSamples:
        # distance from each candidate to nearest selected point:
        dToSel = distMatrix[:, selected].min(axis=1)
        dToSel[selected] = -np.inf  # points dont select themsevles
        nextPoint = np.argmax(dToSel)
        selected.append(nextPoint)
    return selected


def prepareRunDir(sample, numBasis, objective):
    isFease = classifierDummyEval(np.reshape(sample, (numBasis, 2)))
    # print('50')
    if isFease == 0:
        logger.info("Design violates geometry constraints - skipping simulation...")
        numberEfficiency = 0
        return numberEfficiency
    else:
        logger.info("Geometry constraints satisfied, continuing...")

    STC.main(sample, numBasis)

    # subprocess.run(["bash", "allrunInit"], cwd="/home/bm424/Projects/sandTrapShapeOptBenchmarking/qNParEgo/parEgoCaseDir", check=True)
    serverName = "cfd7"

    # objective_strings = ["0.025", "0.075", "1.125"]
    objectiveDict = {0: "0.025", 1: "0.075", 2: "1.125"}

    flowRate = objectiveDict[objective]

    process = subprocess.Popen(
        ["bash", "allrunInit", serverName, flowRate],
        cwd="/home/bm424/Projects/sandTrapShapeOptBenchmarking/HVKG/HVKGCaseDir",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    for line in process.stdout:
        logger.info(line.strip())
    process.wait()

    logger.info("Run directory initialised...")

    numberEfficiency = 1

    return numberEfficiency


def simulateDesign(objective, successBool):
    """
    Simulate the design using the current parameters.
    """

    cwdPath = "/home/bm424/Projects/sandTrapShapeOptBenchmarking/HVKG"

    serverName = "cfd7"

    # objective_strings = ["0.025", "0.075", "1.125"]
    objectiveDict = {0: "0.025", 1: "0.075", 2: "1.125"}

    flowRate = objectiveDict[objective]

    if successBool == 0:
        logger.info(f"{flowRate}: Design not valid, returning efficiency = 0")

        efficiency = 0

        return efficiency

    elif successBool == 1:
        logger.info(f"{flowRate}: Design valid, proceeding...")

        process = subprocess.Popen(
            ["bash", "runServerJob2", serverName, flowRate],
            cwd=cwdPath,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        for line in process.stdout:
            logger.info(line.strip())
        process.wait()

        escapedParticles = np.loadtxt(cwdPath + f"/efficiency_{flowRate}.txt")

        efficiency = (1 - (escapedParticles / 70000)) * 100

        # numberEfficiency = efficiencies[0]
        logger.info(f"{flowRate}: Final efficiency = {efficiency} %")

        return efficiency

    else:
        logger.error("Validity bool not 0 or 1, how has this happened...?")

        return None


def main(numBasis, numObj, initialSamples, seed, classifierMC=False):
    # list of arrays of values for each objective
    featureList = np.empty((0, numBasis * 2))
    targetList = np.empty((0, numObj))
    featureListFease = np.empty((initialSamples, numBasis * 2))
    targetListFease = np.empty((0,))

    # evaluatedObjectives = np.random.randint(
    #     low=0, high=6, size=(initialSamples), dtype=int
    # )

    lbX = 3.7
    lbY = -1.54

    # x upper limit reduced from 23.5 to 21 to stop generation of designs which asymptote downwards towards outlet (unmeshable but valid)
    ubX = 21
    ubY = 1

    bounds = []

    for i in range(numBasis):
        bounds.append([lbX, ubX])
        bounds.append([lbY, ubY])

    bounds = np.array(bounds)

    lowBounds = bounds[:, 0]
    highBounds = bounds[:, 1]

    # uncomment from here for the initial solutions:
    #######################################################

    # # for i in range(numObj):
    # sampler = qmc.LatinHypercube(d=len(bounds), seed=seed+i)

    # # remember this is initial samples per objective, not in total
    # # init per obj should be 2*D + 1

    # samplesMC = sampler.random(n=2500)

    # featureListFease = qmc.scale(samplesMC, lowBounds, highBounds)

    # # creates samples from which we then draw feasible solutions
    # samples = sampler.random(n=initialSamples*10)

    # # Scale samples to bounds
    # initialPopulation = qmc.scale(samples, lowBounds, highBounds)
    # logger.info(f"Initial population shape: {initialPopulation.shape}")
    # mask = classifierDummyEvalBool(initialPopulation)

    # mask = []

    # for sample in initialPopulation:
    #     boolFease = classifierDummyEvalBool(np.reshape(sample, (numBasis,2)))
    #     mask.append(boolFease)

    # # print(mask)
    # validInitPop = initialPopulation[mask]

    # # print(validInitPop.shape)

    # # final initial population is the maximin of the validInitPop

    # idx = greedyMaximin(validInitPop, initialSamples)

    # initialPopulation = validInitPop[idx]

    # logger.info(f"initialPopulation: {initialPopulation}")

    # if classifierMC == True:
    #     # feas_list = torch.empty((0,1), dtype=torch.float32)
    #     for sample in featureListFease:
    #         # print("sample", sample)
    #         # print(sample.shape)
    #         # print(np.reshape(sample, (numBasis,2)))

    #         # call STC and generate spline from current sample

    #         isFeas = classifierDummyEval(np.reshape(sample, (numBasis,2)))

    #         # print('isFeas = ', isFeas)
    #         # sample = np.reshape(np.array(numberEfficiencies), (1,numObj))
    #         # print(targetListFease.shape, isFeas.shape)
    #         targetListFease = np.append(targetListFease, isFeas)

    #     # print('Feas list: ', featureListFease)

    #     # np.savetxt('initFeas_list.txt', torch.Tensor.numpy())

    # # initialPopulation = featureList
    # testedSample = 1
    # for sample in initialPopulation:

    #     logger.info(f"sample {sample}")
    #     logger.info(f"Sample Number = {testedSample}")
    #     # print(sample.shape)
    #     # print(np.reshape(sample, (numBasis,2)))

    #     # call STC and generate spline from current sample

    #     nSims = numObj
    #     # print('266')

    #     prepareRunDir(sample, numBasis)

    #     with Pool(processes=nSims) as pool:
    #         numberEfficiencies = pool.starmap(simulateDesign, [(objIdx, sample, numBasis) for objIdx in range(numObj)])

    #     # print('270')
    #     numberEfficiencies = np.reshape(np.array(numberEfficiencies), (1,numObj))
    #     logger.info(f"Returned efficiencies: {numberEfficiencies}")

    #     # if any entries are 0, replace whole row with 0s
    #     if np.any(numberEfficiencies == 0):
    #         # numberEfficiencies = np.full((1,numObj), 0)
    #         targetListFease = np.append(targetListFease, 0)
    #         featureListFease = np.vstack((featureListFease, sample))
    #     else:
    #         targetListFease = np.append(targetListFease, 1)
    #         targetList = np.vstack((targetList, numberEfficiencies))
    #         featureListFease = np.vstack((featureListFease, sample))
    #         featureList = np.vstack((featureList, sample))

    #     testedSample += 1

    #     # print(featureList.shape)
    #     # print(featureListFease.shape)
    #     # print(targetList.shape)
    #     # print(targetListFease.shape)

    #     # if any entries are nan, replace whole row with nans
    #     # if np.any(np.isnan(numberEfficiencies)):
    #     #     numberEfficiencies = np.full((1,numObj), np.nan)
    #     #     feas_list = torch.vstack((feas_list, torch.tensor([0])))
    #     # else:
    #     #     feas_list = torch.vstack((feas_list, torch.tensor([1])))

    #     # numberEfficiency = simulateDesign(numObj, sample, numBasis)

    #     # targetList[objIdx] = np.append(targetList[objIdx], numberEfficiency)
    #     # print(numberEfficiencies.shape)
    #     # print(targetList.shape)

    #     # print('Target list: ', targetList)
    #     # print('Feature list: ', featureList)
    #     # print('Target Feas list: ', targetListFease)
    #     # print('Feature Fease list: ', featureListFease)

    # logger.info('###########################################')
    # logger.info('INITIAL SAMPLE COMPLETED')
    # logger.info('###########################################')

    # np.savetxt('initTargets.txt', targetList)
    # np.savetxt("initFeatures.txt", featureList)
    # np.savetxt('initTargetsFeas.txt', targetListFease)
    # np.savetxt("initFeaturesFeas.txt", featureListFease)

    featureList = np.loadtxt("initFeatures.txt")
    targetList = np.loadtxt("initTargets.txt")
    targetListFease = np.loadtxt("initTargetsFeas.txt")
    featureListFease = np.loadtxt("initFeaturesFeas.txt")

    qNEHVI = HVKGOpt.HVKG()

    qNEHVI.optimise(
        bounds.T,
        simulateDesign,
        prepareRunDir,
        featureList,
        targetList,
        featureListFease,
        targetListFease,
    )


# python3.11 optimiserHeadHVKG.py --numBasis 4 --numObj 2 --initialSamples 5

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the head script for HydroShield optimisation."
    )
    parser.add_argument(
        "--numBasis",
        type=int,
        default=2,
        help="Number of Beta basis functions to use in the curve generation.",
    )
    parser.add_argument(
        "--numObj",
        type=int,
        default=1,
        help="Number of objectives to be optimised over.",
    )
    parser.add_argument(
        "--initialSamples",
        type=int,
        default=5,
        help="Number of initial samples for the Latin Hypercube sampling.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=3,
        help="Sets the seed for the initial LHS generation",
    )
    parser.add_argument(
        "--classifierMC",
        type=bool,
        default=False,
        help="If True, runs MC dummy check to train classifier on predicted invalid designs.",
    )

    args = parser.parse_args()

    main(args.numBasis, args.numObj, args.initialSamples, args.seed, args.classifierMC)
