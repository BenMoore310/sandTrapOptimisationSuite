import os
import torch
import numpy as np
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from botorch.utils.transforms import unnormalize, normalize
from botorch.utils.sampling import draw_sobol_samples
from botorch.optim.optimize import optimize_acqf, optimize_acqf_list
from botorch.acquisition.objective import GenericMCObjective
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)
from botorch.acquisition.multi_objective.logei import (
    qLogNoisyExpectedHypervolumeImprovement,
)
from botorch.utils.sampling import sample_simplex
import time
import warnings

from botorch import fit_gpytorch_mll
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions.dominated import (
    DominatedPartitioning,
)
from botorch.utils.multi_objective.pareto import is_non_dominated
import multiprocessing
from multiprocessing import Pool
from botorch.acquisition.objective import ConstrainedMCObjective

from botorch.acquisition.multi_objective.objective import IdentityMCMultiOutputObjective

import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from gpytorch.kernels.scale_kernel import ScaleKernel
from botorch.models.gpytorch import GPyTorchModel
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models import SingleTaskGP, ModelListGP
from botorch.fit import fit_gpytorch_mll
from botorch.models.transforms.outcome import Standardize
from botorch.utils.transforms import unnormalize, normalize

from functools import partial

from torch import Tensor

from . import VIClassifier as Classifier

from . import qNEHVIACQF

from scipy.optimize import differential_evolution
import logging

logger = logging.getLogger(__name__)


warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    # "device": "cpu",
}
SMOKE_TEST = os.environ.get("SMOKE_TEST")
verbose = True


class qNEHVI:
    def __init__(self):
        self.problem = None
        self.prepareProblem = None
        self.bounds = None
        self.refVector = None
        self.batchSize = 1
        self.SMOKE_TEST = os.environ.get("SMOKE_TEST")
        self.BATCH_SIZE = 1
        self.NUM_RESTARTS = 50 if not SMOKE_TEST else 2
        self.RAW_SAMPLES = 4096 if not SMOKE_TEST else 4
        self.N_BATCH = 150 if not SMOKE_TEST else 5
        self.MC_SAMPLES = 32000 if not SMOKE_TEST else 16

    def initialize_model(self, train_x, train_obj, featuresFeas, targetsFeas):
        # define models for objective and constraint
        train_x = normalize(train_x, self.bounds)

        # print('trainx:', train_x)
        # print(train_x.shape)
        # print('trainobj:', train_obj)
        # print(train_obj.shape)

        # train_con = train_con.unsqueeze(-1)

        # # print(train_obj.shape, train_con.shape)

        # train_y = torch.cat([train_obj, train_con], dim=-1)

        models = []
        for i in range(train_obj.shape[-1]):
            train_y = train_obj[..., i : i + 1]
            # train_yvar = torch.full_like(train_y, 1e-7)
            models.append(
                SingleTaskGP(train_x, train_y, outcome_transform=Standardize(m=1))
            )

        feaseModel, feaseLikelihood = Classifier.GPTrain(
            normalize(featuresFeas, self.bounds), targetsFeas.squeeze()
        )

        # find the minimum predicted mean value of the feaseModel
        # def surrogate_objectiveMin(x):
        #     newValue = Classifier.GPEval(feaseModel, feaseLikelihood, torch.tensor(x, dtype=torch.float32).unsqueeze(0))

        #     return newValue

        # #find max predicted mean value of the feaseModel
        # def surrogate_objectiveMax(x):
        #     newValue = Classifier.GPEval(feaseModel, feaseLikelihood, torch.tensor(x, dtype=torch.float32).unsqueeze(0))

        #     return -newValue

        # # # change bounds to (min, max) format
        # DEbounds = torch.tensor(self.bounds, dtype=torch.float64)
        # DEbounds = torch.transpose(self.bounds, 0, 1)
        # print(DEbounds.shape)

        # feaseMin = differential_evolution(surrogate_objectiveMin, bounds=DEbounds.numpy(), strategy='best1bin', maxiter=500, popsize=25, tol=0.01)
        # print('Minimum of feasibility classifier:', feaseMin)

        # feaseMax = differential_evolution(surrogate_objectiveMax, bounds=DEbounds.numpy(), strategy='best1bin', maxiter=500, popsize=25, tol=0.01)
        # print('Maximum of feasibility classifier:', feaseMax)

        model = ModelListGP(*models)
        mll = SumMarginalLogLikelihood(model.likelihood, model)
        return mll, model, feaseLikelihood, feaseModel

    def optimize_qnehvi_and_get_observation(
        self,
        model,
        train_x,
        train_obj,
        feaseModel,
        feaseLikelihood,
        sampler,
        standard_bounds,
    ):
        """Optimizes the qEHVI acquisition function, and returns a new candidate and observation."""

        # move everything to GPU if available
        model = model.to(**tkwargs)
        feaseModel = feaseModel.to(**tkwargs)
        train_x = train_x.to(**tkwargs)
        train_obj = train_obj.to(**tkwargs)
        standard_bounds = standard_bounds.to(**tkwargs)
        feaseLikelihood = feaseLikelihood.to(**tkwargs)
        self.bounds = self.bounds.to(**tkwargs)
        self.refVector = self.refVector.to(**tkwargs)

        # partition non-dominated space into disjoint rectangles
        acq_func = qNEHVIACQF.qLogNoisyExpectedHypervolumeImprovementClassifier(
            feaseModel=feaseModel,
            feaseLikelihood=feaseLikelihood,
            model=model,
            ref_point=self.refVector,  # use known reference point
            X_baseline=normalize(train_x, self.bounds),
            prune_baseline=True,  # prune baseline points that have estimated zero probability of being Pareto optimal
            sampler=sampler,
            # TODO add the fease stuff to the acquisition function
        )
        # optimize
        candidates, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=standard_bounds,
            q=self.BATCH_SIZE,
            num_restarts=self.NUM_RESTARTS,
            raw_samples=self.RAW_SAMPLES,  # used for intialization heuristic
            options={"batch_limit": 15, "maxiter": 500},
            sequential=True,
        )
        # observe new values
        new_x = unnormalize(candidates.detach(), bounds=self.bounds)
        # print('passed numObj =', train_obj.shape[-1])
        # new_obj = self.problem(train_obj.shape[-1], new_x.numpy(), int(len(self.bounds[-1])/2))
        # new_obj = new_obj_true + torch.randn_like(new_obj_true) * NOISE_SE

        nSims = train_obj.shape[-1]

        # move everything back to CPU for multiprocessing
        new_x = new_x.to("cpu")
        self.bounds = self.bounds.to("cpu")
        feaseModel = feaseModel.to("cpu")
        feaseLikelihood = feaseLikelihood.to("cpu")

        res = self.prepareProblem(new_x, int(len(self.bounds[-1]) / 2))

        if res == 1:
            logger.info("Case preparation successful")
            with Pool(processes=nSims) as pool:
                numberEfficiencies = pool.starmap(
                    self.problem,
                    [
                        (objIdx, new_x, int(len(self.bounds[-1]) / 2))
                        for objIdx in range(nSims)
                    ],
                )

            numberEfficiencies = torch.reshape(
                torch.tensor(numberEfficiencies), (1, nSims)
            )

            # if any entries are 0, replace whole row with 0s
            # (done after values are returned)
            # if (numberEfficiencies == 0).any():
            #     numberEfficiencies = torch.full((1,nSims), 0)
            #     newFease = torch.tensor([0])
            # else:
            #     newFease = torch.tensor([1])
        else:
            logger.info("returning zeros to optimiser")
            numberEfficiencies = torch.zeros((1, nSims))
        # if torch.isnan(numberEfficiencies).any():
        #     numberEfficiencies = np.full((1,nSims), np.nan)
        #     newFease = torch.tensor([0])
        # else:
        #     newFease = torch.tensor([1])

        return new_x, numberEfficiencies

    def optimise(
        self,
        bounds,
        functionCall,
        prepareFunctionCall,
        features,
        targets,
        featuresFeas,
        targetsFeas,
    ):
        logger.info("###########################################")
        logger.info("BEGINNING OF qNEHVI OPTIMISER")
        logger.info("###########################################")
        totalTimeInitial = time.monotonic()


        hvs_qnehvi = []

        # print("Using device:", tkwargs["device"])
        # print("Torch version", torch.__version__)
        self.problem = functionCall
        self.prepareProblem = prepareFunctionCall
        self.bounds = torch.from_numpy(bounds)
        logger.info(f"Problem bounds: {self.bounds}")
        # print shape of bounds
        logger.info(f"Problem bounds shape: {self.bounds.shape}")
        standard_bounds = torch.zeros(2, len(self.bounds[-1]), **tkwargs)
        standard_bounds[1] = 1.0

        train_x_qnehvi, train_obj_qnehvi, train_x_fease, train_obj_fease = (
            torch.from_numpy(features),
            torch.from_numpy(targets),
            torch.from_numpy(featuresFeas),
            torch.from_numpy(targetsFeas)
            # torch.from_numpy(np.loadtxt('coupledOptimisers/qNEHVIResults/featuresqNEHVI.txt')),
            # torch.from_numpy(np.loadtxt('coupledOptimisers/qNEHVIResults/targetsqNEHVI.txt')),
            # torch.from_numpy(np.loadtxt('coupledOptimisers/qNEHVIResults/featuresFease.txt')),
            # torch.from_numpy(np.loadtxt('coupledOptimisers/qNEHVIResults/targetsFease.txt'))
        )
        train_x_fease = train_x_fease.to(dtype=torch.float32)
        train_obj_fease = train_obj_fease.to(dtype=torch.int32)
        # train_con = feas_list

        logger.info(f"Initial qNEHVI features: {train_x_qnehvi}")
        logger.info(f"Initial qNEHVI targets: {train_obj_qnehvi}")

        # Find the worst value in each objective to set as the reference point
        self.refVector = 0.5 * torch.min(train_obj_qnehvi, dim=0).values

        logger.info(f"Reference point: {self.refVector}")

        mll_qnehvi, model_qnehvi, feaseLikelihood, feaseModel = self.initialize_model(
            train_x_qnehvi, train_obj_qnehvi, train_x_fease, train_obj_fease
        )
        # compute hypervolume

        bd = DominatedPartitioning(
            ref_point=self.refVector.to(**tkwargs), Y=train_obj_qnehvi.to(**tkwargs)
        )
        volume = bd.compute_hypervolume().item()

        hvs_qnehvi.append(volume)

        # run N_BATCH rounds of BayesOpt after the initial random batch

        iteration = 1
        while iteration < self.N_BATCH:
        # for iteration in range(1, self.N_BATCH + 1):
            iterationTimeInitial = time.monotonic()

            # print(mll_qnehvi)

            fit_gpytorch_mll(mll_qnehvi)

            # define the qEI and qNEI acquisition modules using a QMC sampler
            qnehvi_sampler = SobolQMCNormalSampler(
                sample_shape=torch.Size([self.MC_SAMPLES])
            )

            # optimize acquisition functions and get new observations

            (new_x_qnehvi, new_obj_qnehvi) = self.optimize_qnehvi_and_get_observation(
                model_qnehvi,
                train_x_qnehvi,
                train_obj_qnehvi,
                feaseModel,
                feaseLikelihood,
                qnehvi_sampler,
                standard_bounds,
            )

            # print('HERE')

            # print('newx_qnehvi:', new_x_qnehvi)
            # print(new_x_qnehvi.shape)

            if torch.any(new_obj_qnehvi == 0):
                logger.info("New point is infeasible")
                train_x_fease = torch.cat([train_x_fease, new_x_qnehvi])
                train_obj_fease = torch.cat([train_obj_fease, torch.tensor([0])])
                iteration -= 1
                completeIter = False
            else:
                logger.info("New point is feasible")
                train_x_fease = torch.cat([train_x_fease, new_x_qnehvi])
                train_obj_fease = torch.cat([train_obj_fease, torch.tensor([1])])
                train_x_qnehvi = torch.cat([train_x_qnehvi, new_x_qnehvi])
                train_obj_qnehvi = torch.cat([train_obj_qnehvi, new_obj_qnehvi])
                completeIter = True

            # update training points
            # train_x_qnehvi = torch.cat([train_x_qnehvi, new_x_qnehvi])
            # train_obj_qnehvi = torch.cat([train_obj_qnehvi, new_obj_qnehvi])
            # train_con = torch.cat([train_con, newFease])

            bd = DominatedPartitioning(
                ref_point=self.refVector.to(**tkwargs), Y=train_obj_qnehvi.to(**tkwargs)
            )
            volume = bd.compute_hypervolume().item()
            logger.info(f"New hypervolume: {volume}")
            hvs_qnehvi.append(volume)

            # print('###########################################')
            # print('SAVING IS CURRENTLY TURNED OFF')
            # print('###########################################')

            np.savetxt(
                f"qNEHVI/qNEHVIResults/featuresqNEHVI.txt",
                torch.Tensor.numpy(train_x_qnehvi),
            )
            np.savetxt(
                f"qNEHVI/qNEHVIResults/targetsqNEHVI.txt",
                torch.Tensor.numpy(train_obj_qnehvi),
            )
            np.savetxt(
                f"qNEHVI/qNEHVIResults/targetsFease.txt",
                torch.Tensor.numpy(train_obj_fease),
            )
            np.savetxt(
                f"qNEHVI/qNEHVIResults/featuresFease.txt",
                torch.Tensor.numpy(train_x_fease),
            )

            # # update progress
            # for hvs_list, train_obj in zip(
            #     (hvs_qnehvi),
            #     (
            #         train_obj_qnehvi,
            #     ),
            # ):
            #     # compute hypervolume
            #     bd = DominatedPartitioning(ref_point=self.refVector, Y=train_obj)
            #     volume = bd.compute_hypervolume().item()
            #     hvs_list.append(volume)

            # reinitialize the models so they are ready for fitting on next iteration
            # Note: we find improved performance from not warm starting the model hyperparameters
            # using the hyperparameters from the previous iteration
            (
                mll_qnehvi,
                model_qnehvi,
                feaseLikelihood,
                feaseModel,
            ) = self.initialize_model(
                train_x_qnehvi, train_obj_qnehvi, train_x_fease, train_obj_fease
            )

            iterationTimeFinal = time.monotonic()

            if completeIter == True:
                logger.info(f"Iteration time: {iterationTimeFinal - iterationTimeInitial}")

            iteration += 1

        logger.info('Optimisation Complete')
        totalTimeFinal = time.monotonic()
        logger.info(f'Total time: {totalTimeFinal - totalTimeInitial}')
