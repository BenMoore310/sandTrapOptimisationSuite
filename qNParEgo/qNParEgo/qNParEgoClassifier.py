import sys
import os
import torch
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from botorch.utils.transforms import unnormalize, normalize
from botorch.utils.sampling import draw_sobol_samples
from botorch.optim.optimize import optimize_acqf, optimize_acqf_list
from botorch.acquisition.objective import GenericMCObjective
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
import time
import warnings
from multiprocessing import Pool
from botorch import fit_gpytorch_mll
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions.dominated import (
    DominatedPartitioning,
)
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.sampling import sample_simplex

warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
from . import VIClassifier as Classifier
from . import EIACQF
import numpy as np
import logging

logger = logging.getLogger(__name__)


tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    # "device": "cpu",
}
SMOKE_TEST = os.environ.get("SMOKE_TEST")
verbose = True


class qNParEgo:
    def __init__(self):
        self.problem = None
        self.prepareProblem = None
        self.bounds = None
        self.batchSize = 1
        self.SMOKE_TEST = os.environ.get("SMOKE_TEST")
        self.BATCH_SIZE = 1
        self.NUM_RESTARTS = 50 if not SMOKE_TEST else 2
        self.RAW_SAMPLES = 2048 if not SMOKE_TEST else 4
        self.N_BATCH = 150 if not SMOKE_TEST else 5
        self.MC_SAMPLES = 128 if not SMOKE_TEST else 16

    def initialize_model(self, train_x, train_obj, featuresFeas, targetsFeas):
        train_x = normalize(train_x, self.bounds)

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

        model = ModelListGP(*models)
        mll = SumMarginalLogLikelihood(model.likelihood, model)
        return mll, model, feaseLikelihood, feaseModel

    def optimize_qnparego_and_get_observation(
        self,
        model,
        train_x,
        train_obj,
        feaseModel,
        feaseLikelihood,
        sampler,
        standard_bounds,
    ):
        """Samples a set of random weights for each candidate in the batch, performs sequential greedy optimization
        of the qNParEGO acquisition function, and returns a new candidate and observation.
        """
        train_x = normalize(train_x, self.bounds)
        with torch.no_grad():
            pred = model.posterior(train_x).mean
        acq_func_list = []
        for _ in range(self.BATCH_SIZE):
            weights = sample_simplex(train_obj.shape[-1], **tkwargs).squeeze()
            objective = GenericMCObjective(
                get_chebyshev_scalarization(weights=weights, Y=pred)
            )

            acq_func = EIACQF.qLogNoisyExpectedImprovement(  # pyre-ignore: [28]
                feaseModel=feaseModel,
                feaseLikelihood=feaseLikelihood,
                model=model,
                objective=objective,
                X_baseline=train_x,
                sampler=sampler,
                prune_baseline=True,
            )
            acq_func_list.append(acq_func)
        # optimize
        candidates, _ = optimize_acqf_list(
            acq_function_list=acq_func_list,
            bounds=standard_bounds,
            num_restarts=self.NUM_RESTARTS,
            raw_samples=self.RAW_SAMPLES,  # used for intialization heuristic
            options={"batch_limit": 5, "maxiter": 200},
        )
        # observe new values
        new_x = unnormalize(candidates.detach(), bounds=self.bounds)

        nSims = train_obj.shape[-1]

        # move everything back to CPU for multiprocessing
        new_x = new_x.to("cpu")
        self.bounds = self.bounds.to("cpu")
        feaseModel = feaseModel.to("cpu")
        feaseLikelihood = feaseLikelihood.to("cpu")

        # needs to be normalised to make any sense!!
        # new_xNormalised = normalize(new_x, self.bounds)
        # print('feasibility of new x:', Classifier.GPEval(new_xNormalised, feaseLikelihood, new_x))

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
        logger.info("BEGINNING OF qNParEgo OPTIMISER")
        logger.info("###########################################")
        totalTimeInitial = time.monotonic()

        hvs_qNParEgo = []

        # logger.info(f"Using device: {tkwargs["device"]}")
        # logger.info(f"Torch version: {torch.__version__}")
        self.problem = functionCall
        self.prepareProblem = prepareFunctionCall
        self.bounds = torch.from_numpy(bounds)
        logger.info(f"Problem bounds: {self.bounds}")
        # print shape of bounds
        logger.info(f"Problem bounds shape: {self.bounds.shape}")
        standard_bounds = torch.zeros(2, len(self.bounds[-1]), **tkwargs)
        standard_bounds[1] = 1.0

        train_x_parego, train_obj_parego, train_x_fease, train_obj_fease = (
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

        logger.info(f"Initial qNParEgo features: {train_x_parego}")
        logger.info(f"Initial qNParEgo targets: {train_obj_parego}")

        # Find the worst value in each objective to set as the reference point
        self.refVector = 0.5 * torch.min(train_obj_parego, dim=0).values

        logger.info(f"Reference point: {self.refVector}")

        mll_parego, model_parego, feaseLikelihood, feaseModel = self.initialize_model(
            train_x_parego, train_obj_parego, train_x_fease, train_obj_fease
        )
        # compute hypervolume

        bd = DominatedPartitioning(
            ref_point=self.refVector.to(**tkwargs), Y=train_obj_parego.to(**tkwargs)
        )
        volume = bd.compute_hypervolume().item()

        hvs_qNParEgo.append(volume)

        # run N_BATCH rounds of BayesOpt after the initial random batch
        iteration = 1
        while iteration < self.N_BATCH:
        # for iteration in range(1, self.N_BATCH + 1):
            iterationTimeInitial = time.monotonic()

            # print(mll_qnehvi)

            fit_gpytorch_mll(mll_parego)

            # define the qEI and qNEI acquisition modules using a QMC sampler
            parego_sampler = SobolQMCNormalSampler(
                sample_shape=torch.Size([self.MC_SAMPLES])
            )

            # optimize acquisition functions and get new observations

            (new_x_parego, new_obj_parego) = self.optimize_qnparego_and_get_observation(
                model_parego,
                train_x_parego,
                train_obj_parego,
                feaseModel,
                feaseLikelihood,
                parego_sampler,
                standard_bounds,
            )

            # print('HERE')

            # print('newx_qnehvi:', new_x_qnehvi)
            # print(new_x_qnehvi.shape)

            if torch.any(new_obj_parego == 0):
                logger.info("New point is infeasible - re-running iteration")
                train_x_fease = torch.cat([train_x_fease, new_x_parego])
                train_obj_fease = torch.cat([train_obj_fease, torch.tensor([0])])
                iteration -= 1
                completeIter = False
            else:
                logger.info("New point is feasible")
                train_x_fease = torch.cat([train_x_fease, new_x_parego])
                train_obj_fease = torch.cat([train_obj_fease, torch.tensor([1])])
                train_x_parego = torch.cat([train_x_parego, new_x_parego])
                train_obj_parego = torch.cat([train_obj_parego, new_obj_parego])
                completeIter = True

            # update training points
            # train_x_qnehvi = torch.cat([train_x_qnehvi, new_x_qnehvi])
            # train_obj_qnehvi = torch.cat([train_obj_qnehvi, new_obj_qnehvi])
            # train_con = torch.cat([train_con, newFease])

            bd = DominatedPartitioning(
                ref_point=self.refVector.to(**tkwargs), Y=train_obj_parego.to(**tkwargs)
            )
            volume = bd.compute_hypervolume().item()
            logger.info(f"New hypervolume: {volume}")
            hvs_qNParEgo.append(volume)

            # print('###########################################')
            # print('SAVING IS CURRENTLY TURNED OFF')
            # print('###########################################')

            np.savetxt(
                f"qNParEgo/parEgoResults/featuresqNParEgo.txt",
                torch.Tensor.numpy(train_x_parego),
            )
            np.savetxt(
                f"qNParEgo/parEgoResults/targetsqNParEgo.txt",
                torch.Tensor.numpy(train_obj_parego),
            )
            np.savetxt(
                f"qNParEgo/parEgoResults/targetsFease.txt",
                torch.Tensor.numpy(train_obj_fease),
            )
            np.savetxt(
                f"qNParEgo/parEgoResults/featuresFease.txt",
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
                mll_parego,
                model_parego,
                feaseLikelihood,
                feaseModel,
            ) = self.initialize_model(
                train_x_parego, train_obj_parego, train_x_fease, train_obj_fease
            )


            iterationTimeFinal = time.monotonic()
            if completeIter == True:
                logger.info(f"Iteration time: {iterationTimeFinal - iterationTimeInitial}")

            iteration += 1

            # if verbose:
            #     logger.info(
            #         f"\nBatch {iteration:>2}: Hypervolume (qNParEgo) = "
            #         f"({hvs_qNParEgo[-1]:>4.2f}), "
            #         f"time = {t1-t0:>4.2f}.",
            #         end="",
            #     )
            # else:
            #     logger.info(".", end="")
        logger.info('Optimisation Complete')
        totalTimeFinal = time.monotonic()
        logger.info(f'Total time: {totalTimeFinal - totalTimeInitial}')

