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

warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

tkwargs = {
    "dtype": torch.double,
    # "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "device": "cpu",
}
SMOKE_TEST = os.environ.get("SMOKE_TEST")
verbose = True


class qNEHVI:
    def __init__(self):
        self.problem = None
        self.bounds = None
        self.refVector = None
        self.batchSize = 1
        self.SMOKE_TEST = os.environ.get("SMOKE_TEST")
        self.BATCH_SIZE = 1
        self.NUM_RESTARTS = 10 if not SMOKE_TEST else 2
        self.RAW_SAMPLES = 512 if not SMOKE_TEST else 4
        self.N_BATCH = 150 if not SMOKE_TEST else 5
        self.MC_SAMPLES = 128 if not SMOKE_TEST else 16

    def initialize_model(self, train_x, train_obj):
        # define models for objective and constraint
        train_x = normalize(train_x, self.bounds)

        # print('trainx:', train_x)
        # print(train_x.shape)
        # print('trainobj:', train_obj)
        # print(train_obj.shape)

        models = []
        for i in range(train_obj.shape[-1]):
            train_y = train_obj[..., i : i + 1]
            train_yvar = torch.full_like(train_y, 1e-7)
            models.append(
                SingleTaskGP(
                    train_x, train_y, train_yvar, outcome_transform=Standardize(m=1)
                )
            )
        model = ModelListGP(*models)
        mll = SumMarginalLogLikelihood(model.likelihood, model)
        return mll, model

    def optimize_qnehvi_and_get_observation(
        self, model, train_x, train_obj, sampler, standard_bounds
    ):
        """Optimizes the qEHVI acquisition function, and returns a new candidate and observation."""
        # partition non-dominated space into disjoint rectangles
        acq_func = qLogNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=self.refVector,  # use known reference point
            X_baseline=normalize(train_x, self.bounds),
            prune_baseline=True,  # prune baseline points that have estimated zero probability of being Pareto optimal
            sampler=sampler,
        )
        # optimize
        candidates, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=standard_bounds,
            q=self.BATCH_SIZE,
            num_restarts=self.NUM_RESTARTS,
            raw_samples=self.RAW_SAMPLES,  # used for intialization heuristic
            options={"batch_limit": 5, "maxiter": 200},
            sequential=True,
        )
        # observe new values
        new_x = unnormalize(candidates.detach(), bounds=self.bounds)
        # print('passed numObj =', train_obj.shape[-1])
        # new_obj = self.problem(train_obj.shape[-1], new_x.numpy(), int(len(self.bounds[-1])/2))
        # new_obj = new_obj_true + torch.randn_like(new_obj_true) * NOISE_SE

        nSims = train_obj.shape[-1]

        with Pool(processes=nSims) as pool:
            numberEfficiencies = pool.starmap(
                self.problem,
                [
                    (objIdx, new_x, int(len(self.bounds[-1]) / 2))
                    for objIdx in range(nSims)
                ],
            )

        numberEfficiencies = torch.reshape(torch.tensor(numberEfficiencies), (1, nSims))
        return new_x, numberEfficiencies

    def optimise(self, bounds, functionCall, features, targets):
        hvs_qnehvi = []

        print("Using device:", tkwargs["device"])
        print("Torch version", torch.__version__)
        self.problem = functionCall
        self.bounds = torch.from_numpy(bounds)
        print("Problem bounds:", self.bounds)
        # print shape of bounds
        print("Problem bounds shape:", self.bounds.shape)
        standard_bounds = torch.zeros(2, len(self.bounds[-1]), **tkwargs)
        standard_bounds[1] = 1.0

        train_x_qnehvi, train_obj_qnehvi = (
            torch.from_numpy(features),
            torch.from_numpy(targets),
        )

        print("Initial qNEHVI features:", train_x_qnehvi)
        print("Initial qNEHVI targets:", train_obj_qnehvi)

        # Find the worst value in each objective to set as the reference point
        self.refVector = 0.5 * torch.min(train_obj_qnehvi, dim=0).values

        print("Reference point:", self.refVector)

        mll_qnehvi, model_qnehvi = self.initialize_model(
            train_x_qnehvi, train_obj_qnehvi
        )
        # compute hypervolume
        bd = DominatedPartitioning(ref_point=self.refVector, Y=train_obj_qnehvi)
        volume = bd.compute_hypervolume().item()

        hvs_qnehvi.append(volume)

        # run N_BATCH rounds of BayesOpt after the initial random batch
        for iteration in range(1, self.N_BATCH + 1):
            t0 = time.monotonic()

            # print(mll_qnehvi)

            fit_gpytorch_mll(mll_qnehvi)

            # define the qEI and qNEI acquisition modules using a QMC sampler
            qnehvi_sampler = SobolQMCNormalSampler(
                sample_shape=torch.Size([self.MC_SAMPLES])
            )

            # optimize acquisition functions and get new observations

            (
                new_x_qnehvi,
                new_obj_qnehvi,
            ) = self.optimize_qnehvi_and_get_observation(
                model_qnehvi,
                train_x_qnehvi,
                train_obj_qnehvi,
                qnehvi_sampler,
                standard_bounds,
            )

            # print('HERE')

            # print('newx_qnehvi:', new_x_qnehvi)
            # print(new_x_qnehvi.shape)

            # update training points
            train_x_qnehvi = torch.cat([train_x_qnehvi, new_x_qnehvi])
            train_obj_qnehvi = torch.cat([train_obj_qnehvi, new_obj_qnehvi])

            bd = DominatedPartitioning(ref_point=self.refVector, Y=train_obj_qnehvi)
            volume = bd.compute_hypervolume().item()
            print("New hypervolume:  ", volume)
            hvs_qnehvi.append(volume)

            np.savetxt(
                f"coupledOptimisers/qNEHVIResults/featuresqNEHVI.txt",
                torch.Tensor.numpy(train_x_qnehvi),
            )
            np.savetxt(
                f"coupledOptimisers/qNEHVIResults/targetsqNEHVI.txt",
                torch.Tensor.numpy(train_obj_qnehvi),
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
            mll_qnehvi, model_qnehvi = self.initialize_model(
                train_x_qnehvi, train_obj_qnehvi
            )

            t1 = time.monotonic()

            if verbose:
                print(
                    f"\nBatch {iteration:>2}: Hypervolume (qNEHVI) = "
                    f"({hvs_qnehvi[-1]:>4.2f}), "
                    f"time = {t1-t0:>4.2f}.",
                    end="",
                )
            else:
                print(".", end="")
