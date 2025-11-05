import os
import torch
import pymoo
from pymoo.problems import get_problem
from scipy.stats import qmc
import numpy as np
from botorch.test_functions.multi_objective import ZDT2
from botorch.models.cost import FixedCostModel
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.transforms import normalize, unnormalize
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from torch import Tensor
from gpytorch.priors import GammaPrior
from gpytorch.kernels import MaternKernel, ScaleKernel
from botorch.acquisition.multi_objective.logei import (
    qLogNoisyExpectedHypervolumeImprovement,
)
from botorch.optim.optimize import optimize_acqf
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.acquisition.multi_objective.hypervolume_knowledge_gradient import (
    _get_hv_value_function,
    qHypervolumeKnowledgeGradient,
)
from botorch.models.deterministic import GenericDeterministicModel
from botorch.sampling.list_sampler import ListSampler
from botorch.sampling.normal import IIDNormalSampler
import numpy as np
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)
from botorch.utils.multi_objective.pareto import _is_non_dominated_loop
from gpytorch import settings
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
import time
import warnings
from botorch import fit_gpytorch_mll
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.sampling.normal import SobolQMCNormalSampler

warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
import argparse
from multiprocessing import Pool

import logging
from . import VIClassifier as Classifier
from . import HVKGACQF
import subprocess


logger = logging.getLogger(__name__)


tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    # "device": "cpu",
}


class HVKG:
    def __init__(self):
        # self.n_var = n_var
        # self.n_obj = n_obj
        self.problem = None
        self.prepareProblem = None
        self.bounds = None
        self.refVector = None
        self.BATCH_SIZE = 1
        self.NUM_RESTARTS = 10 if not os.environ.get("SMOKE_TEST") else 2
        self.RAW_SAMPLES = 512 if not os.environ.get("SMOKE_TEST") else 4
        self.NUM_PARETO = 2 if os.environ.get("SMOKE_TEST") else 10
        self.NUM_FANTASIES = 2 if os.environ.get("SMOKE_TEST") else 32
        self.NUM_HVKG_RESTARTS = 1
        # self.MC_SAMPLES = 32000 if not os.environ.get("SMOKE_TEST") else 16
        # self.COST_BUDGET = 150 if not os.environ.get("SMOKE_TEST") else 54
        self.N_BATCH = 150
        self.nObj = None

    def initialize_model(self, train_x_list, train_obj_list, featuresFeas, targetsFeas):
        train_x_list = [normalize(train_x, self.bounds) for train_x in train_x_list]
        # print(train_x_list)

        models = []
        for i in range(len(train_obj_list)):
            # train_y = train_obj_list[i].unsqueeze(dim=1)
            # print(train_y.shape)
            # print(train_x_list[i].shape)
            # # train_yvar = torch.full_like(train_y, 1e-7)  # noiseless
            # models.append(
            #     SingleTaskGP(
            #         train_X=train_x_list[i],
            #         train_Y=train_y,
            #         outcome_transform=Standardize(m=1),
            #     )
            # )
            train_y = train_obj_list[i]
            models.append(
                SingleTaskGP(
                    train_X=train_x_list[i],
                    train_Y=train_y,
                    outcome_transform=Standardize(m=1),
                    covar_module=ScaleKernel(
                        MaternKernel(
                            nu=2.5,
                            ard_num_dims=train_x_list[0].shape[-1],
                            lengthscale_prior=GammaPrior(2.0, 2.0),
                        ),
                        outputscale_prior=GammaPrior(2.0, 0.15),
                    ),
                )
            )
        # print(targetsFeas)
        feaseModel, feaseLikelihood = Classifier.GPTrain(
            normalize(featuresFeas, self.bounds), targetsFeas.squeeze()
        )

        model = ModelListGP(*models)
        # print(model)
        mll = SumMarginalLogLikelihood(model.likelihood, model)
        return mll, model, feaseLikelihood, feaseModel

    def get_current_value(
        self,
        model,
        ref_point,
        bounds,
    ):
        """Helper to get the hypervolume of the current hypervolume
        maximizing set.
        """
        curr_val_acqf = _get_hv_value_function(
            model=model,
            ref_point=ref_point,
            use_posterior_mean=True,
        )
        _, current_value = optimize_acqf(
            acq_function=curr_val_acqf,
            bounds=bounds,
            q=self.NUM_PARETO,
            num_restarts=self.NUM_RESTARTS,
            raw_samples=1024,
            return_best_only=True,
            options={"batch_limit": 5},
        )
        return current_value

    def optimize_HVKG_and_get_obs_decoupled(
        self, model, standard_bounds, objective_indices, feaseModel, feaseLikelihood
    ):
        model = model.to(**tkwargs)
        feaseModel = feaseModel.to(**tkwargs)
        standard_bounds = standard_bounds.to(**tkwargs)
        feaseLikelihood = feaseLikelihood.to(**tkwargs)
        self.bounds = self.bounds.to(**tkwargs)
        self.refVector = self.refVector.to(**tkwargs)

        """Utility to initialize and optimize HVKG."""
        # cost_aware_utility = InverseCostWeightedUtility(cost_model=cost_model)

        current_value = self.get_current_value(
            model=model,
            ref_point=self.refVector,  # use known reference point
            bounds=standard_bounds,
        )

        acq_func = HVKGACQF.HVKGClassifier(
            feaseModel=feaseModel,
            feaseLikelihood=feaseLikelihood,
            model=model,
            ref_point=self.refVector,  # use known reference point
            num_fantasies=self.NUM_FANTASIES,
            num_pareto=self.NUM_PARETO,
            current_value=current_value,
        )
        # acq_func = qHypervolumeKnowledgeGradient(
        #     # feaseModel=feaseModel,
        #     # feaseLikelihood=feaseLikelihood,
        #     model=model,
        #     ref_point=self.refVector,  # use known reference point
        #     num_fantasies=self.NUM_FANTASIES,
        #     num_pareto=self.NUM_PARETO,
        #     current_value=current_value,
        # )
        # optimize acquisition functions and get new observations
        objective_vals = []
        objective_candidates = []
        for objective_idx in objective_indices:
            # set evaluation index to only condition on one objective
            # this could be multiple objectives
            X_evaluation_mask = torch.zeros(
                self.BATCH_SIZE,
                len(objective_indices),
                dtype=torch.bool,
                device=standard_bounds.device,
            )
            X_evaluation_mask[0, objective_idx] = 1
            acq_func.X_evaluation_mask = X_evaluation_mask
            logger.info(f"Optimising next candidate for objective {objective_idx+1}...")
            candidates, vals = optimize_acqf(
                acq_function=acq_func,
                num_restarts=self.NUM_HVKG_RESTARTS,
                raw_samples=self.RAW_SAMPLES,
                bounds=standard_bounds,
                q=self.BATCH_SIZE,
                sequential=False,
                options={"batch_limit": 1},
            )
            objective_vals.append(vals.view(-1))
            objective_candidates.append(candidates)
            # print("objective candidates", objective_candidates)
            # print("objective vals", objective_vals)
        # best_objective_index = torch.cat(objective_vals, dim=-1).argmax().item()
        # print('bestobjind', best_objective_index)
        # eval_objective_indices = [best_objective_index]
        # print(", Evaluated Objectives = ", objective_vals)
        # candidates = objective_candidates[best_objective_index]
        # vals = objective_vals[best_objective_index]
        # observe new values
        objective_candidates = torch.stack(objective_candidates, dim=0)
        new_x = unnormalize(objective_candidates.detach(), bounds=self.bounds)
        new_x = new_x.to("cpu")
        self.bounds = self.bounds.to("cpu")
        # for each new_x, initialise run directory and transfer files

        new_obj = []
        prepSuccess = (
            []
        )  # boolean mask as to whether the case prepared successfully or not
        # '0' means simulation will not take place.

        # print(new_x, int(len(self.bounds[-1]) / 2), objective_vals, objective_indices)

        for candidate, flowRate in zip(new_x, objective_indices):
            # print("candidate:", candidate)
            # print("flowRate:", flowRate)
            res = self.prepareProblem(
                candidate, int(len(self.bounds[-1]) / 2), flowRate
            )

            prepSuccess.append(res)

            logger.info(f"Case preparation bool result: {prepSuccess}")

        with Pool(processes=self.nObj) as pool:
            numberEfficiencies = pool.starmap(
                self.problem, zip(objective_indices, prepSuccess)
            )

        # print('new x:',new_x, new_x.shape)
        # TODO replace with function call
        # new_obj = self.problem(best_objective_index, new_x.numpy(), int(len(self.bounds[-1])/2))
        # new_obj = torch.from_numpy(np.array([new_obj])).to(**tkwargs)
        # new_obj = new_obj[..., eval_objective_indices]
        return new_x, torch.tensor(numberEfficiencies), objective_indices

    # define function to find model-estimated pareto set of
    # designs under posterior mean using NSGA-II

    # this is just to compare the estimated HV in each iteration to an analytical pareto front
    # to compare regrets between optimisers.

    # from pymoo.util.termination.max_gen import MaximumGenerationTermination

    def get_model_identified_hv_maximizing_set(
        self,
        model,
        population_size=100,
        max_gen=250,
    ):
        logger.info("Finding model HV")
        """Optimize the posterior mean using NSGA-II."""
        # tkwargs = {
        #     "dtype": problem.ref_point.dtype,
        #     "device": problem.ref_point.device,
        # }
        dim = len(self.bounds[-1])
        # print('dim =', dim)
        # since its bounds for each feature this gives the dimensionality of the feature landscape

        class ModelPosteriorMean(Problem):
            def __init__(self, n_obj):
                super().__init__(
                    n_var=dim,
                    n_obj=n_obj,
                    type_var=np.double,
                )
                self.xl = np.zeros(dim)
                self.xu = np.ones(dim)

            def _evaluate(self, x, out, *args, **kwargs):
                X = torch.from_numpy(x).to(**tkwargs)
                # print(X, X.shape)
                is_fantasy_model = (
                    isinstance(model, ModelListGP)
                    and model.models[0].train_targets.ndim > 2
                ) or (
                    not isinstance(model, ModelListGP) and model.train_targets.ndim > 2
                )
                with torch.no_grad():
                    with settings.cholesky_max_tries(9):
                        # eval in batch mode
                        y = model.posterior(X.unsqueeze(-2)).mean.squeeze(-2)
                        var = model.posterior(X.unsqueeze(-2)).variance.squeeze(-2)
                        std = var.sqrt()
                    if is_fantasy_model:
                        y = y.mean(dim=-2)
                        std = std.mean(dim=-2)
                out["F"] = y.cpu().numpy()
                out[
                    "uncertainty"
                ] = std.cpu().numpy()  # stores the predictive uncertainty

        sol = ModelPosteriorMean(self.nObj)
        algorithm = NSGA2(
            pop_size=population_size,
            eliminate_duplicates=True,
        )

        # sol = sol.to(**tkwargs)
        # algorithm = algorithm.to(**tkwargs)
        model = model.to(**tkwargs)

        res = minimize(
            sol,
            algorithm,
            termination=("n_gen", max_gen),
            # seed=0,  # fix seed
            verbose=False,
        )

        X = torch.tensor(
            res.X,
            **tkwargs,
        )
        X = unnormalize(X, self.bounds.to(**tkwargs))
        # print(X, X.shape)
        # Y = problem(X)
        Y = torch.Tensor(res.F)

        std = torch.Tensor(res.pop.get("uncertainty"))
        # print("std shape:", std.shape)
        # print(Y, Y.shape)
        # compute HV
        # print(self.refVector)
        # print(Y)
        partitioning = FastNondominatedPartitioning(
            ref_point=self.refVector.to(dtype=torch.float32), Y=Y
        )
        return partitioning.compute_hypervolume().item(), X, Y, std

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
        logger.info("BEGINNING OF HVKG OPTIMISER")
        logger.info("###########################################")

        # print("Using device:", tkwargs["device"])
        # print("Torch version", torch.__version__)

        # TODO these costs will need to be changed when I set this up for HydroShield
        # objective_costs = {0: 0.0, 1: 0.0}
        # objective_indices = list(objective_costs.keys())
        # objective_costs = {int(k): v for k, v in objective_costs.items()}
        # objective_costs_t = torch.tensor(
        #     [objective_costs[k] for k in sorted(objective_costs.keys())], **tkwargs
        # )
        # cost_model = FixedCostModel(fixed_cost=objective_costs_t)

        # generating the initial training data - i can replace this with LHS generation
        self.problem = functionCall
        self.prepareProblem = prepareFunctionCall
        self.bounds = torch.from_numpy(bounds)
        logger.info(f"Problem bounds: {self.bounds}")
        # print shape of bounds
        logger.info(f"Problem bounds shape: {self.bounds.shape}")
        standard_bounds = torch.zeros(2, len(self.bounds[-1]), **tkwargs)
        standard_bounds[1] = 1.0

        train_x_hvkg, train_obj_hvkg, train_x_fease, train_obj_fease = (
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

        # Load in initial data into correct data-structures

        # train_x_hvkgList = [torch.tensor([]) for i in range(len(self.bounds[-1]))]
        # train_obj_hvkgList = list(torch.unbind(train_obj_hvkg, dim=1))
        train_obj_hvkgList = list(train_obj_hvkg.split(1, dim=-1))
        train_x_hvkgList = [train_x_hvkg] * len(train_obj_hvkgList)

        self.nObj = len(train_obj_hvkgList)

        objective_indices = []
        for i in range(len(train_obj_hvkgList)):
            objective_indices.append(i)

        logger.info(train_x_hvkgList)
        logger.info(train_obj_hvkgList)

        # torch.manual_seed(0)
        verbose = True
        # N_INIT = 2 * len(self.bounds) + 1

        # total_cost = {"hvkg": 0.0, "qnehvi": 0.0, "random": 0.0}
        # total_cost = {"hvkg": 0.0}

        # call helper functions to generate initial training data and initialize model

        # train_x_hvkg = torch.from_numpy(features)
        # train_x_hvkg_list = list(torch.from_numpy(features))
        # train_obj_hvkg = torch.from_numpy(targets)

        # train_x_hvkg, train_obj_hvkg = self.generate_initial_data(n=N_INIT)
        # train_obj_hvkg_list = list(train_obj_hvkg.split(1, dim=-1))
        # print('trainObjHvkgList  ', train_obj_hvkg_list)
        # train_x_hvkg_list = [train_x_hvkg] * len(train_obj_hvkg_list)
        mll_hvkg, model_hvkg, feaseLikelihood, feaseModel = self.initialize_model(
            train_x_hvkgList, train_obj_hvkgList, train_x_fease, train_obj_fease
        )

        # set the reference vector based on the worst targets in each list in train_obj_hvkg_list
        self.refVector = 0.5 * torch.min(train_obj_hvkg, dim=0).values

        logger.info(f"Reference point: {self.refVector}")

        # self.referenceVector needs to be of shape (2,)
        # if len(self.refVector.shape) > 1:
        #     self.refVector = self.refVector.squeeze()

        # print("Reference Vector:", self.refVector)

        # cost_hvkg = cost_model(train_x_hvkg).sum(dim=-1)
        # total_cost["hvkg"] += cost_hvkg.sum().item()

        # fit the models
        fit_gpytorch_mll(mll_hvkg)

        iteration = 0

        # compute hypervolume
        hv, features, targets, stddv = self.get_model_identified_hv_maximizing_set(
            model=model_hvkg
        )

        np.savetxt(
            f"HVKG/HVKGModelParetoFronts/features/featuresIter{iteration}.txt",
            features.cpu().numpy(),
        )
        np.savetxt(
            f"HVKG/HVKGModelParetoFronts/targets/targetsIter{iteration}.txt",
            targets.cpu().numpy(),
        )
        np.savetxt(
            f"HVKG/HVKGModelParetoFronts/uncertainties/stdIter{iteration}.txt",
            stddv.cpu().numpy(),
        )

        hvs_hvkg = [hv]
        # if verbose:
        #     print(
        #         f"\nInitial: Hypervolume (qHVKG) = " f"({hvs_hvkg[-1]:>4.2f}).\n",
        #         end="",
        #     )
        logger.info(f"New Estimated Hypervolume: {hvs_hvkg[-1]}")
        # run N_BATCH rounds of BayesOpt after the initial random batch
        # active_algos = {k for k, v in total_cost.items() if v < self.COST_BUDGET}
        for iteration in range(1, self.N_BATCH + 1):
            t0 = time.monotonic()

            (
                new_x_hvkg,
                new_obj_hvkg,
                eval_objective_indices_hvkg,
            ) = self.optimize_HVKG_and_get_obs_decoupled(
                model_hvkg,
                standard_bounds=standard_bounds,
                objective_indices=objective_indices,
                feaseModel=feaseModel,
                feaseLikelihood=feaseLikelihood,
            )
            # print("eval objectives: ", eval_objective_indices_hvkg)
            # update training points

            for new_x, new_obj, obj_ind in zip(
                new_x_hvkg, new_obj_hvkg, eval_objective_indices_hvkg
            ):
                if new_obj == 0:
                    train_x_fease = torch.cat([train_x_fease, new_x])
                    train_obj_fease = torch.cat([train_obj_fease, torch.tensor([0])])
                else:
                    train_x_fease = torch.cat([train_x_fease, new_x])
                    train_obj_fease = torch.cat([train_obj_fease, torch.tensor([1])])
                    train_x_hvkgList[obj_ind] = torch.cat(
                        [train_x_hvkgList[obj_ind], new_x], dim=0
                    )
                    train_obj_hvkgList[obj_ind] = torch.cat(
                        [train_obj_hvkgList[obj_ind], new_obj.view(1, 1)], dim=0
                    )

            self.refVector = 0.5 * torch.min(train_obj_hvkg, dim=0).values

            logger.info(f"Reference point: {self.refVector}")

            mll_hvkg, model_hvkg, feaseLikelihood, feaseModel = self.initialize_model(
                train_x_hvkgList, train_obj_hvkgList, train_x_fease, train_obj_fease
            )
            fit_gpytorch_mll(mll_hvkg)

            # compute hypervolume

            hv, features, targets, stddv = self.get_model_identified_hv_maximizing_set(
                model=model_hvkg
            )
            hvs_hvkg.append(hv)
            logger.info(f"New Estimated Hypervolume: {hv}")

            t1 = time.monotonic()
            if verbose:
                logger.info(f"Iteration {iteration}, Iteration time = {t1-t0:>4.2f}.")

            # for each list in train_objv_hvkg_list, save the list as a text file
            for i, train_objv_hvkg in enumerate(train_obj_hvkgList):
                try:
                    np.savetxt(
                        f"objtv{i}/train_obj_hvkg_{iteration}.txt",
                        train_objv_hvkg.cpu().numpy(),
                        delimiter=",",
                    )
                except FileNotFoundError:
                    subprocess.run(
                        ["mkdir", f"objtv{i}"],
                        cwd="/home/bm424/Projects/sandTrapShapeOptBenchmarking/HVKG/",
                    )
                    np.savetxt(
                        f"objtv{i}/train_obj_hvkg_{iteration}.txt",
                        train_objv_hvkg.cpu().numpy(),
                        delimiter=",",
                    )

            for i, train_x_hvkg in enumerate(train_x_hvkgList):
                np.savetxt(
                    f"objtv{i}/train_x_hvkg_{iteration}.txt",
                    train_x_hvkg.cpu().numpy(),
                    delimiter=",",
                )

            np.savetxt("train_x_fease", train_x_fease.cpu().numpy())
            np.savetxt("train_obj_fease", train_obj_fease.cpu().numpy())

            iteration += 1
            np.savetxt(
                f"HVKG/HVKGModelParetoFronts/features/featuresIter{iteration}.txt",
                features.cpu().numpy(),
            )
            np.savetxt(
                f"HVKG/HVKGModelParetoFronts/targets/targetsIter{iteration}.txt",
                targets.cpu().numpy(),
            )
            np.savetxt(
                f"HVKG/HVKGModelParetoFronts/uncertainties/stdIter{iteration}.txt",
                stddv.cpu().numpy(),
            )

            # active_algos = {k for k, v in total_cost.items() if v < self.COST_BUDGET}
