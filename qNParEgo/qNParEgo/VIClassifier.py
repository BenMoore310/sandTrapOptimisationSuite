import torch
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy

from gpytorch.models import ExactGP
from gpytorch.likelihoods import DirichletClassificationLikelihood
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel

import torch
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.likelihoods import BernoulliLikelihood

import logging

logger = logging.getLogger(__name__)


class DirichletGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_classes):
        super(DirichletGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean(batch_shape=torch.Size((num_classes,)))
        self.covar_module = ScaleKernel(
            RBFKernel(batch_shape=torch.Size((num_classes,))),
            batch_shape=torch.Size((num_classes,)),
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def GPTrain(trainx, trainy, training_iter=100):
    # initialize likelihood and model
    # we let the DirichletClassificationLikelihood compute the targets for us
    device = torch.device("cpu")
    logger.info(f"training classifier on {device}")

    trainx = trainx.to(device)
    trainx = torch.tensor(trainx, dtype=torch.float32)

    trainy = trainy.to(device)

    # logger.info(trainx.dtype)

    likelihood = DirichletClassificationLikelihood(
        trainy, learn_additional_noise=True
    ).to(device)
    model = DirichletGPModel(
        trainx,
        likelihood.transformed_targets,
        likelihood,
        num_classes=likelihood.num_classes,
    ).to(device)

    # likelihood.noise = 1e-4
    # likelihood.noise_covar.raw_noise.requires_grad(False)

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.1
    )  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(trainx)
        # Calc loss and backprop gradients
        loss = -mll(output, likelihood.transformed_targets).sum()
        loss.backward()
        if i % 5 == 0:
            logger.info(
                "Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f"
                % (
                    i + 1,
                    training_iter,
                    loss.item(),
                    model.covar_module.base_kernel.lengthscale.mean().item(),
                    model.likelihood.second_noise_covar.noise.mean().item(),
                )
            )
        optimizer.step()

    return model.to("cpu"), likelihood.to("cpu")


def GPEval(model, likelihood, testx):
    model.eval()
    likelihood.eval()
    testx = torch.tensor(testx, dtype=torch.float32)
    with gpytorch.settings.fast_pred_var(), torch.no_grad():
        test_dist = model(testx)
        # pred_means = test_dist.loc

    pred_samples = test_dist.sample(torch.Size((1024,))).exp()
    probabilities = (pred_samples / pred_samples.sum(-2, keepdim=True)).mean(0)
    prob_Fease = probabilities[1]
    return prob_Fease
