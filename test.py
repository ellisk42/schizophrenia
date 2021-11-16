import numpy as np
from scipy.linalg import block_diag
from scipy.linalg import block_diag
import matplotlib.pyplot as plt
import sys
import torch
import torch.nn as nn
import random

np.set_printoptions(threshold=sys.maxsize)

class LatentGP(nn.Module):
    def __init__(self, D, N, inverse_beta=0.):
        """inverse_beta: regularization coefficient for L1 norm of loading matrix
        it has this name because this is equivalent to assuming that the loading matrix injuries are generated from a laplace distribution with this inverse scale"""
        super().__init__()

        self.D = D
        self.N = N

        self.inverse_beta = inverse_beta
        if inverse_beta > 0:
            self.loading_matrix_prior = torch.distributions.laplace.Laplace(torch.zeros(N, D), inverse_beta/torch.ones(N, D))
            A = self.loading_matrix_prior.sample()
        else:
            A = torch.randn(N, D)*((2./N)**0.5)
            
        self.A = nn.Parameter(A, requires_grad=True)

        self.kernel_scale = nn.Parameter(torch.randn(D), requires_grad=True)
        #self.kernel_magnitude = nn.Parameter(torch.randn(D), requires_grad=True)

    def latent_covariance(self, T):
        distances = T.unsqueeze(0) - T.unsqueeze(1)
        distances = distances**2
        distances = distances.unsqueeze(0)

        scale = self.kernel_scale**2
        scale = scale.unsqueeze(-1).unsqueeze(-1)

        magnitude = 1
        #magnitude = self.kernel_magnitude**2
        #magnitude = magnitude.unsqueeze(-1).unsqueeze(-1)        
        
        K = magnitude * (-distances/(2*scale)).exp()
        
        return torch.block_diag(*list(K))

    def latent_distribution(self, T, noise=None):
        sigma = self.latent_covariance(T)

        if noise is not None:
            sigma = sigma + torch.eye(sigma.shape[0]) * noise

        mu = torch.zeros(sigma.shape[0])
        return torch.distributions.multivariate_normal.MultivariateNormal(mu, sigma)
        
        
    def observed_covariance(self, T):
        K = self.latent_covariance(T)
        block_affine = torch.kron(self.A, torch.eye(T.shape[0]))
        return block_affine @ K @ block_affine.T
        

    def observed_distribution(self, T, noise=None):
        sigma = self.observed_covariance(T)

        if noise is not None:
            sigma = sigma + torch.eye(sigma.shape[0]) * noise

        mu = torch.zeros(sigma.shape[0])
        return torch.distributions.multivariate_normal.MultivariateNormal(mu, sigma)

    def joint_covariance(self, T, noise=None):
        K = self.latent_covariance(T)
        
        A_p = torch.cat([torch.eye(self.D), self.A], 0)
        
        block_affine = torch.kron(A_p, torch.eye(T.shape[0]))
        return block_affine @ K @ block_affine.T

    def joint_distribution(self, T, noise=None):
        s = self.joint_covariance(T)
        if noise:
            s += noise*torch.eye(s.shape[0])
            
        mu = torch.zeros(s.shape[0])
        
        return torch.distributions.multivariate_normal.MultivariateNormal(mu, s)

    def conditional_distribution(self, T, X, noise=None):
        assert X.shape[0] == T.shape[0]
        assert X.shape[1] == self.N

        td = T.shape[0] * self.D

        sigma = self.joint_covariance(T)
        if noise: sigma = sigma + torch.eye(sigma.shape[0])
        s11 = sigma[:td, :td]
        s22 = sigma[td:, td:]
        s12 = sigma[:td, td:]
        s21 = sigma[td:, :td]

        X = vectorize(X)

        mu = s12@(torch.linalg.inv(s22))@X
        s = s11 - s12@torch.linalg.inv(s22)@s21
        
        return torch.distributions.multivariate_normal.MultivariateNormal(mu, s)

    def fit(self, observed_signals, steps=1000, noise=None):
        """observed_signals: list of (time_steps, observed_data)
        think of each list element as a patient
        observed_data should be a tensor of shape [number_of_time_steps, number_of_signals], or `vectorize` of that
        """

        def maybe_vectorized(v):
            if len(v.shape) == 2:
                return vectorize(v)
            else:
                return v

        observed_signals = [(T, maybe_vectorized(X))
                            for T, X in observed_signals ]            

        optimizer = torch.optim.Adam(self.parameters())

        for step in range(steps):
            loss = 0.
            for T, X in observed_signals:
                distribution = self.observed_distribution(torch.tensor(T).float(), noise=noise)
                likelihood = distribution.log_prob(X).sum()
                loss = loss -likelihood

            if self.inverse_beta > 0:
                loss = loss - self.loading_matrix_prior.log_prob(self.A).sum()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step%100==0:
                print(step, '\tloss', loss.detach().numpy(),
                      'likelihood', likelihood.detach().numpy())
        return loss.detach().numpy()
        
        

def vectorize(a):
    if isinstance(a, np.ndarray):
        return a.flatten("F")
    if isinstance(a, torch.Tensor):
        return a.contiguous().T.contiguous().view(-1)
    assert False

def reshape(a, n, m):
    if isinstance(a, np.ndarray):
        return np.reshape(a, (n, m), order="F")
    if isinstance(a, torch.Tensor):
        return a.contiguous().view(m, n).T.contiguous()
    assert False    

def test_vector():
    for random_creator in [torch.randn, lambda m,n: np.random.random((m, n))]:
        a, b, c = random.choice(range(2,5)),random.choice(range(2,5)),random.choice(range(2,5))
        A = random_creator(a, b)
        B = random_creator(b, c)
        C = A@B

        # print(A)
        # print(vectorize(A))
        # print(reshape(vectorize(A), a, b))
        # print(vectorize(reshape(vectorize(A), a, b)))

        # assert False

        if isinstance(A, np.ndarray):
            kr = np.kron
            I = np.eye
            error = lambda X, Y: np.max(np.abs(X - Y))
            Bt = B.T
        else:
            kr = torch.kron
            I = torch.eye
            Bt = B.T.contiguous()
            error = lambda X, Y: torch.max(torch.abs(X - Y))
            
        assert error(reshape(vectorize(A), a, b), A) < 1e-5

        assert error(vectorize(C), kr(I(c), A)@vectorize(B)) < 1e-5
        assert error(vectorize(C), kr(Bt, I(a))@vectorize(A)) < 1e-5
for _ in range(100):
    test_vector()
    


D = 3
N = 5
NOISE = 1e-4

def visualize_inference(D, N):
    plt.figure()
    attempts = 4
    for attempt in range(attempts):
        plt.subplot(int(attempts**0.5),int(attempts**0.5),attempt+1)

        model = LatentGP(D, N)

        X = torch.linspace(0., 1., 40)    
        T = X.shape[0]
        print(T)    

        latent_covariance = model.latent_covariance(X)

        print(latent_covariance)

        dZ = model.latent_distribution(X, noise=NOISE)

        vZ = dZ.sample()
        Z = reshape(vZ, T, D)

        for l in range(D):
            plt.plot(X, Z.numpy()[:,l],
                     c="b", label="latent" if l==0 else None)

        observed_signals = Z@model.A.T
        #v(ab)=b.t (*) I   @ v(a)
        #a=Z, b=A.T
        #v(Z@A.T)=(A (*) I)v(z)




        for l in range(N):
            plt.plot(X, observed_signals.detach().numpy()[:,l],
                     c="r", label="observed"  if l==0 else None)

        joint_sample = model.joint_distribution(X, NOISE).sample()
        joint_sample_latent = reshape(joint_sample[:D*T], T, D)
        joint_sample_observed = reshape(joint_sample[D*T:], T, N)

        if False:
            for l in range(D):
                plt.plot(X, joint_sample_latent.numpy()[:,l],
                         ls="--", c="b", label="latent joint")
            for l in range(N):
                plt.plot(X, joint_sample_observed.numpy()[:,l],
                         ls="--", c="r", label="obs joint")

    
        
        conditional_inference = reshape(model.conditional_distribution(X, observed_signals, NOISE*1e-3).mean, T, D)
        for l in range(D):
            plt.plot(X,  conditional_inference.detach().numpy()[:,l],
                     c="g", ls="-", label="conditional"  if l==0 else None)

    plt.legend()
    plt.show()


def visualize_fitting(D, N):
    attempts = 4
    number_of_patients = 2

    ib = 1.
    
    X = torch.linspace(0., 1., 20)    
    for attempt in range(attempts):

        model = LatentGP(D, N, inverse_beta=ib)
        
        ground_truth_observations = [(X, model.observed_distribution(X, NOISE).sample())
                                     for _ in range(number_of_patients) ]

        best_loss = []
        ds = list(range(1, D+3))
        for d in ds:
            model = LatentGP(d, N, inverse_beta=ib)
            best_loss.append(model.fit(ground_truth_observations, noise=NOISE, steps=5000))

        plt.plot(ds, best_loss)
    plt.xlabel("number of latent factors")
    plt.ylabel("loss")
    plt.title(f"{N} signals, {number_of_patients} tasks (patients)")
    plt.show()

visualize_inference(2, 4)
visualize_fitting(4, 6)
