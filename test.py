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
    def __init__(self, D, N):
        super().__init__()

        self.D = D
        self.N = N

        self.A = nn.Parameter(torch.randn(N, D)*((2./N)**0.5),
                              requires_grad=True)

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
    


D = 1
N = 2
NOISE = 1e-4

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

    for l in range(0):
        plt.plot(X, Z.numpy()[:,l],
                 c="b")
    
    d_observed_signals = model.observed_distribution(X, NOISE)
    v_observed_signals = d_observed_signals.sample()
    observed_signals = reshape(v_observed_signals, T, N)
    for l in range(0):
        plt.plot(X, observed_signals.numpy()[:,l],
                 c="k")

    
    
    observed_signals = Z@model.A.T
    #v(ab)=b.t (*) I   @ v(a)
    #a=Z, b=A.T
    #v(Z@A.T)=(A (*) I)v(z)
    
    
    

    for l in range(N):
        plt.plot(X, observed_signals.detach().numpy()[:,l],
                 c="r", ls="--", label="training")

    

    model = LatentGP(D, N)
    
    
    
    
    optimizer = torch.optim.Adam(model.parameters())
    
    for step in range(1000):        
        distribution = model.observed_distribution(torch.tensor(X).float(), noise=NOISE)
        likelihood = distribution.log_prob(v_observed_signals).sum()
        loss = -likelihood/len(v_observed_signals)

        #loss += model.A.abs().sum()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        

        if step%100==0:
            print(step, '\tloss', loss.detach().numpy(),
                              'likelihood', likelihood.detach().numpy())
            print(model.A)
            print()

    print(model.A)
        
    observed_signals = distribution.sample()
    #observed_signals=v_observed_signals
    observed_signals = reshape(observed_signals, T, N)
    observed_signals = observed_signals.cpu().numpy()
    for l in range(N):
        plt.plot(X, observed_signals[:,l],
                 c="g",
                 label='predicted')
    

plt.show()
