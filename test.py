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
        self.kernel_magnitude = nn.Parameter(torch.randn(D), requires_grad=True)

    def observed_covariance(self, T):
        distances = T.unsqueeze(0) - T.unsqueeze(1)
        distances = distances**2

        distances = -distances.unsqueeze(-1)/((2*self.kernel_scale**2).unsqueeze(0).unsqueeze(0))

        K = self.kernel_magnitude * self.kernel_magnitude * distances.exp()

        
        
        matrix = torch.einsum("abd,nd,md->anbm", K, self.A, self.A)

        return matrix

    def observed_distribution(self, T, noise=None):
        sigma = self.observed_covariance(T).view(self.N * T.shape[0], self.N * T.shape[0])

        sigma = (sigma + sigma.T)/2

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
    


D = 2
N = 4

plt.figure()
attempts = 4
for attempt in range(attempts):
    plt.subplot(int(attempts**0.5),int(attempts**0.5),attempt+1)

    A = np.random.normal(np.zeros((N, D)),
                         np.ones((N, D)))

    X = np.linspace(0., 1., 40)
    T = X.shape[0]
    print(T)

    def K(X, parameters=None):
        
        return np.array([ [ parameters[0]**2 * np.exp(-(x-y)**2/(2*parameters[1]*parameters[1])) for x in X] for y in X])

    
    parameters = np.random.random((D, 2))
    latent_covariance = block_diag(*[K(X, th) for th in parameters ])

    print(latent_covariance.shape)

    vZ = np.random.multivariate_normal(np.zeros((D*T,)), latent_covariance)
    Z = np.reshape(vZ, (T,D), order="F")
    
    

    for l in range(D):
        plt.plot(X, Z[:,l],
                 c="b")
    

    block_affine = np.kron(A, np.eye(T))
    
    # for n in range(N):
    #     for d in range(D):
    #         for alpha in range(T):
    #             for beta in range(T):
    #                 if np.abs(block_affine)[n+N]:
    # print("block_affine",block_affine.shape)
    # print(1*(block_affine!=0.))
    # assert False
    

    observed_covariance = block_affine @ latent_covariance @ block_affine.T
    
    #assert np.all(np.abs(observed_covariance)>1e-3)

    for n in range(0):
        for _n in range(N):
            for t in range(T):
                for _t in range(T):
                    gt = observed_covariance[T*n+t, T*_n+_t]
                    prediction = sum( K(X, parameters[d])[t,_t] * A[n,d] * A[_n,d]
                                      for d in range(D) )
                    assert (np.abs(gt - prediction)<1e-5)
    
                    
                    import pdb; pdb.set_trace()
    
    v_observed_signals = np.random.multivariate_normal(np.zeros((N*T,)), observed_covariance)
    observed_signals = np.reshape(v_observed_signals, (T, N), order="F")
    if False:
        for l in range(N):
            plt.plot(X, observed_signals[:,l],
                     c="k")

    observed_signals = Z@A.T
    #v(ab)=b.t (*) I   @ v(a)
    #a=Z, b=A.T
    #v(Z@A.T)=(A (*) I)v(z)
    
    
    #assert np.max(np.abs(vectorize(observed_signals) - block_affine @ vZ)) < 1e-8

    
    
    

    for l in range(N):
        plt.plot(X, observed_signals[:,l],
                 c="g", ls="--")

        

    model = LatentGP(D, N)
    model.A.data = torch.tensor(A).float()
    model.kernel_scale.data = torch.tensor(parameters[:,1]).float()
    model.kernel_magnitude.data = torch.tensor(parameters[:,0]).float()
    
    optimizer = torch.optim.Adam(model.parameters())
    
    for step in range(10):
        v_observed_signals = torch.tensor(v_observed_signals).float()
        #v_observed_signals = v_observed_signals.view(T, N).view(-1)
        
        distribution = model.observed_distribution(torch.tensor(X).float(), noise=0.0001)
        likelihood = distribution.log_prob(v_observed_signals).sum()
        loss = -likelihood/len(v_observed_signals)

        #loss += model.A.abs().sum()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step%100==0: print(step, '\tloss', loss.detach().numpy(),
                              'likelihood', likelihood.detach().numpy())

    print(A)
    print(model.A)
        
    observed_signals = distribution.sample()
    #observed_signals=v_observed_signals
    observed_signals = observed_signals.view(T, N)
    observed_signals = observed_signals.cpu().numpy()
    for l in range(0):
        plt.plot(X, observed_signals[:,l],
                 c="g")
    

plt.show()
