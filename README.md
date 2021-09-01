```
z : TxD // latent trajectory, time by latent dimension
x : TxN // observed trajectory, time by observed dimension. observed dimension is the number of time series that we have per patient

A : NxD // loading matrix. these are the coefficients that connect the latent to the observed space

z_:,d ~ GP(kernel_d) // each of the latent time series is drawn from a latent Gaussian process
x = z A^T + Gaussian noise

It is possible to show that:
vec(x) = (A (*) I) vec(z)
where (*) is the kronecker product, I is the identity matrix

As a consequence, the marginal distribution on vec(x) is also a Gaussian process with the kernel:

(A (*) I) K (A (*) I)^T

where z follows a Gaussian process with kernel K:
K = diagonal_block_matrix(kernel_1, ..., kernel_D)


We assume that the kernel over the latent Gaussian processes is the squared exponential kernel. Without loss of generality we can assume that the magnitude of this kernel is 1, because we can absorb the magnitude into A. Therefore the only thing that we are learning in the latent space is the characteristic time scale of the kernel. So there is only a single learned parameter for each latent signal
```  