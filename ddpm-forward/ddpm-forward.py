import numpy as np

def get_alpha_bar(betas: np.ndarray) -> np.ndarray:
    """
    Compute cumulative product of (1 - beta).
    """
    # YOUR CODE HERE
    alphas = 1.0 - betas
    return np.cumprod(alphas)

def forward_diffusion(
    x_0: np.ndarray,
    t: int,
    betas: np.ndarray
) -> tuple:
    """
    Sample x_t from q(x_t | x_0).
    """
    # YOUR CODE HERE
    alpha_bar = get_alpha_bar(betas)
    # print(alpha_bar)
    noise = np.random.standard_normal(size=x_0.shape)
    # print(noise.shape)
    return (np.sqrt(alpha_bar[t])*x_0 + np.sqrt(1-alpha_bar[t])*noise,noise)
    # return tuple(0,0)
  
