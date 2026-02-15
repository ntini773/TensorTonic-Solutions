import numpy as np
def get_alpha_bar(betas: np.ndarray):
    alphas = 1.0 -betas
    return np.cumprod(alphas)
def reverse_step(
    x_t: np.ndarray,
    t: int,
    epsilon_pred: np.ndarray,
    betas: np.ndarray
) -> np.ndarray:
    """
    Perform one reverse diffusion step.
    """
    # YOUR CODE HERE
    alpha_bars = get_alpha_bar(betas)
    alpha_bar_t = alpha_bars[t]
    alpha_t = 1.0 - betas[t]
    beta_t = betas[t]
    term_to_subtract = (beta_t/np.sqrt(1-alpha_bar_t))*epsilon_pred
    mu = (1.0/np.sqrt(alpha_t))*(x_t-term_to_subtract)
    if t > 1:
      noise = np.random.standard_normal(size=x_t.shape)
      
      return mu + np.sqrt(beta_t)*noise
    else: 
      return mu
  
    
