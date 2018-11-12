def alpha_rec(num_timesteps, initialization, A, emission_prob, data):
    """
    Implements the alpha recursion for HMM. Using dynamic programming approach instead of direct recursion.

    Arguments:
        num_timesteps : Specifies the number of alpha_t values to compute from initialization. Length of   
                        returned "alphas" vector.
        initialization: Initial a_t vector. If starting at timestep zero, this is a multinomial on the
                        possible values of the latent variable.
        A : State transition matrix of size num_states x num_states (of latent variable). Assuming homogeneous 
            across timesteps.
        emission_prob: Function which computes the probability of data given latent variable at given timestep.
        data: array of observed emissions of length num_timesteps.
    Returns:
        alphas : Array of computed alpha values at each timestep. Length num_timesteps.

    """
    pass

def beta_rec(num_timesteps, initialization = 1, A, emission_prob):
    """
    Implements the beta recursion for HMM. Using dynamic programming approach instead of direct recursion.

    Arguments: 
        num_timesteps : Specifies the number of beta_t values to compute from initialization. Length of   
                        returned "betas" vector.
        initialization: Initial b_t vector. If starting at timestep T (end of sequence), this is a vector of ones of the
                        length of the possible values of the latent variable.
        A : State transition matrix of size num_states x num_states (of latent variable). Assuming homogeneous 
            across timesteps.
        emission_prob: Function which computes the probability of data given latent variable at given timestep.
        data: array of observed emissions of length num_timesteps (or num_timesteps - 1, not needed at final timestep).



    """
    pass
