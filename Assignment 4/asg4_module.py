import numpy as np
class HMM:
    def __init__(self, transition_matrix, pi, mu_vec, sigma_vec):
        self.transition_matrix = transition_matrix
        self.pi = pi
        self.mu_vec = mu_vec
        self.sigma_vec = sigma_vec
        self.alphas = None
        self.betas = None
        self.gammas = None
        self.abnormalizers = None

    def density(self,x,mean,sigma, cov_isotropic = False):
        if cov_isotropic == True: return self.iso_density(x,mean,sigma)
        else: return self.gen_density(x, mean ,sigma)

    def iso_density(self, x , mean, sigma):
        """
        Compute probability density of vector x given an isotropic gaussian.

        """
        d = len(x)
        return np.exp((-0.5*np.sum((x-mean)**2))/(sigma**2))/(sigma**d * (np.pi*2)**(d/2))
    
    def gen_density(self, x, mean, sigma):
        """
        Compute probability density of vector x given an isotropic gaussian.
        """
        d = len(x)
        return np.exp(-0.5*(x-mean).T @ np.linalg.inv(sigma) @ (x-mean) ) / ((np.pi*2)**(d/2) * np.sqrt(np.linalg.det(sigma)))
    def emission_prob(self,observation):
        """
        Computes the emission probabilities of an observation under a latent state. 
        Here we are assuming a GMM emission probability with four mixture components, so 
        we will output four emission probabilities for a single observation.

        Arguments: observation : A single emission under the HMM model.
        Returns: Vector of emission probability for each mixture component. 4-dimensional numpy array.
        """
        prob = []
        for i in range(4):
            prob.append(self.density(observation, self.mu_vec[i], self.sigma_vec[i]))
        return np.array(prob).reshape(4,1)
    def alpha_rec(self, data):
        """
        Implements the alpha recursion for HMM. Using dynamic programming approach instead of direct recursion. Note that we are normalizing the messages,
        thus we are simultaneously computing and storing the filtering distributions instead of the regular alphas.
        Important variables:
                num_timesteps : Specifies the number of alpha_t values to compute from initialization. Length of   
                                    returned "alphas" vector.
                pi: Initial a_t vector. If starting at timestep zero, this is a multinomial on the
                                    possible values of the latent variable.
                transition_matrix : State transition matrix of size num_states x num_states (of latent variable). Assuming homogeneous 
                        across timesteps.
                emission_prob: Function which computes the probability of data given latent variable at given timestep. Again
                                assuming homogeneity across timesteps.
        Arguments:
            data: array of observed emissions.
        Returns:
            alphas : Array of computed alpha vectors at each timestep. Length num_timesteps.

        """
        num_timesteps = data.shape[0]
       
        alphas = [None]*num_timesteps
        alphas[0] = self.pi
        self.abnormalizers = [1.0]
        

        for t in range(1,num_timesteps):
            alphas[t] = self.transition_matrix @ alphas[t-1] * self.emission_prob(data[t])
            #Now normalize for numerical stability
            self.abnormalizers.append(np.sum(alphas[t]))
            alphas[t] = alphas[t]/self.abnormalizers[t]
        
        self.alphas = alphas
        return alphas

    def beta_rec(self, data, initialization = np.ones(shape = (4,1))):
        
        """
        Implements the beta recursion for HMM. Using dynamic programming approach instead of direct recursion.
        Important variables:
            A : State transition matrix of size num_states x num_states (of latent variable). Assuming homogeneous 
                across timesteps.
            emission_prob: Function which computes the probability of data given latent variable at given timestep. Again
                            assumes homegeneity across timesteps.
        Arguments: 
            initialization: Initial b_t vector. If starting at timestep T (end of sequence), this is a vector of ones of the
                            length of the possible values of the latent variable.
            data: array of observed emissions
        Returns:
            betas : Array of computed beta vectors at each time step.
        """
        num_timesteps = data.shape[0]
        betas = [None]*num_timesteps
        betas[-1] = initialization

        for t in range(num_timesteps-2, -1,-1):
            betas[t] = self.transition_matrix @ betas[t+1] * self.emission_prob(data[t+1])
            #Normalize for numerical stability
            betas[t] = betas[t]/self.abnormalizers[t+1]
        
        self.betas = betas
        return betas

    def gamma_rec(self):
        gammas = np.zeros(shape = (len(self.alphas),4,1))
        gammas[-1] = self.alphas[-1]
        
        for t in range(gammas.shape[0]-2,-1,-1):
            numer = self.transition_matrix @ self.alphas[t]
            gammas[t] = (numer/np.sum(numer))*(gammas[t+1])
        
        self.gammas = gammas
        return gammas

    def predict_smoothing(self, t, data = None):
        """
        Predict smoothing distribution over z_t given observed output sequence.
        """
        
        if data is None: #Should have already computed alpha-beta recursions
            if self.alphas is None or self.betas is None : raise TypeError("Need to specify data for inference!")
            else: #all good, compute smoothing
                return self.alphas[t]*self.betas[t]/np.sum(self.alphas[t]*self.betas[t])

        else:
            if self.alphas is None or self.betas is None :
                self.alphas = self.alpha_rec(data)
                self.betas = self.betas_rec(data)
            return self.alphas[t]*self.betas[t]/np.sum(self.alphas[t]*self.betas[t])