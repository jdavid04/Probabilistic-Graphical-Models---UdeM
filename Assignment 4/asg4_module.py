import matplotlib.pyplot as plt
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
        self.k = len(pi)

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
        alphas[0] = (self.pi * self.emission_prob(data[0]))
        self.abnormalizers = [np.sum(alphas[0])]
        alphas[0] = alphas[0]/self.abnormalizers[0]
        

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
            betas[t] = ((betas[t+1] * self.emission_prob(data[t+1])).T @ self.transition_matrix).T
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

    def predict_smoothing(self, t):
        """
        Predict smoothing distribution over z_t given observed output sequence.
        Returns an array of shape (4,1).
        """
        return (self.alphas[t]*self.betas[t])/np.sum(self.alphas[t]*self.betas[t])

    def predict_edge_marginal(self, t, data):
        """
        Computes edge marginal probabilities at a given timestep t. 
        Returns a 4x4 matrix.

        """
        edge = np.zeros(shape = (self.k,self.k))
        edge = (((self.emission_prob(data[t+1])*self.betas[t+1]) @ self.alphas[t].T) * self.transition_matrix)/(np.exp(self.norm_log_likelihood(data))*data.shape[0])
        return edge

    def norm_log_likelihood(self, data):
               
        return np.sum(np.log(self.abnormalizers))/data.shape[0]
    def fit(self, data, max_iter = 50, compute_likelihoods = True, test_data = None):
        """
        Implements the EM algorithm to learn the parameters of the model. Assumes fixed initialization.
        """

        #initialization
        if compute_likelihoods : 
            train_likelihoods = []
            if test_data is not  None: test_likelihoods = []

        self.transition_matrix = np.array([[1/2, 1/6,1/6,1/6],
               [1/6,1/2,1/6,1/6],
               [1/6,1/6,1/2,1/6],
               [1/6,1/6,1/6,1/2]])
        self.pi = np.array([1/4]*4).reshape(4,1)
        self.mu_vec = np.array([[-2.0344,4.1726], [3.9779,3.7735], [3.8007,-3.7972], [-3.0620,-3.5345]])
        sigma1 = [[2.9044, 0.2066], [0.2066, 2.7562]]
        sigma2 = [[0.2104, 0.2904], [0.2904, 12.2392]]
        sigma3 = [[0.9213, 0.0574], [0.0574, 1.8660]]
        sigma4 = [[6.2414, 6.0502], [6.0502, 6.1825 ]]
        self.sigma_vec = np.array([sigma1,sigma2,sigma3,sigma4])
        

        converged = False
        iter = 0
        self.tau = np.zeros(shape = (data.shape[0], self.k))
        self.zeta = np.zeros(shape = (data.shape[0]-1, self.k, self.k))

        while (not converged) and iter < max_iter:
            self.alpha_rec(data)
            self.beta_rec(data)
            #E step
            #for each data point, compute probabilities of latent variables
            for t in range(data.shape[0]):
                #compute posterior p(Z|X,theta)
                self.tau[t] = self.predict_smoothing(t).reshape(4)
            for t in range(data.shape[0]- 1):
                #Compute edge marginals
                self.zeta[t] = self.predict_edge_marginal(t,data)

            #update parameters (M step)
            new_pi = self.tau[0].reshape(4,1)
            if not np.linalg.norm(new_pi-self.pi) < 0.01 : 
                converged = False
                
            new_matrix = np.zeros(shape = (self.k, self.k))
            for t in range(len(self.zeta)):
                new_matrix+=self.zeta[t]
            new_matrix = new_matrix/np.sum(new_matrix,axis=0)        
            if converged and not np.linalg.norm(new_matrix - self.transition_matrix) < 0.01 : converged = False
            

            new_mu = (self.tau.T @ data)/np.sum(self.tau, axis = 0)[:,np.newaxis]
            if converged and not np.linalg.norm(new_mu-self.mu_vec) < 0.01 : converged = False
            

            new_sigma = np.zeros(shape = self.sigma_vec.shape)
            for k in range(self.k):
                new_sigma[k] = (self.tau[:,k, np.newaxis] * (data - new_mu[k,:])).T @ (data - new_mu[k,:]) / np.sum(self.tau[:,k])
            if converged and not np.linalg.norm(new_sigma-self.sigma)<0.01 : converged = False
            
            
            self.pi = new_pi
            self.transition_matrix = new_matrix
            self.mu_vec = new_mu
            self.sigma_vec = new_sigma
            if compute_likelihoods:
                train_likelihoods.append(self.norm_log_likelihood(data))
                if test_data is not None : 
                    self.alpha_rec(test_data)
                    self.beta_rec(test_data)
                    test_likelihoods.append(self.norm_log_likelihood(test_data))
            
            
            iter+=1
        
        if compute_likelihoods : 
            if test_data is not None: return train_likelihoods,test_likelihoods
            else: return train_likelihoods
        else : return None
    def viterbi_decoding(self, observations):
        T = len(observations)
        states = np.zeros(T)
        V = np.zeros(shape= (T, self.k))
        pointers = np.zeros(shape = (T,self.k))

        #initialization
        probvec = (self.emission_prob(observations[0]) * self.pi).reshape(self.k)
        V[0,:] = probvec/np.sum(probvec)
        pointers[0,:] = np.zeros(self.k)

        for t in range(1,T):
            for k in range(self.k):
                probvec = self.emission_prob(observations[t])*V[t-1,:].reshape(4,1)*self.transition_matrix[k,:].reshape(4,1)
                probvec = probvec/np.sum(probvec)
                V[t,k] = np.max(probvec)
                pointers[t,k] = np.argmax(probvec)
        
        #Compute state decoding
        states[T-1] = np.argmax(V[T-1,:])
        for t in range(T-1,0,-1):
            states[t-1] = pointers[t,int(states[t])]
        return states

def plot_hmm_clustering(data,labels, model, title, figure_path,contours = True,scale=1,n_points=200):
    xlim = min(data[:,0])*scale, max(data[:,0]*scale)
    ylim = min(data[:,1]*scale), max(data[:,1]*scale)

    xx = np.linspace(xlim[0],xlim[1], n_points)
    yy = np.linspace(ylim[0],ylim[1], n_points)

    X,Y = np.meshgrid(xx,yy)
    points = np.transpose(np.array([X.flatten(), Y.flatten()]))
    
    
    fig = plt.figure(figsize = (10,10))
    plt.title(title)
    plt.scatter(data[:,0], data[:,1], c = labels)
    plt.scatter(model.mu_vec[:,0], model.mu_vec[:,1], color = 'red', marker='+', s = 100)
    if contours == True:
        for cluster in range(4):
            CS = plt.contour(X,Y, np.reshape([model.density(point,model.mu_vec[cluster], model.sigma_vec[cluster])for point in points], (n_points,n_points)))
        plt.clabel(CS, inline=1, fontsize=10)
    fig.savefig(figure_path)
    plt.show()