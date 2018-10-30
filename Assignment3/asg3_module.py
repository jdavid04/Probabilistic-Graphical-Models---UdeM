import numpy as np
from scipy.stats import multivariate_normal


class KMeans:
    def __init__(self, k = 4, max_iter = 500):
        self.k = k
        self.max_iter = max_iter
    def cluster(self, data):
        """
        Implements kmeans clustering algorithm.

        Arguments: data - numpy ndarray
        """
        self.train = data
        #Initialize centroids
        self.centroids = data[np.random.choice(range(data.shape[0]), size = self.k, replace = False),:]
        self.cluster_map = np.zeros(shape = (data.shape[0], self.k))

        #Repeat until convergence (or max_iter)
        converged = False
        iter = 0
        while (not converged) and iter < self.max_iter:
            converged = True
            #for each data point, compute nearest centroid (E step)
            for i in range(data.shape[0]):
                #find nearest centroid and set corresponding index in matrix to 1
                self.cluster_map[i,:] = np.zeros(self.k)
                self.cluster_map[i,  np.argmin(np.linalg.norm(data[i,:] - self.centroids, axis =1))] = 1
            
            #update centroids (M step)
            for i in range(self.k):
                #Indices of non zero elements for cluster i
                indices = np.nonzero(self.cluster_map[:,i])[0]
                new_centroid = np.mean(data[indices,:], axis = 0)

                #Check convergence
                if not np.array_equal(new_centroid,self.centroids[i,:]) : converged = False
                self.centroids[i,:] = new_centroid
            iter+=1

        

    def get_distortion_measure(self):
        """
        Computes and returns distortion measure (objective minimized by kmeans). Must be called after cluster method. 
        """
        return np.sum(np.linalg.norm(self.train - self.centroids[np.nonzero(self.cluster_map)[1]],axis =1)**2)


class GMM:
    def __init__(self, k = 4, max_iter = 100, cov_isotropic = True):
        self.k = k
        self.max_iter = max_iter
        self.cov_isotropic = cov_isotropic
    
    def density(self,x,mean,sigma):
        if self.cov_isotropic == True:return self.iso_density(x,mean,sigma)
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

        
    def expectation(self, x):
        """
        Gives posterior probability for all z given a point x.
        """
       
        posterior_prop = [self.density(x, mean = self.mu[j], sigma = self.sigma[j]) * self.pi[j] for j in range(self.k)]
        return posterior_prop/np.sum(posterior_prop)
    
    def predict(self, x):
        """
        Compute most likely cluster (latent variable) for data point x. Parameters are assumed to be fit.
        """
        if len(x.shape) == 1:
                pred = np.argmax(self.expectation(x))
        else:
            pred = []
            for p in x:
                pred.append(np.argmax(self.expectation(p)))
            pred = np.array(pred)
        return pred

    
    def predict_proba_cluster(self, j, x):
            """
            Predicts probability of Z_j = 1 for some cluster j and data point x. Parameters are assumed to be fit.
            """
            if len(x.shape) == 1:
                prob = self.expectation(x)[j]
            else:
                prob = []
                for p in x:
                    prob.append(self.expectation(p)[j])
                prob = np.array(prob)
            return prob

    def cluster(self, data):
        """
        Implements GMM assuming the the gaussians are isotropic. 

        Arguments: data - numpy ndarray
        """
        
        #Initialize parameters

        #Initialize means with kmeans centroids
        kmeans = KMeans(k = self.k)
        kmeans.cluster(data)
        self.mu = kmeans.centroids

        #Initialize sigma from kmeans as the variance of distance of data points w.r.t their assigned centroids
        if self.cov_isotropic == True:
            self.sigma = np.zeros(self.k)
            for j in range(self.k):
                idx = np.nonzero(kmeans.cluster_map[:,j])[0]
                self.sigma[j] = np.sqrt(np.var(np.linalg.norm(data[idx,:] - kmeans.centroids[j],axis =1)**2))
        else:
            self.sigma = np.zeros(shape = (self.k, data.shape[1], data.shape[1]))
            for j in range(self.k):
                idx = np.nonzero(kmeans.cluster_map[:,j])[0]
                self.sigma[j] = np.cov(data[idx,:].T)
                

        #Initialize pi as proportions in kmeans clusters.
        self.pi = np.mean(kmeans.cluster_map, axis = 0)
        

        #Repeat until convergence (or max_iter)
        converged = False
        self.tau = np.zeros(shape = (data.shape[0], self.k))
        iter = 0
        while (not converged) and iter < self.max_iter:
            
            converged = True
            #for each data point, compute probabilities of latent variables (E step)
            for i in range(data.shape[0]):
                #compute posterior p(Z|X,theta)
                self.tau[i,:] = self.expectation(data[i,:])
            
            #update parameters (M step)
            new_pi = np.mean(self.tau,axis = 0)
            if not np.linalg.norm(new_pi-self.pi) < 0.0001 : 
                converged = False
                

            new_mu = (self.tau.T @ data)/np.sum(self.tau, axis = 0)[:,np.newaxis]
            if converged and not np.linalg.norm(new_mu-self.mu) < 0.0001 : 
                converged = False

            new_sigma = np.zeros(shape = self.sigma.shape)
            if self.cov_isotropic == True:
                d = data.shape[1]
                for j in range(self.k):
                    new_sigma[j] = np.sqrt(np.sum(self.tau[:,j] * np.linalg.norm(data - new_mu[j,:],axis = 1)**2)/(d*np.sum(self.tau[:,j])))
            else:
                for j in range(self.k):
                    new_sigma[j] = (self.tau[:,j, np.newaxis] * (data - new_mu[j,:])).T @ (data - new_mu[j,:]) / np.sum(self.tau[:,j])
            if converged and not np.linalg.norm(new_sigma-self.sigma)<0.0001 : converged = False
            
            self.pi = new_pi
            self.mu = new_mu
            self.sigma = new_sigma
            

            iter+=1
   
        