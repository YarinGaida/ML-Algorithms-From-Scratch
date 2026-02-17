import numpy as np
import copy
import math

class MultipleLinearRegression:
    def __init__(self):
        self.w = None
        self.b = None

    def predict(self, X):
        """
        Single prediction or batch prediction using linear regression.
        Args:
          X (ndarray (m,n)): Data, m examples with n features
        Returns:
          p (ndarray (m,)): Predictions
        """
        return np.dot(X, self.w) + self.b

    def compute_cost(self, X, y, w, b):
        """
        Computes the cost (mean squared error).
        """
        m = X.shape[0]
        cost = 0.0
        for i in range(m):                                
            f_wb_i = np.dot(X[i], w) + b           
            cost = cost + (f_wb_i - y[i])**2       
        cost = cost / (2 * m)                         
        return cost

    def compute_gradient(self, X, y, w, b):
        """
        Computes the gradient for linear regression.
        """
        m, n = X.shape
        dj_dw = np.zeros((n,))
        dj_db = 0.

        for i in range(m):                             
            err = (np.dot(X[i], w) + b) - y[i]   
            for j in range(n):                         
                dj_dw[j] = dj_dw[j] + err * X[i, j]    
            dj_db = dj_db + err                        
        dj_dw = dj_dw / m                                
        dj_db = dj_db / m                                
            
        return dj_db, dj_dw

    def fit(self, X, y, alpha, num_iters):
        """
        Performs batch gradient descent to learn w and b.
        """
        m, n = X.shape
        self.w = np.zeros((n,))
        self.b = 0.
        J_history = []
        
        for i in range(num_iters):
            # Calculate the gradient
            dj_db, dj_dw = self.compute_gradient(X, y, self.w, self.b)   

            # Update Parameters
            self.w = self.w - alpha * dj_dw               
            self.b = self.b - alpha * dj_db               
          
            # Save cost J at each iteration
            if i < 100000:      
                J_history.append(self.compute_cost(X, y, self.w, self.b))

            # Print progress
            if i % math.ceil(num_iters / 10) == 0:
                print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}")
            
        return self.w, self.b, J_history