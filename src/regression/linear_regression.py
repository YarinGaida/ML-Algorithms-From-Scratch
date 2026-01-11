import numpy as np

class LinearRegression:
    """
    Implementation of Linear Regression using the Normal Equation (Closed Form Solution).
    Based on the formula: beta = (X^T * X)^-1 * X^T * Y
    """
    def __init__(self):
        self.beta = None

    def fit(self, X, y):
        """
        Calculates the optimal beta coefficients using the Normal Equation.
        """
        # Adding bias term (intercept) is usually handled here, 
        # but per your assignment strict rules, we assume X is passed correctly.
        
        # Formula: (X^T * X)^-1 * X^T * Y
        # We use pinv (pseudo-inverse) for better stability, but inv is fine too per formula.
        self.beta = np.linalg.inv(X.T @ X) @ X.T @ y

    def predict(self, X):
        """
        Predicts target values for given input X using the learned beta.
        """
        if self.beta is None:
            raise Exception("Model has not been fitted yet.")
        return X @ self.beta

# --- החלק של ה-Main שמדגים את הפתרון לשיעורי הבית ---
if __name__ == "__main__":
    # 1. Configuration
    n_samples = 1000
    m_features = 2
    
    # 2. Generate random uniform matrix X in [0,1]
    X = np.random.rand(n_samples, m_features)
    
    # 3. Define true Beta parameters (Hardcoded for demo, or input)
    # Let's say Beta is [5, 10]
    true_beta = np.array([[5], [10]]) 
    
    print(f"True Beta parameters:\n{true_beta}")

    # 4. Generate Gaussian Noise (sigma = 1)
    sigma = 1.0
    epsilon = np.random.randn(n_samples, 1) * sigma
    
    # 5. Compute Y = X * beta + epsilon
    # Reshaping is crucial for correct matrix multiplication dimensions
    y = X @ true_beta + epsilon

    # --- Using our Class ---
    model = LinearRegression()
    model.fit(X, y)
    estimated_beta = model.beta

    print(f"\nEstimated Beta (Normal Equation):\n{estimated_beta}")
    
    # 6. Comparison
    print("\n--- Comparison ---")
    for i in range(m_features):
        diff = abs(true_beta[i][0] - estimated_beta[i][0])
        print(f"Beta[{i}]: True={true_beta[i][0]}, Est={estimated_beta[i][0]:.4f}, Diff={diff:.4f}")

    # 7. Check influence of larger noise (Sigma Increase)
    print("\n--- Increasing Noise (Sigma = 10) ---")
    sigma_large = 10.0
    epsilon_large = np.random.randn(n_samples, 1) * sigma_large
    y_large = X @ true_beta + epsilon_large
    
    model_large = LinearRegression()
    model_large.fit(X, y_large)
    
    print(f"Estimated Beta with High Noise:\n{model_large.beta}")