import numpy as np
import matplotlib.pyplot as plt
# Importing the class from your src folder
from src.regression.linear_regression import MultipleLinearRegression

# 1. Prepare Data
X_train = np.array([[2104, 5, 1, 45], 
                    [1416, 3, 2, 40], 
                    [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])

# 2. Initialize the Model
model = MultipleLinearRegression()

# 3. Training Settings
alpha = 5.0e-7
iterations = 1000

# 4. Run Training (Fit)
print("Starting Gradient Descent...")
w_final, b_final, J_hist = model.fit(X_train, y_train, alpha, iterations)

# 5. Display Final Parameters
print(f"\nFinal parameters: b = {b_final:0.2f}, w = {w_final}")

# 6. Make Predictions and Compare
print("\nPredictions vs Targets:")
predictions = model.predict(X_train)
for i in range(len(y_train)):
    print(f"Prediction: {predictions[i]:0.2f}, Target: {y_train[i]}")

# 7. Visualization
plt.plot(J_hist)
plt.title("Cost vs. Iteration")
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.show()