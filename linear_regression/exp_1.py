import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# ==========================================
# CONFIG
# ==========================================
DATA_PATH = "student_performance_dataset.csv"
TARGET_COL = "final_exam_score"
TEST_SIZE = 0.2
RANDOM_STATE = 42
LEARNING_RATE = 0.05
ITERATIONS = 1000

# ==========================================
# 1. LOAD DATA
# ==========================================
df = pd.read_csv(DATA_PATH)

# Automatically select numeric columns only (EXCEPT target)
feature_cols = df.select_dtypes(include=[np.number]).columns.drop(TARGET_COL)

# Fill missing values with median (generic)
df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())

X = df[feature_cols].values
y = df[TARGET_COL].values.reshape(-1, 1)

# ==========================================
# 2. TRAIN-TEST SPLIT
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

# ==========================================
# 3. STANDARDIZATION (GENERIC)
# ==========================================
mean = X_train.mean(axis=0)
std = X_train.std(axis=0) + 1e-8   # avoid divide-by-zero

def scale(X):
    return (X - mean) / std

X_train = scale(X_train)
X_test = scale(X_test)

# Add bias term
X_train = np.c_[np.ones((X_train.shape[0], 1)), X_train]
X_test = np.c_[np.ones((X_test.shape[0], 1)), X_test]

# ==========================================
# 4. GRADIENT DESCENT
# ==========================================
def compute_cost(X, y, theta):
    m = len(y)
    return np.sum((X @ theta - y) ** 2) / (2 * m)

def gradient_descent(X, y, theta, lr, iters):
    m = len(y)
    cost_history = []

    for i in range(iters):
        theta -= (lr / m) * (X.T @ (X @ theta - y))
        cost_history.append(compute_cost(X, y, theta))

        if i % 100 == 0:
            print(f"Iter {i}: Cost = {cost_history[-1]:.4f}")

    return theta, cost_history

# ==========================================
# 5. TRAIN MODEL
# ==========================================
theta = np.zeros((X_train.shape[1], 1))

theta, cost_history = gradient_descent(
    X_train, y_train, theta, LEARNING_RATE, ITERATIONS
)

# ==========================================
# 6. EVALUATION (GENERIC)
# ==========================================
def evaluate(X, y, theta, name="Set"):
    preds = X @ theta
    mse = np.mean((preds - y) ** 2)
    rmse = np.sqrt(mse)

    ss_res = np.sum((y - preds) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot

    print(f"\n{name}")
    print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")

    return preds

evaluate(X_train, y_train, theta, "Train")
evaluate(X_test, y_test, theta, "Test")

# ==========================================
# 7. PLOT COST
# ==========================================
plt.plot(cost_history)
plt.title("Gradient Descent Convergence")
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.grid(True)
plt.show()