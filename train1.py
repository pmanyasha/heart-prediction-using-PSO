import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import pyswarms as ps

# Load the dataset
data = pd.read_csv('heart.csv')

# Adjust the column name for the target
target_column = 'DEATH_EVENT'  # Ensure this is your target column

# Define features and target
X = data.drop(target_column, axis=1)
y = data[target_column]

# Function to optimize
def f_per_particle(m, alpha):
    """ Computes for the objective function per particle
    
    Inputs
    ------
    m : numpy.ndarray
        Binary mask that can be used to select features
    alpha : float
        Penalty factor to adjust the feature count's effect
    
    Returns
    -------
    float
        Computed objective function
    """
    total_features = X.shape[1]
    # Apply mask to features
    X_subset = X.iloc[:, m > 0.5]
    
    # Split data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X_subset, y, test_size=0.2, random_state=42)
    
    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize classifier
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)
    P = accuracy_score(y_test, model.predict(X_test_scaled))
    
    # Calculate objective
    j = (alpha * (1.0 - P) + (1.0 - alpha) * (1 - (X_subset.shape[1] / total_features)))
    return j

def f(x, alpha=0.88):
    """ Higher-level method to do PSO across all particles
    
    Inputs
    ------
    x : numpy.ndarray
        The swarm that will perform the search
    alpha : float
        Penalty factor to adjust the feature count's effect
    
    Returns
    -------
    numpy.ndarray
        Computed objective function
    """
    n_particles = x.shape[0]
    j = [f_per_particle(x[i], alpha) for i in range(n_particles)]
    return np.array(j)

# Initialize swarm
options = {'c1': 0.5, 'c2': 0.3, 'w':0.9, 'k': 20, 'p':2}
dimensions = X.shape[1]  # Number of features
optimizer = ps.discrete.BinaryPSO(n_particles=30, dimensions=dimensions, options=options)

# Perform optimization
cost, pos = optimizer.optimize(f, iters=25)

# Apply selected features
selected_features = X.columns[pos > 0.5]
print("Selected Features:", selected_features)

# Redefine X with selected features
X_selected = X[selected_features]

# Continue as before but with X_selected
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the models
models = {
    "Logistic Regression": LogisticRegression(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Support Vector Machine": SVC(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier()
}

# Train and evaluate the models, collect accuracy and execution time for comparison
accuracy_scores = []
execution_times = []
model_names = []

for name, model in models.items():
    start_time = time.time()
    model.fit(X_train_scaled, y_train)
    elapsed_time = time.time() - start_time
    execution_times.append(elapsed_time)

    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores.append(accuracy)
    model_names.append(name)

    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{name} Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()

# Plot accuracy and execution time comparisons
plt.figure(figsize=(10, 5))
plt.bar(model_names, accuracy_scores, color='lightblue')
plt.title('Model Accuracy Comparison with Selected Features')
plt.ylabel('Accuracy Score')
plt.xlabel('Models')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(10, 5))
plt.bar(model_names, execution_times, color='lightgreen')
plt.title('Model Execution Time Comparison with Selected Features')
plt.ylabel('Execution Time (seconds)')
plt.xlabel('Models')
plt.xticks(rotation=45)
plt.show()
