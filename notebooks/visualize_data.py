import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

print("Generating synthetic data for visualization...")

np.random.seed(42)
X_class0 = np.random.normal(loc=[25, 5, 40000], scale=[5, 2, 10000], size=(100, 3))
X_class1 = np.random.normal(loc=[45, 15, 75000], scale=[8, 4, 15000], size=(100, 3))

colors = ['blue']*100 + ['red']*100
X = np.vstack([X_class0, X_class1])

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')


ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=colors, alpha=0.6)

ax.set_xlabel('Age')
ax.set_ylabel('Time_on_Site (min)')
ax.set_zlabel('Estimated_Income ($)')
ax.set_title('3D View of Synthetic Ad Click Data (Class 0=Blue, Class 1=Red)')

print("Data generated. Displaying 3D plot...")
plt.show()