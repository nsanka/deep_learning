# Import Modules
import numpy as np
import matplotlib.pyplot as plt
from IPython import display

# Draw a line between two points
def draw(x1, x2):
    ln = plt.plot(x1, x2)
    # Fix the y-axis limits to keep it fixed image
    plt.ylim(0, top_region[:,1].max())
    # For Interactive Terminal, Clear Previous Plot including scatter plot
    display.clear_output(wait=True)
    plt.pause(0.0001)
    # For Terminal, Remove previously drawn line
    ln[0].remove()

# Define sigmoid activation function
def sigmoid(score):
    return 1/(1+np.exp(-score))

# Calculate cross entroy error: -sigma(y*ln(p) + (1-y)*ln(1-p))/N
def calculate_error(line_parameters, points , y):
    n = points.shape[0]
    p = sigmoid(points*line_parameters)
    cross_entropy = -(1/n)*(np.log(p).T*y + np.log(1-p).T*(1-y))
    return cross_entropy

# Linear Regression using Gradient Descent: w1*x1 + w2*x2 + b = 0
def gradient_descent(line_parameters, points, y , alpha):
    n = points.shape[0]
    for i in range(2000):
        # Calulate activation function value for each point
        p = sigmoid(points*line_parameters)
        # Calculate gradients using learning rate alpha: sigma(x_i*(a(x)-y))*alpha/N
        gradient = points.T*(p-y)*(alpha/n)
        # Update weights
        line_parameters = line_parameters - gradient
        w1 = line_parameters.item(0)
        w2 = line_parameters.item(1)
        b = line_parameters.item(2)
        # Find min and max points on x-axis to plot
        x1 = np.array([points[:,0].min(), points[:,0].max()])
        # Calculate correspoinding y-axis coordinates using updated weights
        x2 = -b/w2 + (x1*(-w1/w2))
        # Plot line
        draw(x1,x2)
    return x1, x2

# Define number of points
n_pts = 100
np.random.seed(0)
# Constant term in the equation
bias = np.ones(n_pts)
# First point set using normal distribution x_avg = 10, y_avg = 12, std =2
top_region = np.array([np.random.normal(10,2,n_pts), np.random.normal(12,2,n_pts), bias]).T
# Second point set using normal distribution x_avg = 5, y_avg = 6, std =2
bottom_region = np.array([np.random.normal(5,2, n_pts), np.random.normal(6,2, n_pts), bias]).T
# Vertically stack both point sets
all_points = np.vstack((top_region, bottom_region))

# Initialize weights to zero
line_parameters = np.matrix([np.zeros(3)]).T
# Define the output labels as 0 for first point set and 1 for second point set
y = np.array([np.zeros(n_pts), np.ones(n_pts)]).reshape(n_pts*2, 1)

# Plot points
_, ax = plt.subplots(figsize=(4,4))
ax.scatter(top_region[:,0], top_region[:,1], color='r')
ax.scatter(bottom_region[:,0], bottom_region[:,1], color='b')
# Get two point coordinates of optimal line which separates the two point sets
x1, x2 = gradient_descent(line_parameters, all_points, y , 0.06)
# Plot the Final Line
ax.plot(x1, x2)
# Show the plot
plt.show()