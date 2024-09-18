import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from io import BytesIO

# Define functions for gradient descent
def square_function(x):
    return x**2

def square_function_derivative(x):
    return 2*x

def cube_function(x):
    return x**3

def cube_function_derivative(x):
    return 3*x**2

def polynomial_function(x):
    return x**4 - 5*x**3 + 6*x**2

def polynomial_function_derivative(x):
    return 4*x**3 - 15*x**2 + 12*x

def gradient_descent(starting_point, learning_rate, num_iterations, derivative_function):
    x = starting_point
    path = [x]
    
    for _ in range(num_iterations):
        grad = derivative_function(x)
        x = x - learning_rate * grad
        path.append(x)
    
    return np.array(path)

# Streamlit App
st.title('Gradient Descent Visualizer')

# User input
function = st.sidebar.selectbox('Select Function', ['Square', 'Cube', 'Polynomial'])
starting_point = st.sidebar.number_input('Starting Point', value=5.0)
learning_rate = st.sidebar.number_input('Learning Rate', value=0.1)
num_iterations = st.sidebar.slider('Number of Iterations', min_value=1, max_value=100, value=50)

# Define the selected function and its derivative
if function == 'Square':
    func = square_function
    derivative_func = square_function_derivative
elif function == 'Cube':
    func = cube_function
    derivative_func = cube_function_derivative
else:
    func = polynomial_function
    derivative_func = polynomial_function_derivative

# Generate x values for plotting the function
x_vals = np.linspace(-10, 10, 400)
y_vals = func(x_vals)

# Perform gradient descent
path = gradient_descent(starting_point, learning_rate, num_iterations, derivative_func)

# State for current iteration
if 'current_iteration' not in st.session_state:
    st.session_state.current_iteration = 0

def next_iteration():
    if st.session_state.current_iteration < num_iterations - 1:
        st.session_state.current_iteration += 1

# Button to move to the next iteration
st.sidebar.button('Next Iteration', on_click=next_iteration)

# Get current iteration index
current_iter = st.session_state.current_iteration

# Create the plot
fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(x_vals, y_vals, label=f'$f(x)$', color='blue')

# Plot gradient descent path up to current iteration
ax.scatter(path[:current_iter+1], func(path[:current_iter+1]), color='red', label='Gradient Descent Steps')
ax.plot(path[:current_iter+1], func(path[:current_iter+1]), 'ro-', markersize=5)

# Plot tangent line for the current iteration
if current_iter > 0:
    x_current = path[current_iter]
    y_current = func(x_current)
    slope = derivative_func(x_current)
    y_intercept = y_current - slope * x_current
    tangent = lambda x: slope * x + y_intercept
    tangent_x_vals = np.linspace(-10, 10, 400)
    tangent_y_vals = tangent(tangent_x_vals)
    ax.plot(tangent_x_vals, tangent_y_vals, '--', color='green', alpha=0.5)

# Add labels and title
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax.set_title(f'Gradient Descent on Selected Function with Tangent Lines')
ax.legend()

# Save plot to BytesIO object and display in Streamlit
buf = BytesIO()
plt.savefig(buf, format='png')
buf.seek(0)
st.image(buf, use_column_width=True)
