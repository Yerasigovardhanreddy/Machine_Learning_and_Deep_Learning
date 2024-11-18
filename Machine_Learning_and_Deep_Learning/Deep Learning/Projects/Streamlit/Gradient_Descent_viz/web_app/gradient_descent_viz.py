# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import sympy as sp

# Pre-defined functions and their derivatives
def square(x): 
    return x**2

def derivative_square(x):
    return 2*x

def cube(x): 
    return x**3

def derivative_cube(x):
    return 3 * x**2

def sin(x):
    return np.sin(x)

def derivative_sin(x):
    return np.cos(x)
    
def inverse(x):
    return 1/x

def derivative_inverse(x):
    return - (1 / x **2)

def poly(x):
    return x + 2 * (x**2) + (0.4) * x**3

def derivative_poly(x):
    return 1 + 4 * x + 1.2 * x**2

# Function to calculate the derivative using SymPy
def calculate_derivative(func_str):
    x = sp.symbols('x')
    try:
        # Parse the function string into a sympy expression
        func = sp.sympify(func_str)
        # Calculate the derivative
        derivative = sp.diff(func, x)
        # Convert sympy expressions to numerical functions
        func_lambdified = sp.lambdify(x, func, "numpy")
        derivative_lambdified = sp.lambdify(x, derivative, "numpy")
        return func_lambdified, derivative_lambdified
    except sp.SympifyError:
        st.error("Invalid function input. Please enter a valid mathematical expression.")
        return None, None

# Title
st.title('Gradient Descent Visualizer')
st.sidebar.title("It's your turn..")

# User input
function = st.sidebar.selectbox('Pre Defined Functions', ['Square', 'Cube', 'Polynomial', 'sin', '1/x', 'None'])
starting_point = st.sidebar.number_input('Starting Point', value=5, step=1)
learning_rate = st.sidebar.number_input('Learning Rate', value=0.1, step=0.01)

# Define the selected function and its derivative
if function == 'Square':
    func = square
    derivative_func = derivative_square
elif function == 'Cube':
    func = cube
    derivative_func = derivative_cube
elif function == 'Polynomial':
    func = poly
    derivative_func = derivative_poly
elif function == 'sin':
    func = sin
    derivative_func = derivative_sin
elif function == '1/x':
    func = inverse
    derivative_func = derivative_inverse
elif function == 'None':
    user_func = st.sidebar.text_input("Enter a function (in terms of x): ")
    func, derivative_func = calculate_derivative(user_func)
    if func is None:
        st.stop()

# Check if the starting point has changed
if 'last_starting_point' not in st.session_state or st.session_state.last_starting_point != starting_point:
    st.session_state.path = [starting_point]
    st.session_state.iteration = 0  
    st.session_state.last_starting_point = starting_point  

# Perform one iteration of gradient descent
if st.sidebar.button('Next Iteration'):
    current_point = st.session_state.path[-1]
    new_point = current_point - learning_rate * derivative_func(current_point)
    st.session_state.path.append(new_point)
    st.session_state.iteration += 1 

# Create an array of values for plotting
x_values = np.linspace(-10, 10, 500)
y_values = func(x_values)

# Dynamic scaling based on the function's range
y_min, y_max = np.min(y_values), np.max(y_values)
y_padding = (y_max - y_min) * 0.1

# Plot the function and the path of points
plt.figure(figsize=(8, 6))
plt.plot(x_values, y_values, label=function, color='blue')
plt.scatter(st.session_state.path, [func(x) for x in st.session_state.path], color='red', zorder=5)

# Calculate and plot the tangent line
current_point = st.session_state.path[-1]
slope = derivative_func(current_point)
y_tangent = slope * (x_values - current_point) + func(current_point)
plt.plot(x_values, y_tangent, '--', color='red')

# Set plot limits dynamically
plt.xlim([-10, 10])
plt.ylim([y_min - y_padding, y_max + y_padding])

# Display the iteration number
plt.title(f'Iteration: {st.session_state.iteration}')

# Labels and legend
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)

# Display the plot
st.pyplot(plt)
