# importing required libraries
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow, keras
import io
from sklearn.datasets import make_classification, make_regression, make_moons, make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import InputLayer, Dense, Dropout
from keras.regularizers import L1, L2
import mlxtend
from mlxtend.plotting import plot_decision_regions
import warnings
warnings.filterwarnings("ignore") 



# Title
st.sidebar.title('Tensorflow Playground')

# Problem Type
problem_type = st.sidebar.selectbox('Problem Type', ['Classification', 'Regression', 'Moons', 'Circles'])

# Learning Rate
learning_rate = st.sidebar.selectbox('Learning Rate', [0.00001, 0.0001, 0.001, 0.01, 0.03, 0.1, 0.3, 1, 3, 10])

# Activation
activation_func = st.sidebar.selectbox('Activation', ['tanh', 'Sigmoid', 'linear', 'relu', 'softmax'])

# Regularization Rate
regularization_rate = st.sidebar.selectbox('Regularization Rate', [0.00001, 0.0001, 0.001, 0.01, 0.03, 0.1, 0.3, 1, 3, 10])

# Regularization
regularization = st.sidebar.selectbox('Regularization', ['None', 'L1', 'L2'])

if regularization == 'None':
    kernel_regularizer = None
    bias_regularizer = None
elif regularization == 'L1':
    kernel_regularizer = L1(regularization_rate)
    bias_regularizer = L1(regularization_rate)
elif regularization == 'L2':
    kernel_regularizer = L2(regularization_rate)
    bias_regularizer = L2(regularization_rate)


# Epochs
epochs = st.sidebar.select_slider("Select number of Epochs", options=[i for i in range(1, 1001)])

# Split Train/Test
test_size = st.sidebar.slider("Test Size (%)", min_value=10, max_value=90, value=40, step=1) / 100

# Hidden layers
hidden_layers = st.sidebar.select_slider('Hidden Layers', options = [i for i in range(1, 51)])

# Build the model
model = Sequential()
model.add(InputLayer(input_shape=(2,)))
for i in range(1, hidden_layers + 1):
    n = st.sidebar.text_input(f'No of Neurons in Layer {i}', '2')
    try:
        n = int(n)
        model.add(Dense(units=n, activation=activation_func, use_bias=True, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer))
    except Exception as e:
        st.error(f"Unexpected error: {e}") 
if problem_type == 'Regression':
    model.add(Dense(units=1, activation='linear', use_bias=True))
else:
    model.add(Dense(units=1, activation='sigmoid', use_bias=True))
    # Classification Task 
    if problem_type == 'Classification':
        # Generate synthetic data
        X, y = make_classification(n_samples=10000, n_features=2, n_informative=2, n_redundant=0, n_repeated=0, n_classes=2, class_sep=2.5, random_state=10)
    elif problem_type == 'Moons':
        X, y = make_moons(n_samples= 10000, noise= 0.1, random_state= 20)
    elif problem_type == 'Circles':
        X, y = make_circles(n_samples= 10000, noise= 0.05, random_state= 20)
        

# Batch Size
batch_size = st.sidebar.select_slider("Batch Size", options=[i for i in range(1, len(X)+ 1)])


# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'history' not in st.session_state:
    st.session_state.history = None
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None

if st.sidebar.button('Submit'):
    
    # Regression Task
    if problem_type == 'Regression':
        
        # Generate synthetic data
        X, y = make_regression(n_samples = 10000, n_features =2, n_informative = 2, n_targets= 1, noise= 0.05, random_state= 20)
        
        # Plot the relationship between X and y
        fig, axs = plt.subplots(1, 2, figsize= (7, 4))
        sns.scatterplot(x= X[:, 0], y= y, ax= axs[0]) 
        axs[0].set_xlabel('Feature 1')
        axs[0].set_ylabel('Target')
        
        sns.scatterplot(x= X[:, 1], y= y, ax= axs[1])
        axs[1].set_xlabel('Feature 2') 
        axs[1].set_ylabel('Target')
        st.pyplot(fig) 
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=20)
        
        # Standardize
        scaler = StandardScaler()
        X_traint = scaler.fit_transform(X_train) 
        X_testt = scaler.transform(X_test)
                
        # Save model and training data in session state
        st.session_state.model = model
        st.session_state.X_train = X_traint
        st.session_state.y_train = y_train
        st.session_state.X_test = X_testt
        st.session_state.y_test = y_test
        
        # Display model summary
        buffer = io.StringIO()
        model.summary(print_fn=lambda x: buffer.write(x + '\n'))
        st.text("Model Summary:")
        st.text(buffer.getvalue())
        buffer.close()

        # Compile and train the model
        model.compile(optimizer='sgd', loss='mse', metrics=['mae', 'mse'])
        st.session_state.history = model.fit(X_traint, y_train, epochs=epochs, batch_size=batch_size, verbose=1, validation_split=0.2)

        # Plot loss and val loss
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(range(1, epochs + 1), st.session_state.history.history['loss'], label='Train loss')
        ax.plot(range(1, epochs + 1), st.session_state.history.history['val_loss'], label='Val loss')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.legend()
        st.pyplot(fig)

        # plot mse and val mse
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(range(1, epochs + 1), st.session_state.history.history['mse'], label='Train mse')
        ax.plot(range(1, epochs + 1), st.session_state.history.history['val_mse'], label='Val mse')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Accuracy')
        ax.legend()
        st.pyplot(fig)
    
    else:
        
        st.subheader("Visualization of Data Points with Class Labels")   

        # Plot the relationship between X and y
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, ax=ax)
        st.pyplot(fig)
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=20, stratify=y)

        # Standardize
        scaler = StandardScaler()
        X_traint = scaler.fit_transform(X_train)
        X_testt = scaler.transform(X_test)
        
        # Save model and training data in session state
        st.session_state.model = model
        st.session_state.X_train = X_traint
        st.session_state.y_train = y_train
        st.session_state.X_test = X_testt
        st.session_state.y_test = y_test 

        # Display model summary
        buffer = io.StringIO()
        model.summary(print_fn=lambda x: buffer.write(x + '\n'))
        st.subheader("Model Summary:")
        st.text(buffer.getvalue())
        buffer.close()

        # Compile and train the model
        model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
        st.session_state.history = model.fit(X_traint, y_train, epochs=epochs, batch_size=batch_size, verbose=1, validation_split=0.2)

        # Plot loss and accuracy
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(range(1, epochs + 1), st.session_state.history.history['loss'], label='Train loss')
        ax.plot(range(1, epochs + 1), st.session_state.history.history['val_loss'], label='Val loss')
        ax.set_title("Training and Validation Loss Analysis")
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.legend()
        st.pyplot(fig)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(range(1, epochs + 1), st.session_state.history.history['accuracy'], label='Train Accuracy')
        ax.plot(range(1, epochs + 1), st.session_state.history.history['val_accuracy'], label='Val Accuracy')
        ax.set_title('Training and Validation Accuracy Analysis')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Accuracy') 
        ax.legend()
        st.pyplot(fig)

        # plot Decision surface for train data
        fig, ax = plt.subplots(figsize =(8,4))
        plot_decision_regions(X = st.session_state.X_train, y= st.session_state.y_train.astype(int), clf= st.session_state.model)
        st.pyplot(fig)
        
        # plot Decision surface for test data
        fig, axs = plt.subplots(figsize = (8,4)) 
        plot_decision_regions(X = st.session_state.X_test, y = st.session_state.y_test.astype(int), clf = st.session_state.model)
        st.pyplot(fig)     
        
        