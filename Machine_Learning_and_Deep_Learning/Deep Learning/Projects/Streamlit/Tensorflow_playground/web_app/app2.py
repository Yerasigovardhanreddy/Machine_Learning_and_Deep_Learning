# Import required libraries
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import io
from sklearn.datasets import make_classification, make_moons, make_circles, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import InputLayer, Dense
from keras.regularizers import L1, L2
from mlxtend.plotting import plot_decision_regions
import warnings
warnings.filterwarnings("ignore")
from io import StringIO

# Main Title
st.title('Tensor Flow Playground')

# Sidebar Title
st.sidebar.title('Tensorflow Playground')

# Problem Type
problem_type = st.sidebar.selectbox('Problem Type', ['None', 'Classification', 'Regression', 'Moons', 'Circles'])

# Choose Dataset
st.sidebar.title('Choose Dataset')

# Datasets
data_set = st.sidebar.selectbox('Datasets', [
    '1.ushape.csv', '2.concerticcir1.csv', '3.concertriccir2.csv', 
    '4.linearsep.csv', '5.outlier.csv', '6.overlap.csv', 
    '7.xor.csv', '8.twospirals.csv', '9.random.csv', 'None'
])

# Learning Rate
learning_rate = st.sidebar.selectbox('Learning Rate', [0.00001, 0.0001, 0.001, 0.01, 0.03, 0.1, 0.3, 1, 3, 10])

# Activation Function
activation_func = st.sidebar.selectbox('Activation', ['tanh', 'Sigmoid', 'linear', 'relu', 'softmax'])

# Regularization Rate
regularization_rate = st.sidebar.selectbox('Regularization Rate', [0.00001, 0.0001, 0.001, 0.01, 0.03, 0.1, 0.3, 1, 3, 10])

# Regularization Type
regularization = st.sidebar.selectbox('Regularization', ['None', 'L1', 'L2'])

# Epochs
epochs = st.sidebar.select_slider("Select number of Epochs", options=[i for i in range(1, 1001)])

# Test Size
test_size = st.sidebar.slider("Test Size (%)", min_value=10, max_value=90, value=25, step=1) / 100

# Hidden Layers
hidden_layers = st.sidebar.select_slider('Hidden Layers', options=[i for i in range(1, 51)])

# Neurons in Each Layer
neurons_per_layer = []
for i in range(1, hidden_layers + 1):
    n = st.sidebar.text_input(f'No of Neurons in Layer {i}', '2')
    try:
        neurons_per_layer.append(int(n))
    except ValueError:
        st.error(f"Invalid input for the number of neurons in Layer {i}. Please enter an integer.")

# Batch Size
if data_set != "None":
    file_path = f"D:\\csv_files9\\{data_set}"
    df = pd.read_csv(file_path)
    X = df.iloc[:, :2].values
    y = df.iloc[:, -1].values
    batch_size = st.sidebar.select_slider("Batch Size", options=[i for i in range(1, X.shape[0] + 1)])
else:
    batch_size = st.sidebar.select_slider("Batch Size", options=[i for i in range(1, 10001)])

# Regularization configuration
kernel_regularizer = None
bias_regularizer = None
if regularization == 'L1':
    kernel_regularizer = L1(regularization_rate)
    bias_regularizer = L1(regularization_rate)
elif regularization == 'L2':
    kernel_regularizer = L2(regularization_rate)
    bias_regularizer = L2(regularization_rate)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'history' not in st.session_state:
    st.session_state.history = None
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None

if st.sidebar.button('Submit'):
    if data_set != "None":
        # Dataset Handling
        file_path = f"D:\\csv_files9\\{data_set}"
        df = pd.read_csv(file_path)
        X = df.iloc[:, :2].values
        y = df.iloc[:, -1].values 
        problem_type = 'Classification'  # Treat as classification if dataset is chosen
    elif problem_type in ['Classification', 'Moons', 'Circles']:
        if problem_type == 'Classification':
            X, y = make_classification(n_samples=10000, n_features=2, n_informative=2, n_redundant=0, n_repeated=0, n_classes=2, class_sep=2.5, random_state=10)
        elif problem_type == 'Moons':
            X, y = make_moons(n_samples=10000, noise=0.1, random_state=20)
        elif problem_type == 'Circles':
            X, y = make_circles(n_samples=10000, noise=0.05, random_state=20)
    elif problem_type == 'Regression':
        X, y = make_regression(n_samples=10000, n_features=2, n_informative=2, n_targets=1, noise=0.05, random_state=20)
    else:
        st.write("Please select a valid dataset or problem type.")
        st.stop()

    # Data visualization
    st.subheader("Visualization of Data Points with Class Labels")
    fig, ax = plt.subplots(figsize=(10, 4))
    if problem_type in ['Classification', 'Moons', 'Circles']:
        sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, ax=ax)
    else:
        sns.scatterplot(x=X[:, 0], y=X[:, 1], ax=ax)
    st.pyplot(fig)

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=20, stratify=y if problem_type != 'Regression' else None)

    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Save model and training data in session state
    st.session_state.X_train = X_train
    st.session_state.y_train = y_train
    st.session_state.X_test = X_test
    st.session_state.y_test = y_test

    # Build the model
    model = Sequential()
    model.add(InputLayer(input_shape=(2,)))
    for neurons in neurons_per_layer:
        model.add(Dense(units=neurons, activation=activation_func, use_bias=True, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer))
    
    # Final layer configuration based on problem type
    if problem_type == 'Regression':
        model.add(Dense(units=1, activation='linear', use_bias=True))
        loss_function = 'mse'
        metrics = ['mse', 'mae']
    else: 
        model.add(Dense(units=1, activation='sigmoid', use_bias=True))
        loss_function = 'binary_crossentropy'
        metrics = ['accuracy']

    # Compile the model
    model.compile(optimizer='sgd', loss=loss_function, metrics=metrics)
        
    # Display model summary
    st.subheader("Model Summary")
    summary_str = StringIO()
    model.summary(print_fn=lambda x: summary_str.write(x + '\n'))
    st.text(summary_str.getvalue())

    # Training the model
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, validation_split=0.2)

    # Save history in session state
    st.session_state.history = history

    # Plot loss and validation loss
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, epochs + 1), history.history['loss'], label='Train loss')
    ax.plot(range(1, epochs + 1), history.history['val_loss'], label='Val loss')
    ax.set_title("Training and Validation Loss Analysis")
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()
    st.pyplot(fig)

    if problem_type != 'Regression':
        # Plot accuracy and validation accuracy
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(range(1, epochs + 1), history.history['accuracy'], label='Train Accuracy')
        ax.plot(range(1, epochs + 1), history.history['val_accuracy'], label='Val Accuracy')
        ax.set_title('Training and Validation Accuracy Analysis')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Accuracy')
        ax.legend()
        st.pyplot(fig)
        
        # Plot decision surface
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_decision_regions(X=st.session_state.X_train, y=st.session_state.y_train.astype(int), clf=model)
        st.pyplot(fig)

        fig, ax = plt.subplots(figsize=(10, 6))
        plot_decision_regions(X=st.session_state.X_test, y=st.session_state.y_test.astype(int), clf=model)
        st.pyplot(fig)
    
    elif problem_type == 'Regression':
        # Plot accuracy and validation accuracy
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(range(1, epochs + 1), history.history['mae'], label='Train mae')
        ax.plot(range(1, epochs + 1), history.history['val_mae'], label='Val mae')
        ax.set_title('Training and Validation MAE Analysis')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('MAE')
        ax.legend()
        st.pyplot(fig)

