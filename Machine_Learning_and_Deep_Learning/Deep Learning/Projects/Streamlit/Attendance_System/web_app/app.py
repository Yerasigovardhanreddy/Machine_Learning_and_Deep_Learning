# Importing required libraries
import streamlit as st
import cv2
import numpy as np
import os
import tensorflow as tf
import csv
from datetime import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Load Haar Cascade for face detection
haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
data_dir = "face_data"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)


# Capture faces and display them on Streamlit
def capture_face(name):
    count = 0
    face_data = []
    vid = cv2.VideoCapture(0)
    stframe = st.empty()  

    while True:
        ret, frame = vid.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        # Capture all detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            if count % 3 == 0: 
                face_roi = gray[y:y + h, x:x + w]
                face_roi_resized = cv2.resize(face_roi, (100, 100))
                face_data.append(face_roi_resized)
                count += 1

        # Convert BGR to RGB for Streamlit display
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame, channels='RGB')

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()

    # Save the face data
    np.save(os.path.join(data_dir, f"{name}.npy"), np.array(face_data))


# Prepare the data for model training
def prepare_data(path):
    npy_files = [file for file in os.listdir(path) if file.endswith('.npy')]

    final_data = []
    final_labels = []

    for label, npy_file in enumerate(npy_files):
        file_path = os.path.join(path, npy_file)
        data = np.load(file_path)
        labels = np.full(len(data), label)
        final_data.append(data)
        final_labels.append(labels)

    data = np.concatenate(final_data, axis=0)
    labels = np.concatenate(final_labels, axis=0)
    
    return data, labels, npy_files


# Build the ANN model
def build_model(data, labels, epochs):
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    model = Sequential()
    model.add(Input(shape=(100, 100)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(len(np.unique(labels)), activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test))

    y_test_pred = model.predict(X_test)
    y_test_pred = np.argmax(y_test_pred, axis=1)
    st.write(f"Model's Accuracy: {accuracy_score(y_test, y_test_pred)}")

    return model


# Mark attendance in a CSV file
def mark_attendance(name):
    today_date = datetime.now().strftime('%Y-%m-%d')
    new_entry = [name, datetime.now().strftime('%Y-%m-%d %H:%M:%S')]

    file_exists = os.path.isfile('attendance.csv')
    if file_exists:
        with open('attendance.csv', 'r', newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) > 1 and row[0] == name and row[1].startswith(today_date):
                    return

    with open('attendance.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Name', 'Datetime'])
        writer.writerow(new_entry)


# Recognize faces using the trained model
def recognize(model, npy_files):
    vid = cv2.VideoCapture(0)
    stframe = st.empty()  

    while True:
        ret, frame = vid.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_region = gray[y:y + h, x:x + w]
            face_resized = cv2.resize(face_region, (100, 100))
            face_resized = face_resized.reshape(1, 100, 100)

            prediction = model.predict(face_resized)
            predicted_class = np.argmax(prediction)
            name = npy_files[predicted_class].split('.')[0]

            mark_attendance(name)

            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Convert BGR to RGB for Streamlit display
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame, channels='RGB')

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()


# Streamlit UI
st.title("Smart Face Attendance System")

action = st.sidebar.radio("Select Action", ["Capture Face", "Build Model", "Mark Attendance"])

if action == "Capture Face":
    name = st.text_input("Enter your name:")
    if st.sidebar.button("Capture"):
        if name:
            st.write("Capturing face images...")
            capture_face(name)
            st.write("Face captured and saved.")
        else:
            st.write("Please enter your name.")

elif action == 'Build Model':
    data, labels, npy_files = prepare_data(path="face_data")
    model = build_model(data, labels, epochs=10)

elif action == "Mark Attendance":
    if st.sidebar.button("Submit"):
        st.write("Starting face recognition...")
        st.write("Attendance marked Successfully.")
        data, labels, npy_files = prepare_data(path="face_data")
        model = build_model(data, labels, epochs=10)
        recognize(model, npy_files)
        
        
        