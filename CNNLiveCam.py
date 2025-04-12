import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
import tensorflow as tf
import numpy as np

# Load the pre-trained model
model = tf.keras.models.load_model('screw_classifier_model.h5')

# Compile the model (to suppress the warning)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Class labels as per your dataset
class_labels = ['Torx', 'Philips', 'Plaster', 'Machine']

# Load saved calibration parameters
calibration_data = np.load('camera_calibration_params.npz')
mtx = calibration_data['mtx']  # Camera matrix
dist = calibration_data['dist']  # Distortion coefficients

# Open the webcam (camera index 0 is typically the default webcam)
cap = cv2.VideoCapture(0)

# Set the webcam resolution (same as the input size used during training)
img_size = 128  # same as training image size
cap.set(cv2.CAP_PROP_FRAME_WIDTH, img_size)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, img_size)

# Loop to capture and process each frame from the webcam
while True:
    ret, frame = cap.read()  # Read a frame from the webcam
    if not ret:
        break

    undistorted_frame = cv2.undistort(frame, mtx, dist)

    # Resize the frame to the same size used for training
    resized_frame = cv2.resize(frame, (img_size, img_size))

    # Normalize the image to the range [0, 1] as done during training
    normalized_frame = resized_frame / 255.0

    # Add batch dimension as the model expects a batch of images
    input_frame = np.expand_dims(normalized_frame, axis=0)

    # Predict the class of the image
    predictions = model.predict(input_frame)
    predicted_class = np.argmax(predictions, axis=1)
    predicted_label = class_labels[predicted_class[0]]

    # Display the predicted class on the live feed
    cv2.putText(frame, f"Predicted: {predicted_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    print(predicted_label)
    
    # Show the live feed with the prediction
    cv2.imshow('Live Camera Feed', undistorted_frame)

    # Exit the loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close any open windows
cap.release()
cv2.destroyAllWindows()
