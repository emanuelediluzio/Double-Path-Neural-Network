import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

# Function to calculate optical flow
def optical_flow(x):
    x_np = x.numpy()
    frame1 = x_np[:, :, :, :3]
    frame2 = x_np[:, :, :, 3:]
    prvs = cv2.cvtColor(frame1[0], cv2.COLOR_BGR2GRAY)
    next = cv2.cvtColor(frame2[0], cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    ang = ang * 180 / np.pi / 2
    flow = np.stack([mag, ang], axis=-1)
    return flow

# Define the neural network model with a double pathway inspired by ventral and dorsal pathways
def build_double_path_model(input_shape, num_classes):
    # Ventral Path (Object Recognition)
    ventral_input = layers.Input(shape=input_shape)
    ventral_conv1 = layers.Conv2D(64, (3, 3), activation='relu')(ventral_input)
    ventral_conv2 = layers.Conv2D(128, (3, 3), activation='relu')(ventral_conv1)
    ventral_pool = layers.MaxPooling2D((2, 2))(ventral_conv2)
    ventral_flat = layers.Flatten()(ventral_pool)
    ventral_output = layers.Dense(256, activation='relu')(ventral_flat)

    # Dorsal Path (Space and Motion Perception)
    dorsal_input = layers.Input(shape=input_shape)
    dorsal_optical_flow = layers.Lambda(optical_flow)(dorsal_input)
    dorsal_conv1 = layers.Conv2D(64, (3, 3), activation='relu')(dorsal_optical_flow)
    dorsal_conv2 = layers.Conv2D(128, (3, 3), activation='relu')(dorsal_conv1)
    dorsal_pool = layers.MaxPooling2D((2, 2))(dorsal_conv2)
    dorsal_flat = layers.Flatten()(dorsal_pool)
    dorsal_output = layers.Dense(256, activation='relu')(dorsal_flat)

    # Concatenate the results of the two pathways
    merged = layers.concatenate([ventral_output, dorsal_output])

    # Add a final fully connected layer for classification
    final_output = layers.Dense(num_classes, activation='softmax')(merged)

    # Create the model
    model = models.Model(inputs=[ventral_input, dorsal_input], outputs=final_output)

    return model

# Specify the input shape and the number of classes
input_shape = (224, 224, 6)  # Input now includes two consecutive frames
num_classes = 10

# Build the double pathway model
double_path_model = build_double_path_model(input_shape, num_classes)

# Display the architecture of the model
double_path_model.summary()

# Generate a sample dataset
num_samples = 1000
image_size = (224, 224, 3)

# Generate random images for the dataset
images_ventral = np.random.randint(0, 256, size=(num_samples, *image_size))
images_dorsal = np.random.randint(0, 256, size=(num_samples, *image_size))

# Create random optical flow for the dorsal pathway
optical_flow_dorsal = np.random.rand(num_samples, *image_size)

# Concatenate images to create the input dataset
input_data = np.concatenate([images_ventral, optical_flow_dorsal], axis=-1)

# Create random labels for the dataset
labels = np.random.randint(0, num_classes, size=(num_samples,))

# Split the dataset into training and test sets
x_train, x_test, y_train, y_test = train_test_split(input_data, labels, test_size=0.2, random_state=42)

# Train the model
double_path_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
double_path_model.fit([x_train[:, :, :, :3], x_train[:, :, :, 3:]], y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model on the test set
test_loss, test_accuracy = double_path_model.evaluate([x_test[:, :, :, :3], x_test[:, :, :, 3:]], y_test)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")
