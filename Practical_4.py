#a. Import Required Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Model
from keras.layers import Input, Dense
from keras.callbacks import EarlyStopping

#b. Upload / Access the Dataset
data = pd.read_csv(r'E:\MY CAREER\BE\SEVEN SEMESTER\DL\LPIV\csv_result-ECG5000_TEST.csv')

# Preview the dataset
print(data.head())

# Split the data into features and labels if necessary
X = data.values  # If there are no labels

# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and testing sets
X_train, X_test = train_test_split(X_scaled, test_size=0.2, random_state=42)


# Define the size of the input
input_dim = X_train.shape[1]


#c. Encoder Converts It into Latent Representation
# Define the encoder
input_layer = Input(shape=(input_dim,))
encoder = Dense(32, activation='relu')(input_layer)
encoder = Dense(16, activation='relu')(encoder)  # Latent representation

#d. Decoder Networks Convert It Back to the Original Input
# Define the decoder
decoder = Dense(32, activation='relu')(encoder)
decoder = Dense(input_dim, activation='sigmoid')(decoder)

# Create the Autoencoder model
autoencoder = Model(inputs=input_layer, outputs=decoder)


#e. Compile the Models with Optimizer, Loss, and Evaluation Metrics
# Compile the model
autoencoder.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Fit the model
early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
history = autoencoder.fit(X_train, X_train,
epochs=50,
batch_size=32,
validation_data=(X_test, X_test),
callbacks=[early_stopping],
verbose=1)

# Evaluate the model
loss = autoencoder.evaluate(X_test, X_test)
print(f"Test Loss: {loss[0]}, Test MAE: {loss[1]}")

# Plot the training loss
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Autoencoder Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# Get the reconstructed data
X_reconstructed = autoencoder.predict(X_test)

# Calculate the reconstruction error
reconstruction_error = np.mean(np.power(X_test - X_reconstructed, 2), axis=1)

# Set a threshold for anomaly detection
threshold = 0.05  # Adjust based on your dataset
anomalies = reconstruction_error > threshold

# Print the indices of detected anomalies
print("Detected anomalies at indices:", np.where(anomalies)[0])
