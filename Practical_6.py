import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras import layers, models, optimizers
import matplotlib.pyplot as plt

# Load the tf_flowers dataset and split into train and test sets
(train_data, test_data) = tfds.load(
    "tf_flowers",
    split=["train[:70%]", "train[70%:]"], 
    as_supervised=True,
    batch_size=-1  # Load full dataset into memory (for resizing)
)

# Extract images and labels from dataset
train_images, train_labels = tfds.as_numpy(train_data)
test_images, test_labels = tfds.as_numpy(test_data)

# Resize images to 150x150
train_images = tf.image.resize(train_images, (150, 150))
test_images = tf.image.resize(test_images, (150, 150))

# Transform labels to one-hot format for categorical crossentropy loss
train_labels = to_categorical(train_labels, num_classes=5)
test_labels = to_categorical(test_labels, num_classes=5)

# Preprocess images for VGG16 model
train_images = preprocess_input(train_images)
test_images = preprocess_input(test_images)

# Load the VGG16 model, excluding the top layers
base_model = VGG16(weights="imagenet", include_top=False, input_shape=(150, 150, 3))

# Freeze all layers in the base model initially
base_model.trainable = False

# Add custom classifier layers on top of the VGG16 base
model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(128, activation='relu'),  # Increase units for tuning
    layers.Dropout(0.5),                   # Added dropout for regularization
    layers.Dense(64, activation='relu'),   # Add one more dense layer
    layers.Dense(5, activation='softmax')  # Final layer for 5 classes
])

# Compile the model with a lower learning rate
model.compile(
    optimizer=optimizers.Adam(learning_rate=0.0001),  # Adjusted learning rate for tuning
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Print model summary to verify structure
model.summary()

# Train the model with custom layers on the flower dataset
history = model.fit(
    train_images, train_labels,
    epochs=2,  # Short initial training to confirm functionality
    validation_split=0.2,
    batch_size=32
)

# Unfreeze the top layers of the base model for fine-tuning
# Here, we unfreeze the last 4 layers as an example
for layer in base_model.layers[-4:]:
    layer.trainable = True

# Recompile the model with a smaller learning rate for fine-tuning
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-5),  # Smaller LR for fine-tuning
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Further train the model on the dataset with fine-tuned layers
fine_tune_epochs = 2
total_epochs = 2 + fine_tune_epochs
history_fine = model.fit(
    train_images, train_labels,
    epochs=total_epochs,
    initial_epoch=history.epoch[-1],
    validation_split=0.2,
    batch_size=32
)

# Evaluate the model on the test set
loss, accuracy = model.evaluate(test_images, test_labels)
print("Test Loss: {:.4f}, Test Accuracy: {:.4f}".format(loss, accuracy))

# Plot accuracy of training and validation over epochs
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.plot(history_fine.history['accuracy'], label='Training Accuracy (Fine-tune)')
plt.plot(history_fine.history['val_accuracy'], label='Validation Accuracy (Fine-tune)')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='upper left')
plt.show()
