#Import the necessary packages
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as pt
import random

#Load the training and testing data (MNIST)
mnist = tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test) = mnist.load_data()

x_train = x_train/255
x_test = x_test/255

#Define the network architecture using Kera
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
    ])

model.summary()

#Train the model using SGD
model.compile(optimizer="sgd",
loss="sparse_categorical_crossentropy", metrics=['accuracy'])

history=model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=3)

#Evaluate the network
test_loss,test_acc=model.evaluate(x_test,y_test)
print("Loss=%.3f" %test_loss)
print("Accuracy=%.3f" %test_acc)

n=random.randint(0, 9999)
pt.imshow(x_test[n])
pt.show()
predicted_value=model.predict(x_test)
pt.imshow(x_test[n])
pt.show()

print('Predicted Value:',predicted_value[n])

#Plot the training loss and accuracy

#Plot the training accuracy
pt.plot(history.history['accuracy'])
pt.plot(history.history['val_accuracy'])
pt.title('model accuracy')
pt.ylabel('accuarcy')
pt.xlabel('epoch')
pt.legend(['Train', 'Validation'], loc='upper right')
pt.show()


#Plot the training loss 
pt.plot(history.history['loss'])
pt.plot(history.history['val_loss'])
pt.title('model loss')
pt.ylabel('loss')
pt.xlabel('epoch')
pt.legend(['Train', 'Validation'], loc='upper left')
pt.show()