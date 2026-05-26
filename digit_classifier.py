import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1. Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 2. Normalize the data (scale pixel values from 0-255 down to 0-1)
x_train, x_test = x_train / 255.0, x_test / 255.0

# Reshape the data to include the channel dimension (required for ImageDataGenerator)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# 3. Data Augmentation (generates slightly altered images to make the AI smarter)
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1
)
datagen.fit(x_train)

# 4. Build the model with LeakyReLU activation functions
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28, 1)),
    layers.Dense(128, activation=tf.keras.layers.LeakyReLU(alpha=0.1)),
    layers.Dense(64, activation=tf.keras.layers.LeakyReLU(alpha=0.1)),
    layers.Dense(10, activation='softmax')
])

# 5. Compile the model with the RMSprop optimizer
model.compile(optimizer='rmsprop',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 6. Train the model with augmented data for 10 epochs
print("Training model... this might take a minute!")
model.fit(datagen.flow(x_train, y_train, batch_size=32), epochs=10)

# 7. Evaluate the model on unseen test data
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"\nTest accuracy: {test_acc:.4f}")

# 8. Make predictions and display the very first test image with its predicted label
predictions = model.predict(x_test)

plt.imshow(x_test[0].reshape(28, 28), cmap=plt.cm.binary)
plt.title(f"Predicted: {predictions[0].argmax()}")
plt.show()