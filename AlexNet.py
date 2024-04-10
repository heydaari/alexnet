# importing necessary libraries
import tensorflow as tf
from tensorflow import keras

# loading CIFAR10 dataset from keras.datasets

(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

# we only use 1/5 of the whole dataset due to limitations
X_train, y_train, X_test, y_test = X_train[:10000], y_train[:10000], X_test[:1000], y_test[:1000]

y_train.resize(y_train.shape[0])
y_test.resize(y_test.shape[0])

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape, sep='\n')


# Assuming X_train and X_test are your numpy arrays of images
# And each image in X_train and X_test is of shape (height, width, channels)

def resize_images(images):
    return tf.image.resize(images, [128, 128])

X_train = resize_images(X_train)
X_test = resize_images(X_test)

print(X_train.shape, X_test.shape, sep = '\n')

# Creating AlexNet model with keras Sequential API
model = keras.models.Sequential([
    keras.layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(128,128,3)),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=384, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=256, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(4096, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(4096, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation='softmax')
])

alexnet_optimizer = optimizer=tf.optimizers.SGD(learning_rate = 0.001)

model.compile(loss='sparse_categorical_crossentropy', optimizer = alexnet_optimizer,  metrics=['accuracy'])
print(model.summary())

model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs = 64)

