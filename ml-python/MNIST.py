from keras import layers, models
from keras.datasets import mnist
from keras.utils import to_categorical
import json

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print(len(train_images))
print(len(train_labels))
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


if True:
    model = models.Sequential()
    model.add(layers.Flatten(input_shape=(28, 28, 1)))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    model.fit(train_images, train_labels, epochs=200, batch_size=64, validation_data=(test_images, test_labels))

    weights_and_biases = {}
    print(model.layers)
    for layer_num, layer in enumerate(model.layers):
        if len(layer.get_weights()) > 0:
            weights = layer.get_weights()[0].tolist()
            biases = layer.get_weights()[1].tolist()
            weights_and_biases[f'layer_{layer_num}'] = {'weights': weights, 'biases': biases}

    with open('weights_and_biases.json', 'w') as f:
        json.dump(weights_and_biases, f)
