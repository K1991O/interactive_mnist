import tkinter as tk
import numpy as np
from keras.preprocessing.image import img_to_array
from PIL import Image, ImageDraw
from keras import layers, models
from keras.datasets import mnist
from keras.utils import to_categorical
import json


with open('weights_and_biases_300_16.json', 'r') as f:
    weights_and_biases = json.load(f)

new_model = models.Sequential()
new_model.add(layers.Flatten(input_shape=(28, 28, 1)))
new_model.add(layers.Dense(16, activation='relu'))
new_model.add(layers.Dense(16, activation='relu'))
new_model.add(layers.Dense(10, activation='softmax'))

for layer_num, layer in enumerate(new_model.layers):
    if f'layer_{layer_num}' in weights_and_biases:
        weights = np.array(weights_and_biases[f'layer_{layer_num}']['weights'])
        biases = np.array(weights_and_biases[f'layer_{layer_num}']['biases'])
        layer.set_weights([weights, biases])

# Create a new Tkinter window
window = tk.Tk()

# Create a canvas for drawing
canvas = tk.Canvas(window, width=280, height=280, bg="white")
canvas.pack()

# Create an in-memory image to draw on, and a draw tool
image = Image.new("RGB", (28, 28), "white")
draw = ImageDraw.Draw(image)


# Function to clear the canvas
def clear_canvas():
    canvas.delete("all")
    draw.rectangle([(0, 0), (28, 28)], fill="white")


# Function to predict the digit
def predict_digit():
    # Convert the in-memory image to a NumPy array
    img_array = img_to_array(image.resize((28, 28))).reshape(1, 28, 28, 1).astype('float32') / 255

    # Predict the digit using the trained model
    prediction = new_model.predict(img_array)
    predicted_digit = np.argmax(prediction)

    # Print the prediction
    print(f"Predicted Digit: {predicted_digit}")


# Function to draw on the canvas
def paint(event):
    color = "black"
    x1, y1 = (event.x - 5), (event.y - 5)
    x2, y2 = (event.x + 5), (event.y + 5)
    canvas.create_oval(x1, y1, x2, y2, fill=color, width=10)

    # Draw on the in-memory image
    draw.line([x1, y1, x2, y2], fill=color, width=10)


canvas.bind("<B1-Motion>", paint)

# Button to clear the canvas
clear_button = tk.Button(window, text="Clear", command=clear_canvas)
clear_button.pack()

# Button to predict the digit
predict_button = tk.Button(window, text="Predict", command=predict_digit)
predict_button.pack()

window.mainloop()