import cv2
import numpy as np
from models.model_utils import load_model
import config

model = load_model(config.MODEL_PATH)

img = cv2.imread("opencv_app/drawn_digit.png", cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (28, 28))
img = 255 - img
img = img / 255.0
input_data = img.reshape(1, 784)

output = model.forward(input_data)
predicted = np.argmax(output)
confidence = np.max(output) * 100

print(f"Predicted: {predicted}")
print(f"Confidence: {confidence:.2f}%")