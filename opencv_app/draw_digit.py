import cv2
import numpy as np
import sys
import os

# Add root to sys.path for relative imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.model_utils import load_model
import config

# Load the trained model
model = load_model(config.MODEL_PATH)

# Initialize a black canvas
canvas_width, canvas_height = 480, 580
canvas = np.zeros((canvas_height, canvas_width), dtype=np.uint8)

drawing = False
last_point = None

# Button coordinates
button_top_left = (10, 10)
button_bottom_right = (110, 40)

def preprocess(img):
    img = cv2.resize(img, (28, 28))
    img = img.astype(np.float32) / 255.0
    img = img.flatten().reshape(1, -1)
    return img

def get_prediction(img):
    processed = preprocess(img)
    output = model.forward(processed)
    prediction = np.argmax(output)
    confidence = np.max(output) * 100
    return prediction, confidence

def draw(event, x, y, flags, param):
    global drawing, last_point

    if event == cv2.EVENT_LBUTTONDOWN:
        # If clicked on "Clear" button
        if button_top_left[0] <= x <= button_bottom_right[0] and button_top_left[1] <= y <= button_bottom_right[1]:
            canvas[:] = 0
        else:
            drawing = True
            last_point = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        cv2.line(canvas, last_point, (x, y), color=255, thickness=20)
        last_point = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        last_point = None

# Create window
cv2.namedWindow("Draw a Digit")
cv2.setMouseCallback("Draw a Digit", draw)

while True:
    canvas_rgb = cv2.cvtColor(canvas.copy(), cv2.COLOR_GRAY2BGR)

    # Draw "Clear" button
    cv2.rectangle(canvas_rgb, button_top_left, button_bottom_right, (200, 200, 200), -1)
    cv2.putText(canvas_rgb, "Clear", (25, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # Show live prediction if canvas not empty
    if np.count_nonzero(canvas) > 0:
        digit, conf = get_prediction(canvas)
        text1 = f"Prediction: {digit}"
        text2 =  f"Confidence : ({conf:.2f}%)"
        cv2.putText(canvas_rgb, text1, (10, canvas_height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(canvas_rgb, text2, (10, canvas_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Draw a Digit", canvas_rgb)
    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC to exit
        break
    elif key == ord('c'):  # 'c' key to clear as alternative
        canvas[:] = 0

cv2.destroyAllWindows()
