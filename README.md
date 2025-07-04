# **🧠 Neural Network from Scratch: Handwritten Digit Recognition**
This project demonstrates how to build a simple feedforward neural network from scratch (without relying on deep learning libraries like TensorFlow or PyTorch) to classify handwritten digits using the MNIST dataset.

## **✨ Project Highlights**
Model Implementation: A complete neural network model built using only NumPy.

Training & Evaluation: Includes a robust training loop and evaluation on unseen test data.

Live Drawing App: An interactive application built with OpenCV allows you to draw digits and get real-time predictions.

Prediction Visualization: Utilities to visualize model predictions and confidence scores.

![Screenshot 2025-07-04 212001](https://github.com/user-attachments/assets/7a006db0-4df2-4997-bfae-0585b3294da4)


## **📁 Project Structure**

```bash
NeuralNetworkFromScratch/
│
├── data/                     # Datasets (CSV, images, etc.)
│   └── mnsitdatasets
│
├── models/                   # Core neural network implementation
│   ├── activations.py        # Sigmoid, ReLU, etc.
│   ├── layers.py             # Layer classes
│   ├── losses.py             # MSE, Cross Entropy, etc.
│   ├── neural_net.py         # Main NN class (forward, backward)
│   └── optimizer.py          # SGD, Momentum, etc.
│
├── utils/                    # Helper functions
│   ├── data_loader.py        # Load and preprocess datasets
│   └── plot.py               # For plotting training loss/accuracy
│
├── notebooks/                # Jupyter notebooks for experiments
│   └── test_nn.ipynb
│
├── main.py                   # Entry point to train/test the model
├── config.py                 # Parameters (learning rate, epochs, etc.)
├── requirements.txt          # Python dependencies
└── README.md                 # Project overview
```
## **📦 Requirements**
Install the necessary dependencies using pip:

pip install numpy pandas opencv-python matplotlib

## **🧪 Dataset**
This project utilizes the CSV version of the MNIST dataset. You can download the required files from Kaggle:https://www.kaggle.com/datasets/oddrationale/mnist-in-csv

## **📥 Kaggle - MNIST in CSV**

Please download mnist_train.csv and mnist_test.csv and place them inside the data/ folder.

## **🚀 How to Run**
Follow these steps to get the project up and running:

1. Train the Model
Execute the training script to train the neural network:

python train_model.py

This will train a simple neural network and save the learned model to models/trained_model.pkl.

2. Visualize Predictions
Open the Jupyter Notebook to explore predictions on test samples with their corresponding confidence scores:

jupyter notebook visualize_digits.ipynb

3. Live Drawing with OpenCV
Launch the live drawing application to interactively test the model:

python opencv_app/draw_digit.py

Draw a digit on the canvas using your mouse and observe the real-time prediction.

Press C to clear the drawing screen.

Press ESC to exit the application.

## **📌 Features**
Built from Scratch: Core components implemented using only NumPy, demonstrating a deep understanding of neural network mechanics.

Forward & Backward Propagation: Includes the fundamental algorithms for neural network training.

Real-time Prediction: Seamless integration with an OpenCV canvas for interactive, live digit recognition.

Modular Design: A clean and organized project structure separates concerns for training, prediction, and visualization.

# **🤖 Author**
Jakir Hussain

## **📄 License**
This project is open-source and freely available under the MIT License.

