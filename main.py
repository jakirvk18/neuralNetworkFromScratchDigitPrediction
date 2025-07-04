import numpy as np
import time
from utils.data_loader import load_data
from models.neural_net import NeuralNetwork
from models.model_utils import save_model
from utils.plot import plot_loss_accuracy
import config

def train():
    # Load data
    X_train, y_train, X_test, y_test = load_data()

    # Initialize model
    model = NeuralNetwork(config.INPUT_SIZE, config.HIDDEN_SIZE, config.OUTPUT_SIZE)

    losses, accuracies = [], []

    print("🔁 Starting training...\n")
    start_time = time.time()

    # Training loop
    for epoch in range(config.EPOCHS):
        # Forward pass
        output = model.forward(X_train)

        # Compute loss and accuracy
        loss = model.compute_loss(output, y_train)
        preds = np.argmax(output, axis=1)
        labels = np.argmax(y_train, axis=1)
        acc = np.mean(preds == labels)

        # Backward pass and update weights
        model.backward(X_train, y_train, output, config.LEARNING_RATE)

        losses.append(loss)
        accuracies.append(acc)

        print(f"➡️ Epoch {epoch+1}/{config.EPOCHS} - Loss: {loss:.4f}, Accuracy: {acc*100:.2f}%")

    end_time = time.time()
    print(f"\n✅ Training complete in {end_time - start_time:.2f} seconds.")

    # Final test accuracy
    test_output = model.forward(X_test)
    test_preds = np.argmax(test_output, axis=1)
    test_labels = np.argmax(y_test, axis=1)
    test_accuracy = np.mean(test_preds == test_labels)
    print(f"🧪 Final Test Accuracy: {test_accuracy * 100:.2f}%")

    # Save the model
    save_model(model, config.MODEL_PATH)
    print(f"💾 Model saved to {config.MODEL_PATH}")

    # Plot training metrics
    plot_loss_accuracy(losses, accuracies)

if __name__ == "__main__":
    train()
