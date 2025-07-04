import matplotlib.pyplot as plt

def plot_loss_accuracy(losses, accuracies):
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(losses, label='Loss' , color='red')
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(accuracies, label='Accuracy' , color='green')
    plt.title("Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()