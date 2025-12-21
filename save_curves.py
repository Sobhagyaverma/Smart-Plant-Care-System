import matplotlib.pyplot as plt
import pickle
import os

HISTORY_FILE = 'training_history-NEW.pkl'
ACC_IMG = 'accuracy_curve.png'
LOSS_IMG = 'loss_curve.png'

def generate_curves():
    if not os.path.exists(HISTORY_FILE):
        print(f"Error: {HISTORY_FILE} not found.")
        return

    with open(HISTORY_FILE, 'rb') as f:
        history = pickle.load(f)

    acc = history['accuracy']
    val_acc = history['val_accuracy']
    loss = history['loss']
    val_loss = history['val_loss']
    epochs = range(1, len(acc) + 1)

    # Plot Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, acc, 'bo-', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r*-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(ACC_IMG)
    plt.close()
    print(f"Saved accuracy curve to {ACC_IMG}")

    # Plot Loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss, 'bo-', label='Training Loss')
    plt.plot(epochs, val_loss, 'r*-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(LOSS_IMG)
    plt.close()
    print(f"Saved loss curve to {LOSS_IMG}")

if __name__ == "__main__":
    generate_curves()
