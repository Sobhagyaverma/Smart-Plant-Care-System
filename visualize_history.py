import matplotlib.pyplot as plt
import pickle

try:
    with open('training_history.pkl', 'rb') as f:
        history = pickle.load(f)

    acc = history['accuracy']
    val_acc = history['val_accuracy']  # <-- CORRECTED
    loss = history['loss']
    val_loss = history['val_loss']      # <-- CORRECTED
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'bo', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.show()

except FileNotFoundError:
    print("Error: 'training_history.pkl' not found.")
    print("Please run the training script first to generate the history file.")