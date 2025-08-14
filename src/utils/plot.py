import matplotlib.pyplot as plt

def plot_training_history(epoch_accuracies, epoch_losses):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(len(epoch_accuracies)), epoch_accuracies)
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')

    plt.subplot(1, 2, 2)
    plt.plot(range(len(epoch_losses)), epoch_losses)
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')

    plt.suptitle('Model Training History')
    plt.show()