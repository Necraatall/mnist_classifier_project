# ./src/utils/visualization.py

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_misclassified_images(x_test, y_test, y_pred, num_images=10, seed=None):
    """
    Displays several misclassified images along with predicted and true values.
    """
    # Get indices of misclassified images
    misclassified_idx = np.nonzero(y_pred != y_test)[0]

    # Randomly select misclassified images
    rng = np.random.default_rng(seed)
    random_idx = rng.choice(misclassified_idx, min(num_images, len(misclassified_idx)), replace=False)

    # Display images
    plt.figure(figsize=(12, 8))
    for i, idx in enumerate(random_idx):
        plt.subplot(2, 5, i + 1)
        plt.imshow(x_test[idx].reshape(28, 28), cmap='gray')
        plt.title(f'Pred: {y_pred[idx]}, True: {y_test.iloc[idx]}')  # Use .iloc to access values
        plt.axis('off')
    plt.show()

def plot_confusion_matrix(y_test, y_pred):
    """
    Displays a confusion matrix using a heatmap.
    """
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.show()
