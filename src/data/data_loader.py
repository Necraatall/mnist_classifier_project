# ./src/data_loader.py

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_mnist_data():
    """
    Loads the MNIST dataset and performs basic data preprocessing.
    """
    # Load the MNIST dataset
    mnist = fetch_openml('mnist_784', version=1)
    x, y = mnist.data, mnist.target
    
    # Normalize pixel values
    scaler = StandardScaler()
    x_normalized = scaler.fit_transform(x)
    
    return x_normalized, y

def split_data(x, y, test_size=0.2, random_state=42):
    """
    Splits the data into training and testing sets.
    """
    return train_test_split(x, y, test_size=test_size, random_state=random_state)
