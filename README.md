# MNIST Classifier Project

This project is aimed at building a simple classifier for handwritten digits using the MNIST dataset. The main steps involved are:

1. Data Loading
2. Data Preprocessing
3. Model Training
4. Model Evaluation
5. Model Prediction

## Project Description

The goal of this project is to create a classifier capable of recognizing handwritten digits (0-9) from the MNIST dataset. This project includes loading and preprocessing the data, training a model using Support Vector Machine (SVM), evaluating the model's performance, and making predictions.

## Dependencies

- numpy==1.26.1
- pandas==2.2.0
- matplotlib==3.8.0
- seaborn==0.13.2
- scikit-learn==1.4.0
- psutil==5.9.6
- joblib==1.3.2

You can install the dependencies using the following command:
```sh
pip install -r requirements.txt
```

## How to Run

1. Install dependencies: `pip install -r requirements.txt`
2. Run the main script: `python main.py`

## Project Structure

- `data/`: Contains the dataset.
- `models/`: Contains the trained models.
- `src/data/`: Data loading scripts.
- `src/train/`: Model training and evaluation scripts.
- `src/utils/`: Utility scripts.
- `src/predict/`: Prediction scripts.

## Usage

    Data Loading and Preprocessing: Load and preprocess the MNIST dataset.
    Model Training: Train an SVM model with hyperparameter optimization using Grid Search.
    Model Evaluation: Evaluate the trained model using classification reports and confusion matrices.
    Model Prediction: Use the trained model to make predictions on new data.

## Results

The model achieves an accuracy of approximately 98% on the test set. The confusion matrix and classification report provide detailed performance metrics.
Authors

    Milan "Necraatall" Zlamal

## License

"THE BEER-WARE LICENSE" (Revision 42):
Milan "Necraatall" Zlamal wrote this code. As long as you retain this notice, you can do whatever you want with this stuff. If we meet some day, and you think this stuff is worth it, you can buy me a beer in return.