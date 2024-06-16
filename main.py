import logging
import time
import psutil
from joblib import dump, load
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_predict, train_test_split
from sklearn.svm import SVC
from src.data.data_loader import load_mnist_data
from src.utils.visualization import plot_misclassified_images, plot_confusion_matrix

# Logging setup
logging.basicConfig(
    filename='mnist_classifier.log',
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def log_hardware_info():
    """Log CPU and RAM usage information."""
    logging.info(f"CPU usage: {psutil.cpu_percent()}%")
    logging.info(f"RAM usage: {psutil.virtual_memory().percent}%")
    logging.info(f"Available RAM: {psutil.virtual_memory().available / (1024 * 1024)} MB")

def save_model(model, path):
    """Save the model to the specified path."""
    dump(model, path)

def load_model(path):
    """Load the model from the specified path."""
    return load(path)

def evaluate_classification(y_true, y_pred):
    """Evaluate the classification performance and return the report and confusion matrix."""
    report = classification_report(y_true, y_pred, output_dict=True)
    matrix = confusion_matrix(y_true, y_pred)
    return report, matrix

def print_and_log_report(report):
    """Print and log the classification report."""
    logging.info("Classification Report:")
    print("Classification Report:")
    for label, metrics in report.items():
        if isinstance(metrics, dict):
            logging.info(f"Class {label}:")
            logging.info(f"  Precision: {metrics['precision']:.2f}")
            logging.info(f"  Recall: {metrics['recall']:.2f}")
            logging.info(f"  F1-score: {metrics['f1-score']:.2f}")
            logging.info(f"  Support: {metrics['support']}")
            print(f"Class {label}:")
            print(f"  Precision: {metrics['precision']:.2f}")
            print(f"  Recall: {metrics['recall']:.2f}")
            print(f"  F1-score: {metrics['f1-score']:.2f}")
            print(f"  Support: {metrics['support']}")

def print_and_log_confusion_matrix(matrix):
    """Print and log the confusion matrix."""
    logging.info("Confusion Matrix:")
    print("Confusion Matrix:")
    for row in matrix:
        logging.info(f"{row}")
        print(f"{row}")

def main():
    try:
        logging.info('Program started')
        log_hardware_info()

        start_time = time.time()
        logging.info('Loading and preprocessing data...')
        print('Loading and preprocessing data...')
        x, y = load_mnist_data()

        # Split the data into training and testing sets with random_state for reproducibility
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        logging.info(f'Data loaded and preprocessed in {time.time() - start_time:.2f} seconds')
        print(f'Data loaded and preprocessed in {time.time() - start_time:.2f} seconds')

        # Hyperparameter optimization using Grid Search with random_state for reproducibility
        param_grid = {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto'], 'kernel': ['rbf', 'linear']}
        grid = GridSearchCV(SVC(random_state=42), param_grid, refit=True, verbose=2, cv=5)

        start_time = time.time()
        logging.info('Training model...')
        print('Training model...')
        grid.fit(x_train, y_train)
        logging.info(f'Model trained with Grid Search in {time.time() - start_time:.2f} seconds')
        print(f'Model trained with Grid Search in {time.time() - start_time:.2f} seconds')

        logging.info('Evaluating model using cross-validation...')
        print('Evaluating model using cross-validation...')
        y_pred = cross_val_predict(grid.best_estimator_, x_test, y_test, cv=5, n_jobs=-1)

        report, matrix = evaluate_classification(y_test, y_pred)
        logging.info('Model evaluated with classification report and confusion matrix.')
        print('Model evaluated with classification report and confusion matrix.')

        # Print and log the classification report and confusion matrix
        print_and_log_report(report)
        print_and_log_confusion_matrix(matrix)

        model_path = 'models/best_svm_model.pkl'
        save_model(grid.best_estimator_, model_path)
        logging.info(f'Best model saved as: {model_path}')
        print(f'Best model saved as: {model_path}')

        best_model = load_model(model_path)
        y_pred = best_model.predict(x_test)
        plot_misclassified_images(x_test, y_test, y_pred, num_images=10, seed=42)
        plot_confusion_matrix(y_test, y_pred)

        logging.info(f'Final model accuracy on test data: {model_path}')
        logging.info('Program finished')
        log_hardware_info()

    except Exception as e:
        logging.error(f'An error occurred: {str(e)}', exc_info=True)
        print(f'An error occurred: {str(e)}')

if __name__ == "__main__":
    main()
