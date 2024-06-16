# ./src/train/model_evaluation.py

from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(model, x_test, y_test):
    """
    Evaluates the performance of the model on the test data.
    """
    # Predict values on the test set
    y_pred = model.predict(x_test)
    
    # Calculate metrics
    report = classification_report(y_test, y_pred, output_dict=True)
    matrix = confusion_matrix(y_test, y_pred)
    
    return report, matrix
