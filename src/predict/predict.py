# ./src/predict.py

def predict_digits(model, x_new):
    """
    Predicts digits using the trained model.
    """
    predictions = model.predict(x_new)
    return predictions
