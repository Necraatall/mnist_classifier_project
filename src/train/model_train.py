# ./src/model_trainer.py

from sklearn.svm import SVC

def train_model(x_train, y_train, kernel='rbf', c=1.0, gamma='scale', random_state=None):
    """
    Trains an SVM model on the training data.
    """
    svm_model = SVC(kernel=kernel, C=c, gamma=gamma, random_state=random_state)
    svm_model.fit(x_train, y_train)
    return svm_model
