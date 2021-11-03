import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import f1_score

def plot_roc(fpr, tpr):
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle='--', color='red', label='Random Classifier')
    plt.plot([0, 0, 1], [0, 1, 1], linestyle=':', color='green', label='Perfect Classifier')
    plt.show()

def evaluate_model(y_train, y_test, y_pred_prob_train, y_pred_prob_test):
    """
    prob의 경우는 (n, 1)짜리 np.array
    """
    y_pred_train = y_pred_prob_train.round()
    y_pred_test = y_pred_prob_test.round()

    # y_pred_train = model.predict(X_train)
    print("Train Confusion Matrix")
    print(confusion_matrix(y_train, y_pred_train), "\n")

    ## AUC on train
    fpr, tpr, thresholds = roc_curve(y_train, y_pred_prob_train, pos_label=1)
    auc_train = auc(fpr, tpr)
    f1_train = f1_score(y_train, y_pred_train)

    # y_pred_test = model.predict(X_test)
    print("Test Confusion Matrix")
    print(confusion_matrix(y_test, y_pred_test), "\n")

    ## AUC on test
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob_test, pos_label=1)
    auc_test = auc(fpr, tpr)
    f1_test = f1_score(y_test, y_pred_test)
    plot_roc(fpr, tpr)


    print(f"AUC on train: {auc_train:.4f}")
    print(f"AUC on test: {auc_test:.4f}")
    print(f"f1_score on train: {f1_train:.4f}")
    print(f"f1_score on test: {f1_test:.4f}")

    return auc_test