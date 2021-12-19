from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def scoring_function_sensitiv(y_true, y_pred) :

    cnf_matrix = confusion_matrix(y_true, y_pred)

    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix) 
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)
    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)

    a = (TP/(TP+FN))

    result  = 0
    for i in range(len(a)) :
        result= result +a[i]

    return result/4