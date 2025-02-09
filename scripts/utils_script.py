from tqdm import tqdm
import math

def estimate_scores(TP, FP, TN, FN):

    """ 
    compute all binary classification 
    """

    # accuracy
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    # precision
    precision = TP / (TP + FP)
    # recall / sensitivity
    recall = TP / (TP + FN)
    # specificity
    specifity = TN / (TN + FP)
    # f1-score
    F1 = 2 * precision * recall / (precision + recall)
    # false positive rate
    FPR = FP / (FP + TN)
    # false negative rate
    FNR = FN / (FN + TP)
    # matthews correlation coefficient
    MCC = (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))

    return accuracy, precision, recall, specifity, F1, FPR, FNR, MCC


def compute(preds, labels):
    TP, FP, TN, FN = 0, 0, 0, 0
    assert len(preds) == len(labels)
    for (pred, label) in tqdm(zip(preds, labels), total=len(preds)):
        if pred == 0 and label == 1:
            FN += 1
        elif pred == 1 and label == 1:
            TP += 1
        elif pred == 0 and label == 0:
            TN += 1
        elif pred == 1 and label == 0:
            FP += 1
        else:
            raise ValueError("error value {} and {}".format(pred, label))
    return TP, FP, TN, FN
