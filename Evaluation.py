from sklearn.metrics import roc_auc_score


def evaluate_model(labels, predictions):
    print("AUC ROC Score: ",roc_auc_score(labels, predictions, multi_class="ovo"))  # print AUROC score
    print("number of 'Benign' predictions: ", list(predictions).count(1))
    print("number of 'anomaly' predictions: ", list(predictions).count(-1))

    # this next section creates a confusion matrix for the results
    truePositiveCount = 0
    falsePositiveCount = 0
    trueNegativeCount = 0
    falseNegativeCount = 0
    counter = 0
    for pred in predictions:  # 1 indicates normal and -1 indicates anomaly
        if labels[counter] != "Benign" and pred == -1:
            truePositiveCount += 1
        elif labels[counter] != "Benign" and pred == 1:
            falsePositiveCount += 1
        elif labels[counter] == "Benign" and pred == -1:
            falseNegativeCount += 1
        else:
            trueNegativeCount += 1
        counter += 1

    print("confusion matrix:")
    print(f"{truePositiveCount}\t {falsePositiveCount} \n {falseNegativeCount} \t {trueNegativeCount}")

    f1_score_of_model = (2 * truePositiveCount) / (2 * truePositiveCount + falsePositiveCount + falseNegativeCount)
    print("F1 Score: ", f1_score_of_model)

    precision_of_model = truePositiveCount / (truePositiveCount + falsePositiveCount)
    print("Precision: ", precision_of_model)

    recall_of_model = truePositiveCount / (truePositiveCount + falseNegativeCount)
    print("Recall: ", recall_of_model)

    true_negative_rate = trueNegativeCount / (trueNegativeCount + falsePositiveCount)
    print("True Negative Rate: ", true_negative_rate)

    false_negetive_rate = falseNegativeCount / (falseNegativeCount + truePositiveCount)
    print("False Negative Rate: ", false_negetive_rate)

    false_positive_rate = falsePositiveCount / (falsePositiveCount + trueNegativeCount)
    print("False Positive Rate: ", false_positive_rate)