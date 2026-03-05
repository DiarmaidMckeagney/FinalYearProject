from sklearn.metrics import roc_auc_score


def evaluate_model(labels, predictions, isBeth):
    #print("AUC ROC Score: ",roc_auc_score(labels, predictions, multi_class="ovo"))  # print AUROC score
    print("number of 'Benign' predictions: ", list(predictions).count(1))
    print("number of 'anomaly' predictions: ", list(predictions).count(-1))

    if isBeth:
        labelBenign = 0
    else:
        labelBenign = "Benign"

    # this next section creates a confusion matrix for the results
    truePositiveCount = 0
    falsePositiveCount = 0
    trueNegativeCount = 0
    falseNegativeCount = 0
    counter = 0
    for pred in predictions:  # 1 indicates normal and -1 indicates anomaly
        if labels[counter] != labelBenign and pred == -1:
            truePositiveCount += 1
        elif labels[counter] != labelBenign and pred == 1:
            falseNegativeCount += 1
        elif labels[counter] == labelBenign and pred == -1:
            falsePositiveCount += 1
        else:
            trueNegativeCount += 1
        counter += 1

    print("confusion matrix:")
    print("\t\t Actual Values")
    print(f'\t\tAnomaly\tBenign')
    print(f"Anomaly {truePositiveCount}\t{falsePositiveCount}\t Predicted \nBenign {falseNegativeCount}\t{trueNegativeCount}\t values")
    f1_score_of_model = (2 * truePositiveCount) / (2 * truePositiveCount + falsePositiveCount + falseNegativeCount)
    print("F1 Score: ", f1_score_of_model)

    precision_of_model = truePositiveCount / (truePositiveCount + falsePositiveCount) if (truePositiveCount + falsePositiveCount) > 0 else 0
    print("Precision: ", precision_of_model)

    recall_of_model = truePositiveCount / (truePositiveCount + falseNegativeCount) if (truePositiveCount + falseNegativeCount) > 0 else 0
    print("Recall: ", recall_of_model)

    true_negative_rate = trueNegativeCount / (trueNegativeCount + falsePositiveCount) if (trueNegativeCount + falsePositiveCount) > 0 else 0
    print("True Negative Rate: ", true_negative_rate)

    false_negetive_rate = falseNegativeCount / (falseNegativeCount + truePositiveCount) if (falseNegativeCount + truePositiveCount) > 0 else 0
    print("False Negative Rate: ", false_negetive_rate)

    false_positive_rate = falsePositiveCount / (falsePositiveCount + trueNegativeCount) if (falsePositiveCount + trueNegativeCount) > 0 else 0
    print("False Positive Rate: ", false_positive_rate)