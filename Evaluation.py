import csv

def evaluate_model(labels, predictions, fileToWriteTo, isStart, feature):
    print("number of 'Benign' predictions: ", list(predictions).count(1))
    print("number of 'anomaly' predictions: ", list(predictions).count(-1))

    truePositiveCount,trueNegativeCount,falsePositiveCount,falseNegativeCount,f1_score_of_model, precision_of_model, recall_of_model, true_negative_rate, false_negative_rate, false_positive_rate, changeInRecall, changeInTrueNegativeRate, totalChangeInAccuracy = run_calculations(labels,predictions,isStart,fileToWriteTo)

    print("confusion matrix:")
    print("\t\t Actual Values")
    print(f'\t\tAnomaly\tBenign')
    print(f"Anomaly {truePositiveCount}\t{falsePositiveCount}\t Predicted \nBenign {falseNegativeCount}\t{trueNegativeCount}\t values")
    print("F1 Score: ", f1_score_of_model)
    print("Precision: ", precision_of_model)
    print("Recall: ", recall_of_model)
    print("True Negative Rate: ", true_negative_rate)
    print("False Negative Rate: ", false_negative_rate)
    print("False Positive Rate: ", false_positive_rate)

    if not isStart:
        print("Change in Recall compared to Default Model: ", changeInRecall)
        print("Change in True Negative Rate compared to Default Model: ", changeInTrueNegativeRate)

    with open(fileToWriteTo,"a") as csvfile:
        ws = csv.writer(csvfile, delimiter=',')
        if isStart:
            ws.writerow(["Feature Removed","F1 Score", "Precision", "Recall", "True Negative Rate", "False Negative Rate", "False Positive Rate","Change in Recall from Default","Change in TNR from Default", "Total Change in Accuracy"])
        ws.writerow([feature, f1_score_of_model, precision_of_model, recall_of_model, true_negative_rate, false_negative_rate, false_positive_rate, changeInRecall, changeInTrueNegativeRate, totalChangeInAccuracy])
    csvfile.close()

def run_calculations(labels, predictions,isStart,fileToWriteTo):
    defaultRecall = 0
    defaultTrueNegativeRate = 0

    if not isStart:
        with open(fileToWriteTo, "r") as csvfile:
            defaultReader = csv.reader(csvfile, delimiter=',')
            next(defaultReader)  # skip headers
            defaultResults = next(defaultReader)
            defaultRecall = float(defaultResults[3])
            defaultTrueNegativeRate = float(defaultResults[4])
        csvfile.close()

    truePositiveCount = 0
    falsePositiveCount = 0
    trueNegativeCount = 0
    falseNegativeCount = 0
    counter = 0
    for pred in predictions:  # 1 indicates normal and -1 indicates anomaly
        if labels[counter] != 0 and pred == -1:
            truePositiveCount += 1
        elif labels[counter] != 0 and pred == 1:
            falseNegativeCount += 1
        elif labels[counter] == 0 and pred == -1:
            falsePositiveCount += 1
        else:
            trueNegativeCount += 1
        counter += 1

    f1_score_of_model = (2 * truePositiveCount) / (2 * truePositiveCount + falsePositiveCount + falseNegativeCount) if (2 * truePositiveCount + falsePositiveCount + falseNegativeCount) else 0
    precision_of_model = truePositiveCount / (truePositiveCount + falsePositiveCount) if (truePositiveCount + falsePositiveCount) > 0 else 0
    recall_of_model = truePositiveCount / (truePositiveCount + falseNegativeCount) if (truePositiveCount + falseNegativeCount) > 0 else 0
    true_negative_rate = trueNegativeCount / (trueNegativeCount + falsePositiveCount) if (trueNegativeCount + falsePositiveCount) > 0 else 0
    false_negative_rate = falseNegativeCount / (falseNegativeCount + truePositiveCount) if (falseNegativeCount + truePositiveCount) > 0 else 0
    false_positive_rate = falsePositiveCount / (falsePositiveCount + trueNegativeCount) if (falsePositiveCount + trueNegativeCount) > 0 else 0
    if not isStart:
        changeInRecall = recall_of_model - defaultRecall
        changeInTrueNegativeRate = true_negative_rate - defaultTrueNegativeRate
        totalChangeInAccuracy = changeInRecall + changeInTrueNegativeRate
    else:
        changeInRecall = 0
        changeInTrueNegativeRate = 0
        totalChangeInAccuracy = 0

    return truePositiveCount,trueNegativeCount,falsePositiveCount,falseNegativeCount,f1_score_of_model,precision_of_model,recall_of_model,true_negative_rate,false_negative_rate,false_positive_rate,changeInRecall,changeInTrueNegativeRate,totalChangeInAccuracy

def evaluate_hyper_model():
    print("hello")