import csv
from sklearn.metrics import roc_auc_score

def evaluate_feature_selection_model(labels, predictions, fileToWriteTo, isStart, feature):
    print("number of 'Benign' predictions: ", list(predictions).count(1))
    print("number of 'anomaly' predictions: ", list(predictions).count(-1))

    roc,truePositiveCount,trueNegativeCount,falsePositiveCount,falseNegativeCount,f1_score_of_model, precision_of_model, recall_of_model, true_negative_rate, false_negative_rate, false_positive_rate, changeInAUROC = run_calculations(labels,predictions,isStart,fileToWriteTo)

    print("AUROC Score: ", roc)
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
        print("Change in AUROC compared to Default Model: ", changeInAUROC)

    with open(fileToWriteTo,"a") as csvfile: # writing the results to the output file
        ws = csv.writer(csvfile, delimiter=',')
        if isStart: # writes the header once
            ws.writerow(["Feature Removed","AUROC Score","F1 Score", "Precision", "Recall", "True Negative Rate", "False Negative Rate", "False Positive Rate"])
        ws.writerow([feature,roc, f1_score_of_model, precision_of_model, recall_of_model, true_negative_rate, false_negative_rate, false_positive_rate, changeInAUROC])
    csvfile.close()

def run_calculations(labels, predictions,isStart,fileToWriteTo):
    roc = 1 - roc_auc_score(labels,predictions)
    defaultAUROC = 0

    if not isStart and fileToWriteTo is not None: # gets the AUROC score of the baseline model
        with open(fileToWriteTo, "r") as csvfile:
            defaultReader = csv.reader(csvfile, delimiter=',')
            next(defaultReader)  # skip headers
            defaultResults = next(defaultReader)
            defaultAUROC = float(defaultResults[1])

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
    #calculate the various statistics on the models.
    f1_score_of_model = (2 * truePositiveCount) / (2 * truePositiveCount + falsePositiveCount + falseNegativeCount) if (2 * truePositiveCount + falsePositiveCount + falseNegativeCount) else 0
    precision_of_model = truePositiveCount / (truePositiveCount + falsePositiveCount) if (truePositiveCount + falsePositiveCount) > 0 else 0
    recall_of_model = truePositiveCount / (truePositiveCount + falseNegativeCount) if (truePositiveCount + falseNegativeCount) > 0 else 0
    true_negative_rate = trueNegativeCount / (trueNegativeCount + falsePositiveCount) if (trueNegativeCount + falsePositiveCount) > 0 else 0
    false_negative_rate = falseNegativeCount / (falseNegativeCount + truePositiveCount) if (falseNegativeCount + truePositiveCount) > 0 else 0
    false_positive_rate = falsePositiveCount / (falsePositiveCount + trueNegativeCount) if (falsePositiveCount + trueNegativeCount) > 0 else 0
    if not isStart:
        changeInAUROC = roc - defaultAUROC # if not the default model, calculate the change in AUROC.
    else:
        changeInAUROC = 0

    return roc,truePositiveCount,trueNegativeCount,falsePositiveCount,falseNegativeCount,f1_score_of_model,precision_of_model,recall_of_model,true_negative_rate,false_negative_rate,false_positive_rate,changeInAUROC

def evaluate_hyper_model(labels, predictions, isStart, fileToWriteTo, config):
    roc,truePositiveCount, trueNegativeCount, falsePositiveCount, falseNegativeCount, f1_score_of_model, precision_of_model, recall_of_model, true_negative_rate, false_negative_rate, false_positive_rate, changeInAUROC = run_calculations(labels, predictions, isStart, fileToWriteTo)

    with open(fileToWriteTo, "a") as csvfile: # write the result of a hyperparametered run
        ws = csv.writer(csvfile, delimiter=',')
        if isStart:
            titleString = [f"Config {i+1}" for i in range(len(config))]
            titleString.extend([
                "AUROC Score","F1 Score", "Precision", "Recall", "True Negative Rate",
                "False Negative Rate", "False Positive Rate",
                "Change in AUROC"
            ])
            ws.writerow(titleString)

        dataRow = list(config)#adding the config list to the row
        dataRow.extend([roc,f1_score_of_model, precision_of_model, recall_of_model, true_negative_rate, false_negative_rate,false_positive_rate, changeInAUROC])

        ws.writerow(dataRow)

def final_eval_model(labels, predictions):
    roc,truePositiveCount,trueNegativeCount,falsePositiveCount,falseNegativeCount,f1_score_of_model, precision_of_model, recall_of_model, true_negative_rate, false_negative_rate, false_positive_rate, changeInAUROC = run_calculations(labels,predictions,True,None)
    #print the results
    print("AUROC Score: ", roc)
    print("number of 'Benign' predictions: ", list(predictions).count(1))
    print("number of 'anomaly' predictions: ", list(predictions).count(-1))
    print("confusion matrix:")
    print("\t\t Actual Values")
    print(f'\t\tAnomaly\tBenign')
    print(
        f"Anomaly {truePositiveCount}\t{falsePositiveCount}\t Predicted \nBenign {falseNegativeCount}\t{trueNegativeCount}\t values")
    print("F1 Score: ", f1_score_of_model)
    print("Precision: ", precision_of_model)
    print("Recall: ", recall_of_model)
    print("True Negative Rate: ", true_negative_rate)
    print("False Negative Rate: ", false_negative_rate)
    print("False Positive Rate: ", false_positive_rate)