from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from sklearn.metrics import confusion_matrix

#serve per poter installare il pkg pandas, altrimenti ToPandas non va
from pandas import DataFrame

def evaluation_metrics(df_with_predictions, best_model):

    #evaluator non binario
    evaluator = MulticlassClassificationEvaluator(
        labelCol="Fire_Alarm",
        predictionCol="prediction",
    )

    #evaluator binario
    evaluator_binary = BinaryClassificationEvaluator(
        labelCol="Fire_Alarm",
        rawPredictionCol="rawPrediction",
    )

    #metriche
    accuracy = evaluator.setMetricName("accuracy").evaluate(df_with_predictions)
    precision = evaluator.setMetricName("weightedPrecision").evaluate(df_with_predictions)
    recall = evaluator.setMetricName("weightedRecall").evaluate(df_with_predictions)
    f1 = evaluator.setMetricName("f1").evaluate(df_with_predictions)
    auc = evaluator_binary.setMetricName("areaUnderROC").evaluate(df_with_predictions)


    #serve per calcolare la confusion matrix
    predictions_pd = df_with_predictions.toPandas()
    conf_matrix = confusion_matrix(predictions_pd["Fire_Alarm"], predictions_pd["prediction"])
    print("Confusion Matrix:")
    print(conf_matrix)

    #stampa tutti i risultati con evaluator non binario
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Weighted Precision: {precision * 100:.2f}%")
    print(f"Weighted Recall: {recall * 100:.2f}%")
    print(f"F1 Score: {f1 * 100:.2f}%")

    #stampa i risultati con evaluator binario ROC
    print(f"Area UnderROC: {auc * 100:.2f}%")

    #matrice dei pesi
    pesi = best_model.coefficientMatrix
    print("coefficient matrix" + str(pesi))

    #stampa i parametri migliori
    print("Best model:")
    print(f"regParam = {best_model.getOrDefault('regParam')}")
    print(f"elasticNetParam = {best_model.getOrDefault('elasticNetParam')}")
    print(f"threshold = {best_model.getOrDefault('threshold')}")