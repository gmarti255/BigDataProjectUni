from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

def prediction_function(train_dataset, test_dataset):

    #usa l'algoritmo DecisonTreeClass per fare classification
    classifier_DTC = DecisionTreeClassifier(
        featuresCol="features",
        labelCol="Fire_Alarm",
    )

    #paramatri da settare per l'algoritmo DTC
    paramgrid_DTC = (
        ParamGridBuilder()
        .addGrid(classifier_DTC.maxDepth, [9])
        .addGrid(classifier_DTC.minInstancesPerNode, [5])
        .build()
    )

    #serve per capire il modello migliore, in base a f1, tra quelli creati con i valori passati nella paramgrid
    evaluator = MulticlassClassificationEvaluator(
        labelCol="Fire_Alarm",
        predictionCol="prediction",
        metricName="f1",
    )

    #training con train Validation
    tvs = TrainValidationSplit(
        estimator=classifier_DTC,
        estimatorParamMaps=paramgrid_DTC,
        evaluator=evaluator,
        trainRatio=0.8,
        seed=404,
    )

    #crea il set di modelli, trova quello migliore e fai il training sul df di test
    set_training_model = tvs.fit(train_dataset)
    best_model = set_training_model.bestModel
    df_with_predictions = best_model.transform(test_dataset)


    return df_with_predictions, best_model