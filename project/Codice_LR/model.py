from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit, CrossValidator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator

def prediction_function(train_dataset, test_dataset):

    #usa l'algoritmo di LogistiRegression per fare classification
    classifier = LogisticRegression(
        featuresCol="scaled_features",
        labelCol="Fire_Alarm"
    )

    #hyperparameter imposta come si deve comportare l'algoritmo
    paramgrid =  (
        ParamGridBuilder()
        .addGrid(classifier.regParam, [0.001])
        .addGrid(classifier.elasticNetParam, [1.0])
        .addGrid(classifier.threshold, [0.6])
        .build()
    )

    #serve per capire il modello migliore in base a f1 tra quelli creati con i valori passati nella param. grid
    evaluator = MulticlassClassificationEvaluator(
        labelCol="Fire_Alarm",
        predictionCol="prediction",
        metricName="f1",
    )

    #training con train Validation
    tvs = TrainValidationSplit(
        estimator=classifier,
        estimatorParamMaps=paramgrid,
        evaluator=evaluator,
        trainRatio=0.8,
        seed=404,
    )

    #crea il set di modelli, trova quello migliore e fai il training sul df di test
    set_training_model = tvs.fit(train_dataset)
    best_model = set_training_model.bestModel
    df_with_predictions = best_model.transform(test_dataset)


    return df_with_predictions, best_model