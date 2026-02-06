from pyspark.ml.classification import LinearSVC
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

def prediction_function(train_dataset, test_dataset):

    #classifier di linear
    classifier_SVC = LinearSVC(
        featuresCol= "scaled_features",
        labelCol= "Fire_Alarm"
    )

    #paramgrid di lieanr svc
    paramgrid_svc = (
        ParamGridBuilder()
        .addGrid(classifier_SVC.regParam, [0.01])
        .addGrid(classifier_SVC.maxIter, [50])
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
        estimator=classifier_SVC,
        estimatorParamMaps=paramgrid_svc,
        evaluator=evaluator,
        trainRatio=0.8,
        seed=404,
    )

    #crea il set di modelli, trova quello migliore e fai il training sul df di test
    set_training_model = tvs.fit(train_dataset)
    best_model = set_training_model.bestModel
    df_with_predictions = best_model.transform(test_dataset)


    return df_with_predictions, best_model