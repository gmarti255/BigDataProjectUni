from pyspark.sql import SparkSession

from model import prediction_function
from preprocessing import preprocessing_dataset
from evaluation import evaluation_metrics

#crea la sessione spark e importa il dataset
spark = SparkSession.builder.appName("FireDetection").getOrCreate()
URL_INPUT = "path_del_dataset.csv"
fireDs = spark.read.csv(URL_INPUT, header=True, inferSchema=True)

#richiamo la funzione che mi permette di fare preprocessing sui dati
train_dataset, test_dataset = preprocessing_dataset(fireDs)

#chiamo la funzione per fare training e predizione sui dataset
df_with_prediction, best_model= prediction_function(train_dataset, test_dataset)

#funzione che valuta tutte le metriche sul dataset di test
evaluation_metrics(df_with_prediction, best_model)

spark.stop()
