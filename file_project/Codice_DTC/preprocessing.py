from pyspark.ml.feature import VectorAssembler


def preprocessing_dataset(df):

    #tutte le colonne che vengono usate per allenare il modello
    feature_cols = [
        "Temperature_C",
        "Humidity_prc",
        "TVOC_ppb",
        "eCO2_ppm",
        "Raw_H2",
        "Raw_Ethanol",
        "Pressure_hPa",
        "PM1_0",
        "PM2_5",
        "NC0_5",
        "NC1_0",
        "NC2_5"
    ]

    #Aggiungo al dataframe la colonna delle features
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    df_with_features = assembler.transform(df)

    #faccio lo split nei due dataset per il training e il test
    df_finale_train, df_finale_test = df_with_features.randomSplit([0.8, 0.2], seed=404)

    return df_finale_train, df_finale_test