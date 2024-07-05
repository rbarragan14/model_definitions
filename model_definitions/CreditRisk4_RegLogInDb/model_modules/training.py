import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True

import warnings
import getpass
import os
import random
warnings.filterwarnings('ignore')

#Conexión con Vantage
from teradataml import create_context, DataFrame, get_context, copy_to_sql, in_schema, remove_context, display_analytic_functions

#Este paquete permite que SQLAlchemy se conecte a la base de datos Teradata.
from teradatasqlalchemy.types import * 
from teradatasqlalchemy import INTEGER

#Creación de modelo GLM
from teradataml import GLM, TDGLMPredict


def train(context: ModelContext, **kwargs):
    aoa_create_context()

    feature_names = context.dataset_info.feature_names
    target_name = context.dataset_info.target_names[0]

    # read training dataset from Teradata and convert to pandas
    train_df = DataFrame.from_query(context.dataset_info.sql)
    train_pdf = train_df.to_pandas(all_rows=True)

    # split data into X and y
    X_train = train_pdf[feature_names]
    y_train = train_pdf[target_name]

    print("Starting training...")

    # fit model to training data
    ads=DataFrame.from_query("SELECT a.*, SAMPLEID as sid"
                               " FROM (SELECT * FROM CreditRisk_dataset) a"
                               " SAMPLE RANDOMIZED ALLOCATION 0.3, 0.3, 0.4"
                           )
    #Lista de variables para el análisis
    
    lista01 = ads.columns
    lista02 = [e for e in lista01 if e not in ['id', 'loan_status', 'sid']]
    
    #Separacion de muestras
    
    ads_train=ads[ads["sid"]==1]
    ads_test=ads[ads["sid"]==2]

    # Creacion del Modelo

    model = GLM(input_columns= lista02,
                response_column = "loan_status",
                data = ads_train, 
                family = 'BINOMIAL',
                iter_max = 100,
                learning_rate = 'OPTIMAL',
                momentum = 0.6
               )

    print("Finished training")

    # export model artefacts
    joblib.dump(model, f"{context.artifact_output_path}/model.joblib")

    # we can also save as pmml so it can be used for In-Vantage scoring etc.
    #xgboost_to_pmml(pipeline=model, col_names=feature_names, target_name=target_name,
    #                pmml_f_name=f"{context.artifact_output_path}/model.pmml")

    print("Saved trained model")

    #from xgboost import plot_importance
    #model["xgb"].get_booster().feature_names = feature_names
    #plot_importance(model["xgb"].get_booster(), max_num_features=10)
    #save_plot("feature_importance.png", context=context)

    #feature_importance = model["xgb"].get_booster().get_score(importance_type="weight")

    #record_training_stats(train_df,
    #                      features=feature_names,
    #                      targets=[target_name],
    #                      categorical=[target_name],
    #                      feature_importance=feature_importance,
    #                      context=context)
