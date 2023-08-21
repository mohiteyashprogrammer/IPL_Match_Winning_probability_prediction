import pandas as pd
import numpy as np
import os
import sys
import pickle
from src.logger import logging
from dataclasses import dataclass
from src.exception import CustomException
from src.utils import save_object,model_traning

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.metrics import accuracy_score,confusion_matrix,roc_auc_score,classification_report


@dataclass
class ModelTraningConfig:
    trainng_model_file_path = os.path.join("artifcats","model.pkl")


class ModelTraning:

    def __init__(self):
        self.model_traning_config = ModelTraningConfig()

    def initated_model_traning(self,train_array,test_array):

        '''
        This Function Will Train Model And Give The Model Pickel file

        '''
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1],
            )

            models = {
                "LogisticRegression":LogisticRegression(solver='liblinear')
            }

            model_report:dict = model_traning(X_train, y_train, X_test, y_test, models)
            print(model_report)
            print("\n***************************************************************\n")
            logging.info(f"Model Report: {model_report}")

            ## To get the best model score from dict
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            print(f"Best Model Found, Model Name : {best_model_name}, R2 Score : {best_model_score}")
            print("\n***************************************************************\n")
            logging.info(f'Best Model Found , Model Name : {best_model_name} , Accuray Score : {best_model_score}')

            save_object(file_path = self.model_traning_config.trainng_model_file_path,
                obj = best_model
            )

            
        except Exception as e:
            logging.info("Exception Occured at Model Traning")
            raise CustomException(e,sys)



