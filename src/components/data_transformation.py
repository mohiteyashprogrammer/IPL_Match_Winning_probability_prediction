import pandas as pd
import numpy as np
import os
import sys
from src.logger import logging
from dataclasses import dataclass
from src.exception import CustomException
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifcats","preprocessor.pkl")


class DataTransformation:

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()


    def get_data_transformation_object(self):

        '''
        This Function Give Preprocessor Object to transform Data
        
        '''
        try:
            logging.info("Start Data Transformation")

            catigorical_features = ['batting_team', 'bowling_team', 'city']

            numerical_features = ['runs_left', 'balls_left', 'wickets_left', 'total_runs_x', 'run_rate',
                                    'require_run_rate']
            

            # Create numerical and catigorical pipline
            num_pipline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
            ]
        )

            cato_pipline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("onehot",OneHotEncoder(sparse=False,handle_unknown="ignore",drop="first")),
                    ("scaler",StandardScaler(with_mean=False))
            ]
        )       

            ## get Preprocessor object
            preprocessor = ColumnTransformer([
                ("num_pipline",num_pipline,numerical_features),
                ("cato_pipline",cato_pipline,catigorical_features)
            ])

            return preprocessor

            logging.info("Pipline Complite")

        except Exception as e:
            logging.info("Error Occured Data Transformation Stage")
            raise CustomException(e, sys)


    
    def initated_data_transformation(self,train_path,test_path):

        '''
        This Method Will TransformaData And Return It

        '''

        try:
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)

            logging.info("Read Traning And Test Data Complited")
            logging.info(f'Train Dataframe Head : \n{train_data.head().to_string()}')
            logging.info(f'Test Dataframe Head  : \n{test_data.head().to_string()}')


            logging.info("Obtaining Preprosser object")


            preprocessor_obj = self.get_data_transformation_object()

            target_columns_name = "result"
            drop_columns = [target_columns_name]

            ## spliting dependent and indipend veriable
            input_features_train_data = train_data.drop(drop_columns,axis=1)
            target_feature_train_data = train_data[target_columns_name]

            ## spliting dependent and indipend veriable
            input_features_test_data = test_data.drop(drop_columns,axis=1)
            target_feature_test_data = test_data[target_columns_name]

            #apply preprocessor object to transform data
            input_feature_train_arr = preprocessor_obj.fit_transform(input_features_train_data)
            input_feature_test_arr = preprocessor_obj.transform(input_features_test_data)

            logging.info("Apply Preprocessor Object on train and test Data")

            ## Convert in to array to become fast
            train_array = np.c_[input_feature_train_arr,np.array(target_feature_train_data)]
            test_array = np.c_[input_feature_test_arr,np.array(target_feature_test_data)]

            ## Callling Save object to save preprocessor pkl file
            save_object(file_path = self.data_transformation_config.preprocessor_obj_file_path,
            obj = preprocessor_obj)

            logging.info("Preprocessor Object File is Save")

            return (
                train_array,
                test_array,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            logging.info("Error Occured Data Transformation Stage")
            raise CustomException(e, sys)