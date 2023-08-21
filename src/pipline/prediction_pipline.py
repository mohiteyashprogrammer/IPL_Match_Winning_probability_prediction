import pandas as pd
import numpy as np
import os
import sys
from src.logger import logging
from dataclasses import dataclass
from src.exception import CustomException

from src.utils import load_object


class PredictPipline:
    def __init__(self):
        pass

    def  predict_probability(self,features):
        try:
            ## This line of path code work i both windos and linex
            preprocessing_path = os.path.join("artifcats","preprocessor.pkl")
            model_path = os.path.join("artifcats","model.pkl")

            preprocessing = load_object(preprocessing_path)
            model = load_object(model_path)

            data_scaled = preprocessing.transform(features)

            pred = model.predict_proba(data_scaled)
            return pred

        except Exception as e:
            logging.info("Error Ocure in Prediction PIPLine")
            raise CustomException(e, sys)