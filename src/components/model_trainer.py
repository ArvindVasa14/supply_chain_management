from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from dataclasses import dataclass
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import accuracy_score
import logging
import pickle
import numpy as np

from data_transformation import DataTransformation

@dataclass
class ModelTrainerConfig:
    train_data_path= r'artifacts\data\train_data.csv'
    test_data_path= r'artifacts\data\test_data.csv'
    model_save_path= r'artifacts\model.pkl'
    preprocessor_path= r'artifacts\preprocessor.pkl'

class ModelTrainer:
    def __init__(self) -> None:
        self.trainer_config= ModelTrainerConfig()

    def evaluate_model(self, actual, predicted):
        mse = mean_squared_error(actual, predicted)
        mae = mean_absolute_error(actual, predicted)
        rmse= np.sqrt(mse)
        r2 = r2_score(actual, predicted)
        
        return r2, mae, mse, rmse

    def initiate_model_training(self):
        train_data= pd.read_csv(self.trainer_config.train_data_path)
        test_data= pd.read_csv(self.trainer_config.test_data_path)

        with open(self.trainer_config.preprocessor_path, 'rb') as preprocessor_file:
            preprocessor= pickle.load(preprocessor_file)

        train_data= pd.DataFrame(preprocessor.fit_transform(train_data))
        test_data= pd.DataFrame(preprocessor.fit_transform(test_data))

        X_train, y_train= train_data.iloc[:,:-1], train_data.iloc[:,-1]
        X_test, y_test= train_data.iloc[:,:-1], train_data.iloc[:,-1]

        models= {
        'LinearRegression':LinearRegression(),
        'Lasso':Lasso(max_iter=1000),
        'Ridge':Ridge(),
        'Elasticnet':ElasticNet(),
        }

        r2_max= 0
        
        for model_name, model in models.items():
            model.fit(X_train, y_train)
        #     y_train_pred= model.predict(X_train)
            y_pred_test= model.predict(X_test)
            r2, mae, mse, rmse= self.evaluate_model(y_test, y_pred_test)
        
            logging.info(f'{model_name} Training Performance, MSE: {mse:.2f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2 score: {r2*100:.2f}')
            print(model_name, r2)
            if r2>r2_max:
                r2_max= r2
                final_model= model
                final_model_name= model_name

        print(final_model_name)    
                

        with open(self.trainer_config.model_save_path, 'wb') as model_file:
            pickle.dump(final_model, model_file)
            
        return final_model


train_obj= ModelTrainer()
train_obj.initiate_model_training()
