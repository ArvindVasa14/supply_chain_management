from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle
import pandas as pd

@dataclass
class DataTransformationConfig:
    train_data_path= r'artifacts\data\train_data.csv'
    preprocessor_save_path= r'artifacts\preprocessor.pkl'

class DataTransformation:
    def __init__(self) -> None:
        self.transform_config= DataTransformationConfig()

    def initiate_data_transformation(self):
        categorical_columns= ['Location_type', 'WH_capacity_size', 'zone', 'WH_regional_zone',
       'wh_owner_type', 'approved_wh_govt_certificate']
        columns= ['Location_type', 'WH_capacity_size', 'zone', 'WH_regional_zone',
       'num_refill_req_l3m', 'transport_issue_l1y', 'Competitor_in_mkt',
       'retail_shop_num', 'wh_owner_type', 'distributor_num', 'flood_impacted',
       'flood_proof', 'electric_supply', 'dist_from_hub', 'workers_num',
       'storage_issue_reported_l3m', 'temp_reg_mach',
       'approved_wh_govt_certificate', 'wh_breakdown_l3m', 'govt_check_l3m',
        ]
        columns= ['num_refill_req_l3m', 'transport_issue_l1y', 'Competitor_in_mkt',
       'retail_shop_num', 'distributor_num', 'flood_impacted',
       'flood_proof', 'electric_supply', 'dist_from_hub', 'workers_num',
       'storage_issue_reported_l3m', 'temp_reg_mach',
        'wh_breakdown_l3m', 'govt_check_l3m',
        ]
        nominal_features= ['Location_type','zone','WH_regional_zone','wh_owner_type']
        ordinal_features= ['WH_capacity_size','approved_wh_govt_certificate']
            

        preprocessor= ColumnTransformer(
            transformers=[
                ('OneHotEncoding', OneHotEncoder(), nominal_features),
                ('LabelEncoding', OrdinalEncoder(), ordinal_features),
                ('StandardScaler', StandardScaler(), columns),
                
            ], remainder='passthrough'
        )
        
        with open(self.transform_config.preprocessor_save_path, 'wb') as preprocessor_file:
            pickle.dump(preprocessor, preprocessor_file)

        return preprocessor
        
    def transform_data(self):

        train_data= pd.read_csv(r'artifacts\data\train_data.csv')
        test_data= pd.read_csv(r'artifacts\data\test_data.csv')

        with open(r'artifacts\preprocessor.pkl', 'rb') as preprocessor_file:
            preprocessor= pickle.load(preprocessor_file)

        train_data= pd.DataFrame(preprocessor.fit_transform(train_data))
        test_data= pd.DataFrame(preprocessor.fit_transform(test_data))

        X_train, y_train= train_data.iloc[:,:-1], train_data.iloc[:,-1]
        X_test, y_test= train_data.iloc[:,:-1], train_data.iloc[:,-1]

obj= DataTransformation()
obj.initiate_data_transformation()
obj.transform_data()