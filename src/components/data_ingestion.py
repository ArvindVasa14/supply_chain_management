import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
import os
import warnings
warnings.filterwarnings('ignore')
from logger import logging
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    data_path= r'artifacts\data\data.csv'
    train_data_path= r'artifacts\data\train_data.csv'
    test_data_path= r'artifacts\data\test_data.csv'


class DataIngestion:
    def __init__(self) -> None:
        self.ingestion_config= DataIngestionConfig()

    # def download_file(self):
    #     if not os.path.exists(self.config.local_data_file):
    #         filename, headers = request.urlretrieve(
    #             url = self.config.source_URL,
    #             filename = self.config.local_data_file
    #         )
    #         logging.info(f"{filename} download! with following info: \n{headers}")
    #     else:
    #         logging.info(f"File already exists of size: {get_size(Path(self.config.local_data_file))}")  


    # def extract_zip_file(self):
    #     """
    #     zip_file_path: str
    #     Extracts the zip file into the data directory
    #     Function returns None
    #     """
    #     unzip_path = self.config.unzip_dir
    #     os.makedirs(unzip_path, exist_ok=True)
    #     with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
    #         zip_ref.extractall(unzip_path)

    def initiate_train_test_data(self):
        data= pd.read_csv(self.ingestion_config.data_path)
        train_data, test_data= train_test_split(data, test_size=0.2, random_state= 101)
        train_data.to_csv(self.ingestion_config.train_data_path)
        test_data.to_csv(self.ingestion_config.test_data_path)

        return (
            os.path.join(self.ingestion_config.train_data_path),
            os.path.join(self.ingestion_config.train_data_path)    
                )


# obj = DataIngestion()
# obj.initiate_train_test_data()
