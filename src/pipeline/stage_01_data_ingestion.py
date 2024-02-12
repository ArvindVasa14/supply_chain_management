
from src.components.data_ingestion import DataIngestion
from logger import logging
import os
STAGE_NAME="DATA INGESTION"

class DataIngestionPipeline:
    def __init__(self) -> None:
        pass

    def main(self):
        data_ingestion= DataIngestion()
        data_ingestion.initiate_train_test_data()


if __name__ == '__main__':
    try:
        print(os.getcwd())
        logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataIngestionPipeline()
        obj.main()
        logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logging.exception(e)
        raise e
