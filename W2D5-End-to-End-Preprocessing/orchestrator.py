from src.ingest import load_data
from src.clean import clean_data
from src.preprocess import preprocess_data
from src.logger import log_pipeline
from src.ragas_tests import validate_data

def run_pipeline():

    print("\nSTARTING TITANIC PIPELINE\n")


    # Step 1: Data ingestion
    df = load_data(
        "data/train.csv"
    )


    # Step 2: Cleaning
    df = clean_data(df)


    # Step 3: Preprocessing
    processed_df = preprocess_data(df)


    # Step 4: Data validation
    validate_data(
        processed_df
    )


    # Step 5: MLflow logging
    log_pipeline()


    print(
        "\nPIPELINE COMPLETED SUCCESSFULLY"
    )



if __name__ == "__main__":

    run_pipeline()