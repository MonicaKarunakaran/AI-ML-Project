import pandas as pd


def load_data(path):

    print("Loading dataset...")

    df = pd.read_csv(r"C:\Users\Monika\CynarisInternship\AI-ML-Project\W2D5-End-to-End-Preprocessing\data\train.csv")

    print("Dataset Loaded")
    print("Shape:", df.shape)

    return df