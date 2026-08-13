import os

# Points to the project root directory (one level above src/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")
RESULT_DIR = os.path.join(OUTPUT_DIR, "results")

MODEL_DIR = os.path.join(BASE_DIR, "models")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
BINARY_MODEL_PATH = os.path.join(MODEL_DIR, "logreg_binary.pkl")
MULTI_MODEL_PATH = os.path.join(MODEL_DIR, "logreg_multi.pkl")

TEST_SIZE = 0.2
RANDOM_STATE = 42