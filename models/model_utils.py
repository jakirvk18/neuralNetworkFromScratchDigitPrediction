import pickle
import os

def save_model(model, path):
    """
    Save a trained model to disk using pickle.

    Args:
        model: Trained model object.
        path (str): File path to save the model.
    """
    try:
        with open(path, 'wb') as f:
            pickle.dump(model, f)
        print(f"[INFO] Model saved to {path}")
    except Exception as e:
        print(f"[ERROR] Failed to save model: {e}")


def load_model(path):
    """
    Load a pickled model from disk.

    Args:
        path (str): File path to load the model from.

    Returns:
        The loaded model object.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at: {path}")
    
    try:
        with open(path, 'rb') as f:
            model = pickle.load(f)
        print(f"[INFO] Model loaded from {path}")
        return model
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        raise
