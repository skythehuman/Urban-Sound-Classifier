"""
classifier.py — Urban Sound Classification Pipeline
=====================================================
Architecture: ResNet-based DCNN (4-fold × 2 features = 8 models)
Features:     Mel-spectrogram (n_mels=40) + MFCC (n_mfcc=40)
Ensemble:     Average probabilities across all 8 models
"""

import os
import warnings
import numpy as np
import librosa

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
warnings.filterwarnings("ignore")

import tensorflow as tf
tf.get_logger().setLevel("ERROR")

# ── Constants (from notebook) ────────────────────────────────────────
SR = 20000           # Sample rate
N_MELS = 40          # n_mels = n_mfcc = size
N_MFCC = 40
PAD_SIZE = 2344      # max_length from training data
N_FOLDS = 4          # 4-fold cross-validation

CLASS_NAMES = [
    "Airplane", "Blast_Fan", "Car_Driving", "Car_Horn",
    "Compressor", "Concrete_Pump", "Dog_Barking", "Engine_Idling",
    "Helicopter", "Jackhammer", "Motor_Driving", "Motor_Horn",
    "Pile_Driver", "Rack_Cutting_Machine", "Siren",
    "Street_Music", "Train",
]

# Noise source taxonomy (Level 1 → Level 2 → Level 3)
TAXONOMY = {
    "Transportation": {
        "Air":     ["Airplane", "Helicopter"],
        "Railway": ["Train"],
        "Road":    ["Car_Driving", "Car_Horn", "Motor_Driving",
                    "Motor_Horn", "Engine_Idling"],
        "Signal":  ["Siren"],
    },
    "Industrial Activity": {
        "Industry":     ["Blast_Fan", "Compressor", "Rack_Cutting_Machine"],
        "Construction": ["Pile_Driver", "Concrete_Pump", "Jackhammer"],
    },
    "Music":  {"Outdoor": ["Street_Music"]},
    "Animal": {"Domestic": ["Dog_Barking"]},
}

# Friendly display names
DISPLAY_NAMES = {
    "Airplane": "✈️ Airplane",
    "Blast_Fan": "🌀 Blast Fan",
    "Car_Driving": "🚗 Car Driving",
    "Car_Horn": "📯 Car Horn",
    "Compressor": "⚙️ Compressor",
    "Concrete_Pump": "🏗️ Concrete Pump",
    "Dog_Barking": "🐕 Dog Barking",
    "Engine_Idling": "🚙 Engine Idling",
    "Helicopter": "🚁 Helicopter",
    "Jackhammer": "🔨 Jackhammer",
    "Motor_Driving": "🏍️ Motorcycle",
    "Motor_Horn": "📢 Motorcycle Horn",
    "Pile_Driver": "🔩 Pile Driver",
    "Rack_Cutting_Machine": "🪚 Rack Cutting Machine",
    "Siren": "🚨 Siren",
    "Street_Music": "🎵 Street Music",
    "Train": "🚆 Train",
}


def get_category(class_name: str) -> tuple[str, str]:
    """Return (Level 1, Level 2) category for a class name."""
    for l1, l2_dict in TAXONOMY.items():
        for l2, classes in l2_dict.items():
            if class_name in classes:
                return l1, l2
    return "Unknown", "Unknown"


# ── Feature Extraction (exact copy from notebook) ────────────────────

def random_pad(features: np.ndarray, pad_size: int,
               is_mfcc: bool = False) -> np.ndarray:
    """
    Pad feature matrix to fixed width with random left/right split.
    Normalization order differs for Mel vs MFCC (matches notebook).

    For MFCC:  pad first → then normalize
    For Mel:   normalize first → then pad
    """
    pad_width = pad_size - features.shape[1]

    if pad_width <= 0:
        # Truncate if longer than pad_size
        features = features[:, :pad_size]
        local_max, local_min = features.max(), features.min()
        if local_max == local_min:
            return np.zeros_like(features)
        return (features - local_min) / (local_max - local_min)

    rand = np.random.rand()
    left = int(pad_width * rand)
    right = pad_width - left

    if is_mfcc:
        features = np.pad(features, ((0, 0), (left, right)), mode="constant")
        local_max, local_min = features.max(), features.min()
        if local_max == local_min:
            return np.zeros_like(features)
        features = (features - local_min) / (local_max - local_min)
    else:
        local_max, local_min = features.max(), features.min()
        if local_max == local_min:
            features = np.zeros_like(features)
        else:
            features = (features - local_min) / (local_max - local_min)
        features = np.pad(features, ((0, 0), (left, right)), mode="constant")

    return features


def extract_features(audio: np.ndarray,
                     sr: int = SR) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract Mel-spectrogram and MFCC features from raw audio.
    Returns two arrays of shape (1, 40, 2344, 1) ready for model input.
    """
    # Mel-spectrogram
    mels = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=N_MELS)
    mels = librosa.power_to_db(mels, ref=np.max)
    mels = random_pad(mels, PAD_SIZE, is_mfcc=False)

    # MFCC
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
    mfcc = random_pad(mfcc, PAD_SIZE, is_mfcc=True)

    # Reshape to (1, 40, 2344, 1) for model input
    mels = mels[np.newaxis, ..., np.newaxis].astype(np.float64)
    mfcc = mfcc[np.newaxis, ..., np.newaxis].astype(np.float64)

    return mels, mfcc


def get_mel_spectrogram(audio: np.ndarray, sr: int = SR) -> np.ndarray:
    """Return raw Mel-spectrogram (dB) for visualization."""
    mels = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=N_MELS)
    return librosa.power_to_db(mels, ref=np.max)


def get_mfcc(audio: np.ndarray, sr: int = SR) -> np.ndarray:
    """Return raw MFCC matrix for visualization."""
    return librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)


# ── Model Loading ────────────────────────────────────────────────────

@tf.function(reduce_retracing=True)
def _predict(model, x):
    return model(x, training=False)


def load_ensemble(model_dir: str = "models") -> tuple[list, list]:
    """
    Load 4 Mel models + 4 MFCC models.
    Returns (mel_models, mfcc_models).
    """
    mel_models = []
    mfcc_models = []

    for fold in range(N_FOLDS):
        mel_path = os.path.join(model_dir, f"model_res_test_0214-3_mels_{fold}.hdf5")
        mfcc_path = os.path.join(model_dir, f"model_res_test_0214-3_mfcc_{fold}.hdf5")

        if os.path.exists(mel_path):
            mel_models.append(tf.keras.models.load_model(mel_path, compile=False))
        if os.path.exists(mfcc_path):
            mfcc_models.append(tf.keras.models.load_model(mfcc_path, compile=False))

    return mel_models, mfcc_models


def ensemble_predict(mel_models: list, mfcc_models: list,
                     mel_features: np.ndarray,
                     mfcc_features: np.ndarray) -> np.ndarray:
    """
    Run ensemble prediction: average probabilities across all models.
    Returns array of shape (17,) with class probabilities.
    """
    all_preds = []

    for model in mel_models:
        pred = model.predict(mel_features, verbose=0)
        all_preds.append(pred[0])

    for model in mfcc_models:
        pred = model.predict(mfcc_features, verbose=0)
        all_preds.append(pred[0])

    # Average across all fold models
    ensemble = np.mean(all_preds, axis=0)
    return ensemble
