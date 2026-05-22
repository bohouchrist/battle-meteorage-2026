"""Configuration centralisée de l'app Streamlit."""
from pathlib import Path

ROOT = Path(__file__).parent.parent

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CHEMINS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MODEL_PATH = ROOT / "models" / "gbs_v6_model.pkl"
SCALER_PATH = ROOT / "models" / "gbs_v6_scaler.pkl"
METADATA_PATH = ROOT / "models" / "gbs_v6_metadata.json"
DATA_TEST_PATH = ROOT / "models" / "test_data.parquet"
LOG_FILE = ROOT / "logs" / "app.log"

# Créer les répertoires
MODEL_PATH.parent.mkdir(exist_ok=True, parents=True)
LOG_FILE.parent.mkdir(exist_ok=True, parents=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FEATURES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FEATURES = [
    'h_cos', 'h_sin', 'doy_cos', 'doy_sin', 'saison',
    'dist_centre', 'dist_avg_5', 'dist_min_so_far',
    'silence_min', 'freq_5min', 'rang', 'rang_norm',
]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# JURY CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

JURY_CONFIG = {
    'max_gap_minutes': 30,
    'distance_critique_km': 3,
    'risque_max': 0.02,
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# COULEURS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

COLORS = {
    'success': '#10B981',
    'warning': '#F59E0B',
    'danger': '#EF4444',
    'info': '#3B82F6',
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SAISONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SAISON_LABELS = {
    1: 'DJF (Hiver)',
    2: 'MAM (Printemps)',
    3: 'JJA (Été)',
    4: 'SON (Automne)',
}
