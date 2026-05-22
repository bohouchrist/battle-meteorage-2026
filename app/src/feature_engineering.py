"""Feature engineering."""
import numpy as np
import pandas as pd

def build_survival(df, max_gap=30):
    """Construire variables de survie (duree, event)."""
    df = df.copy()
    df = df.sort_values(['airport', 'airport_alert_id', 'date'])

    g = df.groupby(['airport', 'airport_alert_id'])
    df['next_date'] = g['date'].shift(-1)
    df['duree'] = (df['next_date'] - df['date']).dt.total_seconds() / 60

    df['event'] = df['duree'].notna() & (df['duree'] <= max_gap)
    df.loc[~df['event'], 'duree'] = max_gap
    df['duree'] = df['duree'].clip(lower=0.01)

    return df

def add_temporal_features(df):
    """Features temporelles cycliques."""
    df = df.copy()

    # Heure
    h = df['date'].dt.hour + df['date'].dt.minute / 60
    df['h_cos'] = np.cos(2 * np.pi * h / 24)
    df['h_sin'] = np.sin(2 * np.pi * h / 24)

    # Jour de l'année
    doy = df['date'].dt.dayofyear
    df['doy_cos'] = np.cos(2 * np.pi * doy / 365)
    df['doy_sin'] = np.sin(2 * np.pi * doy / 365)

    # Saison
    df['saison'] = ((df['date'].dt.month % 12) // 3) + 1

    return df

def add_silence_features(df):
    """Features silence/fréquence."""
    df = df.copy()
    g = df.groupby(['airport', 'airport_alert_id'])

    prev = g['date'].shift(1)
    df['silence_min'] = ((df['date'] - prev).dt.total_seconds() / 60).fillna(30).clip(0, 60)
    df['freq_5min'] = (1 / df['silence_min'].clip(lower=0.5)).clip(upper=10)

    return df

def add_distance_features(df):
    """Features distance."""
    df = df.copy()
    g = df.groupby(['airport', 'airport_alert_id'])

    df['dist_centre'] = df['dist']
    df['dist_avg_5'] = g['dist'].transform(lambda x: x.rolling(5, min_periods=1).mean())
    df['dist_min_so_far'] = g['dist'].cummin()

    return df

def add_alert_maturity(df):
    """Features maturité alerte."""
    df = df.copy()
    g = df.groupby(['airport', 'airport_alert_id'])

    df['rang'] = g.cumcount()
    df['rang_norm'] = df['rang'] / g['rang'].transform('max').clip(lower=1)

    return df

def build_features(df):
    """Pipeline complet de feature engineering."""
    df = df.copy()

    df = add_temporal_features(df)
    df = add_silence_features(df)
    df = add_distance_features(df)
    df = add_alert_maturity(df)

    # Imputation simple
    features = [
        'h_cos', 'h_sin', 'doy_cos', 'doy_sin', 'saison',
        'dist_centre', 'dist_avg_5', 'dist_min_so_far',
        'silence_min', 'freq_5min', 'rang', 'rang_norm',
    ]

    for col in features:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    return df, features
