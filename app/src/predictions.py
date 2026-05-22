"""Inférence et extraction TTC."""
import numpy as np
import pandas as pd

def extract_ttc(surv_fns, max_ttc=30.0):
    """Extraire TTC depuis courbes de survie."""
    surv_fns = list(surv_fns)
    times = np.linspace(0, 30, 121)

    # Matrice (n_obs x n_times)
    surv_mat = np.row_stack([
        np.interp(times, sf.x, sf.y, right=sf.y[-1])
        for sf in surv_fns
    ])

    # TTC médiane (S < 0.5)
    idx = (surv_mat < 0.5).argmax(axis=1)
    idx[surv_mat[:, -1] >= 0.5] = len(times) - 1
    ttc = np.minimum(times[idx], max_ttc)

    # Confiance @ 30 min
    conf = surv_mat[:, np.searchsorted(times, 30.0)]

    return {
        'ttc_median': ttc,
        'confidence': conf,
        'gain': 30 - ttc,
    }

def make_predictions(model, X, df_context=None):
    """Générer prédictions au format jury."""
    surv_fns = list(model.predict_survival_function(X))
    ttc_dict = extract_ttc(surv_fns)

    df_pred = pd.DataFrame(ttc_dict)

    if df_context is not None:
        for col in ['airport', 'airport_alert_id', 'date']:
            if col in df_context.columns:
                df_pred[col] = df_context[col].values

        df_pred['prediction_date'] = df_pred['date']
        df_pred['predicted_date_end_alert'] = (
            df_pred['date'] + pd.to_timedelta(df_pred['ttc_median'], unit='m')
        )

    return df_pred
