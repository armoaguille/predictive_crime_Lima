"""
forecast_utils.py
------------------
Utilidades compartidas por 07_optuna_tuning.ipynb y 08_final_refit_eval.ipynb.

Guardar este archivo en Modelado/ (junto a los demás notebooks) e importar con:
    import sys; sys.path.append(".")
    from forecast_utils import forecast_recursive_vectorized, burn_in_splits, rmse, mae, poisson_deviance

Contiene:
- forecast_recursive_vectorized: misma logica que `forecasting_recursivo` de los
  notebooks 04/05, pero opera semana a semana con tablas anchas (pivot) en vez de
  iterrows() + filtrado booleano fila por fila. Mismo resultado, mucho mas rapido
  (necesario porque Optuna va a llamar esta funcion cientos de veces).
- burn_in_splits: TimeSeriesSplit "consciente" de que count_lag_52w y
  media_historica necesitan >= 52 semanas reales de historia antes de ser
  confiables. Evita folds tipo "fold 1" donde esos features quedan en 0/NaN
  para casi todas las filas, que es lo que estaba inflando tu RMSE de CV.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Metricas
# ---------------------------------------------------------------------------

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def poisson_deviance(y_true, y_pred, eps=1e-10):
    y_pred = np.clip(np.asarray(y_pred, dtype=float), eps, None)
    y_true = np.asarray(y_true, dtype=float)
    term1 = np.where(y_true > 0, y_true * np.log(y_true / y_pred), 0.0)
    return float(2 * np.sum(term1 - (y_true - y_pred)))


def mcfadden_r2(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    dev_model = poisson_deviance(y_true, y_pred)
    dev_null = poisson_deviance(y_true, np.full_like(y_true, y_true.mean()))
    return 1 - dev_model / dev_null


# ---------------------------------------------------------------------------
# Forecast recursivo vectorizado
# ---------------------------------------------------------------------------

def _build_cell_id(df: pd.DataFrame) -> pd.Series:
    return df["distrito_hecho"].astype(str) + "||" + df["turno_hecho"].astype(str)


def forecast_recursive_vectorized(
    modelo,
    df_future: pd.DataFrame,
    df_history: pd.DataFrame,
    feature_cols: list[str],
    fillna_value: float | None = 0.0,
):
    """
    Prediccion multi-step recursiva, vectorizada por semana.

    Parametros
    ----------
    modelo : objeto con .predict(X) -> array
    df_future : filas a predecir (holdout o test de un fold), con TODAS las
        columnas de feature_cols ya presentes (las no-recursivas se copian
        tal cual; las recursivas -lags/rolling/media_historica- se recalculan).
    df_history : filas de historia real conocida (train o dev del fold),
        con count_robos verdadero.
    feature_cols : columnas en el orden que espera el modelo.
    fillna_value : con que rellenar cuando no hay suficiente historia
        (52 semanas para lag_52w, etc). Usa None para dejar NaN y que
        XGBoost/LightGBM lo manejen nativamente (recomendado); usa 0.0
        para replicar el comportamiento original de los notebooks 04/05.

    Devuelve
    --------
    y_true, y_pred : arrays en el mismo orden que df_future.index
    rmse_por_semana : lista de dicts {semana, rmse}
    """
    df_history = df_history.copy()
    df_future = df_future.copy()
    df_history["cell_id"] = _build_cell_id(df_history)
    df_future["cell_id"] = _build_cell_id(df_future)

    todas_celdas = sorted(set(df_history["cell_id"]) | set(df_future["cell_id"]))

    # tabla ancha semana_global x celda -> count_robos (se extiende con cada prediccion)
    wide = (
        df_history.pivot_table(index="semana_global", columns="cell_id",
                                values="count_robos", aggfunc="first")
        .reindex(columns=todas_celdas)
    )

    # mapeo semana_global -> iso_week (para media_historica)
    calendario = (
        pd.concat([df_history[["semana_global", "iso_week"]],
                   df_future[["semana_global", "iso_week"]]])
        .drop_duplicates("semana_global")
        .set_index("semana_global")["iso_week"]
    )

    semanas = sorted(df_future["semana_global"].unique())
    pred_por_idx, true_por_idx = {}, {}
    rmse_por_semana = []

    for s in semanas:
        df_sem = df_future[df_future["semana_global"] == s].copy()
        iw = calendario.loc[s]
        hist_prev = wide.loc[wide.index < s]

        # --- lags ---
        for k in (1, 2, 4, 8, 52):
            if (s - k) in wide.index:
                fila = wide.loc[s - k]
            else:
                fila = pd.Series(np.nan, index=wide.columns)
            vals = df_sem["cell_id"].map(fila)
            df_sem[f"count_lag_{k}w"] = vals if fillna_value is None else vals.fillna(fillna_value)

        # --- rolling mean / std (ultimas k semanas ANTES de s) ---
        for k in (4, 8, 12):
            media = hist_prev.tail(k).mean(axis=0)
            vals = df_sem["cell_id"].map(media)
            df_sem[f"rolling_mean_{k}w"] = vals if fillna_value is None else vals.fillna(fillna_value)
        for k in (4, 8):
            std = hist_prev.tail(k).std(axis=0, ddof=0)
            vals = df_sem["cell_id"].map(std)
            df_sem[f"rolling_std_{k}w"] = vals if fillna_value is None else vals.fillna(fillna_value)

        # --- media historica (mismo iso_week, semanas < s) ---
        mask_iw = calendario.reindex(hist_prev.index).values == iw
        misma_semana = hist_prev.loc[mask_iw]
        if len(misma_semana):
            media_hist = misma_semana.mean(axis=0)
        else:
            media_hist = pd.Series(np.nan, index=wide.columns)
        vals = df_sem["cell_id"].map(media_hist)
        df_sem["media_historica"] = vals if fillna_value is None else vals.fillna(fillna_value)

        # --- predecir ---
        X = df_sem[feature_cols].astype(float).values
        yp = modelo.predict(X)
        yp = np.clip(yp, 0, None)  # conteos no pueden ser negativos

        for idx, val in zip(df_sem.index, yp):
            pred_por_idx[idx] = val
        for idx, val in zip(df_sem.index, df_sem["count_robos"].values):
            true_por_idx[idx] = val

        y_true_sem = df_sem["count_robos"].values
        rmse_por_semana.append({"semana": s, "rmse": rmse(y_true_sem, yp)})

        # --- actualizar la tabla ancha con lo predicho (para semanas futuras) ---
        nueva_fila = pd.Series(np.nan, index=wide.columns)
        nueva_fila.update(pd.Series(yp, index=df_sem["cell_id"].values))
        wide.loc[s] = nueva_fila

    y_true = np.array([true_por_idx[i] for i in df_future.index])
    y_pred = np.array([pred_por_idx[i] for i in df_future.index])
    return y_true, y_pred, rmse_por_semana


# ---------------------------------------------------------------------------
# CV con burn-in
# ---------------------------------------------------------------------------

def burn_in_splits(df_dev: pd.DataFrame, min_burn_weeks: int = 52,
                    test_weeks: int = 4, n_splits: int | None = None):
    """
    Genera folds tipo TimeSeriesSplit pero garantizando que el train de CADA
    fold tenga al menos `min_burn_weeks` semanas reales antes del primer test,
    para que count_lag_52w y media_historica no queden sistematicamente en
    NaN/0 en los folds tempranos (que es lo que infla el RMSE de tu CV actual).

    Devuelve lista de (train_idx, test_idx) como posiciones enteras (para usar
    con .iloc) sobre `df_dev` ORDENADO por semana_global (esta funcion ya lo
    ordena y resetea el indice internamente).

    Nota: con pocas semanas totales de dev, esto deja pocos folds. Es preferible
    tener menos folds pero comparables, que 5 folds donde los primeros estan
    sistematicamente sesgados.
    """
    df_dev_sorted = df_dev.sort_values("semana_global").reset_index(drop=True)
    semanas = sorted(df_dev_sorted["semana_global"].unique())

    primer_test_semana = semanas[0] + min_burn_weeks
    semanas_test_candidatas = [s for s in semanas if s >= primer_test_semana]

    if not semanas_test_candidatas:
        raise ValueError(
            f"No hay suficientes semanas para min_burn_weeks={min_burn_weeks}. "
            f"Semanas disponibles: {len(semanas)}. Reduce min_burn_weeks o "
            f"consigue mas historia."
        )

    max_splits_posibles = len(semanas_test_candidatas) // test_weeks
    if max_splits_posibles == 0:
        raise ValueError(
            f"Solo quedan {len(semanas_test_candidatas)} semanas despues del "
            f"burn-in, no alcanza ni para 1 fold de test_weeks={test_weeks}."
        )
    if n_splits is None:
        n_splits = max_splits_posibles
    else:
        n_splits = min(n_splits, max_splits_posibles)

    folds = []
    for i in range(n_splits):
        ini = i * test_weeks
        test_sem = semanas_test_candidatas[ini: ini + test_weeks]
        if len(test_sem) < test_weeks:
            break
        train_sem = [s for s in semanas if s < test_sem[0]]
        train_idx = df_dev_sorted.index[df_dev_sorted["semana_global"].isin(train_sem)].to_numpy()
        test_idx = df_dev_sorted.index[df_dev_sorted["semana_global"].isin(test_sem)].to_numpy()
        folds.append((train_idx, test_idx))

    return df_dev_sorted, folds
