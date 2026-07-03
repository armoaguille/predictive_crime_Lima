# Modelo Predictivo de Robos en Lima

Prediccion de robos por **distrito x turno** en Lima Metropolitana, usando Machine Learning sobre datos del SIDPOL (2025-2026).

---

## Producto

**Input**: periodo de tiempo (semana o dia).
**Output**: matriz de frecuencias esperadas de robos por celda `(distrito, turno)`.

Dos versiones implementadas:

| Version | Granularidad | Celdas | Features |
|---|---|---|---|
| v1 (semanal) | `(distrito, turno, semana)` | 12,728 | 33 |

---

## Resultados

Evaluados en el holdout final con prediccion **multi-step recursiva** (simula el escenario real donde no se tienen los valores futuros).

### Semanal (12 semanas, 12,728 filas)

| Modelo | RMSE holdout | RMSE CV (5-fold) | RMSE CV std |
|---|---|---|---|
| Baseline media global | 12.15 | 13.84 | 3.94 |
| Baseline media historica | 8.61 | - | - |
| Poisson lineal | 6.90 | **9.19** | 2.55 |
| **XGBoost Poisson** | **6.48** | 16.57 | 4.47 |
| **LightGBM Poisson** | **6.51** | 16.42 | 4.25 |

**Ganador por métrica**:
- Holdout: **XGBoost** (menor RMSE absoluto)
- CV: **Poisson lineal** (más estable, mejor generalización)

---

## Pipeline

```
LimpiarDataGrande.ipynb     →  Data/lima_robos_limpio.csv  (700MB → 30MB, solo Lima)
EDA_Data_Lima.ipynb         →  Data/{train,test}_*.csv     (split + encoding)
Modelado/01_agregacion_*    →  Data/df_*.csv               (pivot a granularidad)
Modelado/02_features_*      →  Data/df_features_*.csv      (33 o 42 features)
Modelado/03_baseline_*      →  baselines + Poisson lineal
Modelado/04_xgb_lgbm_*      →  XGBoost + LightGBM
Modelado/05_validacion_*    →  TimeSeriesSplit 5-fold + holdout recursivo
Modelado/06_heatmap_*       →  heatmap 43x4 + ranking top-15
Modelado/07_optuna_tuning   →  tuning bayesiano (Optuna) con CV burn-in
Modelado/08_final_refit_*   →  reentrenamiento final + evaluacion en holdout
```

**Target**: `count_robos` en la celda `(distrito, turno, tiempo)`. Modelos usan perdida Poisson (natural para conteos).

**Evaluacion**: prediccion **multi-step recursiva** (cada prediccion se usa como lag de la siguiente). Esto simula produccion real; el esquema single-step con observacion era articialmente optimista.

**Features** (resumen, todas legítimas, sin fuga de target):
- Calendario ciclico (sin/cos de semana, mes, dia_semana, dia_mes)
- Indicadores (quincena, finde, feriado)
- Lags del target (1, 2, 3, 7, 14, 30, 52 segun modelo) — todos con `shift(N)`
- Rolling stats (mean y std con ventanas 3-30) — sobre los lags, no el target
- Media historica estacional (mismo distrito+turno+semana_del_anio, con `shift(1)`)
- OHE de turno + LabelEncoder de distrito
---

## Estructura del repositorio

```
TA_IA_Aplicada/
├── README.md
├── HALLAZGOS_TUNING_BAYESIANO.md         # analisis de resultados y hallazgos
├── CHANGELOG.md                          # bitacora de cambios
├── CAMBIOS_NOTEBOOKS.md                  # cambios sugeridos a notebooks existentes
│
├── LimpiarDataGrande.ipynb               # limpieza 700MB → 30MB
├── EDA_Data_Lima.ipynb                   # EDA + split + escalado
│
├── Data/                                 # artefactos intermedios
│
├── Modelado/                             
    ├── 01_agregacion_semanal.ipynb
    ├── 02_feature_engineering.ipynb
    ├── 03_baseline_poisson.ipynb
    ├── 04_xgb_lgbm.ipynb
    ├── 05_validacion_ts.ipynb
    ├── 06_heatmap_ranking.ipynb
    ├── 07_optuna_tuning.ipynb            # tuning bayesiano con burn-in CV
    ├── 08_final_refit_eval.ipynb          # reentrenamiento + evaluacion final
    ├── forecast_utils.py                  # funciones compartidas (forecast vectorizado, metrics)
    ├── figures/                           # graficos generados
    └── optuna/                           # resultados del tuning (params, models, plots)

```
---

## Stack

- Python 3.10+, pandas, numpy
- sklearn, xgboost, lightgbm
- matplotlib, seaborn, holidays
- optuna (tuning bayesiano, opcional)

---
