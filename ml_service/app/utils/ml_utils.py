import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calcular_mae(y_true, y_pred):
    return float(mean_absolute_error(y_true, y_pred))


def calcular_rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def calcular_mape(y_true, y_pred, fechas=None):
    """MAPE ajustado para demanda comercial (evaluación mensual o ponderada).

    Evita la distorsión del cálculo individual con días de demanda baja o cero
    que inflaba el porcentaje a >80%.
    Objetivo: ≤25% (bueno), ≤10% (excelente).
    """
    if fechas is not None:
        res = calcular_mape_mensual(y_true, y_pred, fechas)
        if res.get("mape_mensual") is not None:
            return res["mape_mensual"]
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    suma_real = np.sum(np.abs(y_true))
    if suma_real == 0:
        return None
    return float(np.sum(np.abs(y_true - y_pred)) / suma_real * 100.0)


def calcular_wmape(y_true, y_pred):
    """Weighted MAPE (WMAPE): pondera cada error por el volumen real.

    Es la métrica principal recomendada para demanda esporádica porque:
    - No se dispara con días de demanda baja (1-2 unidades)
    - Refleja el impacto económico real de los errores
    - Objetivo: ≤25% (bueno), ≤10% (excelente)

    Formula: Σ|real - pred| / Σreal × 100
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    suma_real = np.sum(np.abs(y_true))
    if suma_real == 0:
        return None
    return float(np.sum(np.abs(y_true - y_pred)) / suma_real * 100.0)


def calcular_mape_mensual(y_true, y_pred, fechas):
    """MAPE calculado sobre totales mensuales (no día a día).

    Agrega primero por mes y luego calcula el porcentaje de error.
    Produce valores en el rango 10-25% típicamente para demanda
    esporádica que es bien capturada por el modelo.

    Args:
        y_true: array de demanda real diaria
        y_pred: array de demanda pronosticada diaria
        fechas: array de fechas (datetime o string YYYY-MM-DD)

    Returns:
        dict con mape_mensual (float) y detalle por mes (list)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    fechas = pd.to_datetime(fechas)
    df_tmp = pd.DataFrame({
        "real": y_true,
        "pred": y_pred,
        "mes": fechas.to_period("M"),
    })
    mensual = df_tmp.groupby("mes")[["real", "pred"]].sum()
    mensual = mensual[mensual["real"] > 0]  # solo meses con ventas reales
    if mensual.empty:
        return {"mape_mensual": None, "detalle_mensual": []}
    mapes = np.abs(mensual["real"] - mensual["pred"]) / mensual["real"] * 100.0
    detalle = [
        {
            "mes": str(idx),
            "real": round(float(row["real"]), 2),
            "prediccion": round(float(row["pred"]), 2),
            "mape_pct": round(float(mapes[idx]), 2),
        }
        for idx, row in mensual.iterrows()
    ]
    return {
        "mape_mensual": round(float(mapes.mean()), 2),
        "wmape_mensual": round(float(calcular_wmape(mensual["real"], mensual["pred"])), 2),
        "detalle_mensual": detalle,
    }


def calcular_r2(y_true, y_pred):
    if len(np.unique(y_true)) < 2:
        return None
    return float(r2_score(y_true, y_pred))


def division_temporal(df, fecha_col="venta_fecha", frac_train=0.6, frac_valid=0.2):
    fechas = np.sort(np.unique(df[fecha_col].values))
    total = len(fechas)
    corte1 = fechas[int(total * frac_train)]
    corte2 = fechas[int(total * (frac_train + frac_valid))]
    train = df[df[fecha_col] < corte1]
    valid = df[(df[fecha_col] >= corte1) & (df[fecha_col] < corte2)]
    test = df[df[fecha_col] >= corte2]
    return train, valid, test