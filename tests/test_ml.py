import io
import os
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.services.gemelo import simular
from app.services.preprocess import (
    FEATURE_COLS,
    agregar_diaria,
    construir_features,
    generar_features_recientes,
    rellenar_series,
)
from app.services.recommendations import generar_recomendaciones
from app.services.train import leer_registro, siguiente_version
from app.utils.ml_utils import (
    calcular_mae,
    calcular_mape,
    calcular_r2,
    calcular_rmse,
    division_temporal,
)

PASES = []
FALLOS = []


def check(nombre, condicion, detalle=""):
    if condicion:
        PASES.append(nombre)
    else:
        FALLOS.append(f"{nombre} :: {detalle}")
        print(f"[FALLO] {nombre} :: {detalle}")


def df_sintetico():
    fechas = pd.date_range("2024-01-01", periods=120, freq="D").strftime("%Y-%m-%d")
    filas = []
    for pid in (101, 102):
        for f in fechas:
            filas.append(
                {"producto_id": pid, "producto_codigo": f"P-{pid}", "producto_nombre": f"Prod {pid}", "venta_fecha": f, "cantidad": 1}
            )
    return pd.DataFrame(filas)


def test_ml_utils():
    y = np.array([1.0, 2, 3, 4, 5])
    yp = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
    check("mae", round(calcular_mae(y, yp), 3) == 0.14, str(calcular_mae(y, yp)))
    check("rmse", round(calcular_rmse(y, yp), 4) == 0.1483, str(calcular_rmse(y, yp)))
    mape = calcular_mape(y, yp)
    check("mape entre 1 y 10", mape is not None and 1 < mape < 10, str(mape))
    check("r2", calcular_r2(y, yp) > 0.9, str(calcular_r2(y, yp)))
    yc = np.array([0, 0, 0])
    check("mape cero-guardado", calcular_mape(yc, yc) is None)


def test_preprocess():
    df = df_sintetico()
    agg = agregar_diaria(df)
    check("agregado 2 productos x 120 dias", len(agg) == 240, str(len(agg)))
    check("demanda acumulada", int(agg["demanda"].sum()) == 240)
    relleno = rellenar_series(agg)
    check("relleno conserva 240 filas", len(relleno) == 240)
    feat = construir_features(relleno)
    cols = set(FEATURE_COLS).issubset(set(feat.columns))
    check("features presentes", cols)
    check("features sin nulos", feat[FEATURE_COLS].notna().all().all())
    check("lag_1 en fila valida", int(feat[feat["producto_id"] == 101].iloc[0]["demanda_lag_1"]) == 1)
    fr = generar_features_recientes(relleno, 101, pd.Timestamp("2024-05-01"))
    check("features_recientes", fr is not None and fr.shape == (1, 10), str(None if fr is None else fr.shape))


def test_division_temporal():
    df = df_sintetico()
    t, v, e = division_temporal(df)
    check("split deja 3 partes", len(t) + len(v) + len(e) == len(df))
    check("split no vacio", len(t) > 0 and len(v) > 0 and len(e) > 0)


def test_registro_versiones(tmp):
    os.makedirs(tmp, exist_ok=True)
    from app.services import train as train_mod

    real_dir = train_mod.MODELOS_DIR
    real_reg = train_mod.REGISTRY_PATH
    train_mod.MODELOS_DIR = tmp
    train_mod.REGISTRY_PATH = os.path.join(tmp, "registry.json")
    try:
        reg = leer_registro()
        v1 = siguiente_version("RL", reg, "ventas")
        check("version RL-ventas-001 inicial", v1 == "RL-ventas-001", v1)
        reg["RL-ventas-001"] = {"algoritmo": "RL", "version": "RL-ventas-001", "modelo_tipo": "ventas"}
        v2 = siguiente_version("RL", reg, "ventas")
        check("version incrementa RL-ventas-002", v2 == "RL-ventas-002", v2)
        v3 = siguiente_version("RF", reg, "ventas")
        check("version RF-ventas-001 inicial", v3 == "RF-ventas-001", v3)
    finally:
        train_mod.MODELOS_DIR = real_dir
        train_mod.REGISTRY_PATH = real_reg


def test_reglas_recomendaciones():
    base = {
        "producto_id": 101,
        "producto_codigo": "P-101",
        "producto_nombre": "Prod A",
        "stock_actual": 10,
        "demanda_total_proyectada": 50.0,
    }
    r = generar_recomendaciones([{**base, "demanda_total_proyectada": 50.0, "stock_actual": 10}], {"stock_seguridad": {}}) 
    check("riesgo quiebre detectado", r[0]["riesgo"] == "RIESGO_QUIEBRE_STOCK", r[0]["riesgo"])
    check("sugerencia de reposicion", r[0]["cantidad_sugerida"] == 40)

    r = generar_recomendaciones([{**base, "stock_actual": 0}], None)
    check("agotado detectado", r[0]["riesgo"] == "AGOTADO_QUIEBRE", r[0]["riesgo"])
    check("sugerencia agotado", r[0]["cantidad_sugerida"] == 50)

    r = generar_recomendaciones([{**base, "stock_actual": 300, "demanda_total_proyectada": 5.0}], {"rotacion": {"101": "BAJA_ROTACION"}})
    check("exceso detectado con baja rotacion", r[0]["riesgo"] == "EXCESO_INVENTARIO", r[0]["riesgo"])

    r = generar_recomendaciones([{**base, "stock_actual": 60, "demanda_total_proyectada": 30.0}], None)
    check("stock adecuado", r[0]["riesgo"] == "STOCK_ADECUADO", r[0]["riesgo"])


def test_gemelo():
    pred = {"stock_actual": 5, "demanda_total_proyectada": 120.0, "horizonte_dias": 30}
    s = simular(pred, compra_simulada=200)
    check("gemelo suficiente", s["riesgo"] == "STOCK_SUFICIENTE", s["riesgo"])
    check("gemelo respeta stock real", s["stock_real_sin_cambios"] == 5)
    s2 = simular(pred, compra_simulada=50)
    check("gemelo quiebre", s2["riesgo"] == "RIESGO_QUIEBRE_STOCK", s2["riesgo"])
    check("gemelo es simulacion", s2["es_simulacion"] is True)


def main():
    real = os.environ.get("ML_MODELS_DIR")
    if not real:
        os.environ["ML_MODELS_DIR"] = os.path.join(os.path.dirname(__file__), "..", "models")
    test_ml_utils()
    test_preprocess()
    test_division_temporal()
    test_registro_versiones(os.path.join(os.path.dirname(__file__), "_tmp"))
    test_reglas_recomendaciones()
    test_gemelo()
    clean = os.path.join(os.path.dirname(__file__), "_tmp")
    if os.path.isdir(clean):
        import shutil

        shutil.rmtree(clean, ignore_errors=True)
    print(f"\nRESULTADO: {len(PASES)} pasan, {len(FALLOS)} fallan")
    return 1 if FALLOS else 0


if __name__ == "__main__":
    sys.exit(main())