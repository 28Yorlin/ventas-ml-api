import datetime
import json
import os

import joblib
import numpy as np
from sklearn.linear_model import LinearRegression

from app.services.preprocess import (
    SALES_FEATURE_COLS,
    INVENTORY_FEATURE_COLS,
    preparar_dataset_ventas,
    preparar_dataset_inventario,
)
from app.utils.ml_utils import (
    calcular_mae, calcular_mape, calcular_wmape,
    calcular_r2, calcular_rmse, division_temporal,
)

MODELOS_DIR = os.environ.get(
    "ML_MODELS_DIR", os.path.join(os.path.dirname(__file__), "..", "..", "models")
)
REGISTRY_PATH = os.path.join(MODELOS_DIR, "registry.json")


def leer_registro():
    if os.path.exists(REGISTRY_PATH):
        with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def escribir_registro(registro):
    os.makedirs(MODELOS_DIR, exist_ok=True)
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(registro, f, ensure_ascii=False, indent=2)


def siguiente_version(algoritmo, registro, modelo_tipo=""):
    versiones = [
        int(k["version"].split("-")[-1])
        for k in registro.values()
        if k.get("algoritmo") == algoritmo 
        and k.get("modelo_tipo") == modelo_tipo
        and k.get("version", "").startswith(algoritmo)
    ]
    numero = max(versiones, default=0) + 1
    return f"{algoritmo}-{modelo_tipo}-{numero:03d}" if modelo_tipo else f"{algoritmo}-{numero:03d}"


def evaluar(modelo, X, y, fechas=None):
    y_pred = np.maximum(modelo.predict(X), 0)
    return {
        "mae": calcular_mae(y, y_pred),
        "rmse": calcular_rmse(y, y_pred),
        "mape": calcular_mape(y, y_pred, fechas),
        "r2": calcular_r2(y, y_pred),
    }


def entrenar_modelo(datos, feature_cols, target_col, modelo_tipo, registro):
    """Train Linear Regression model for a specific target (ventas or inventario)"""
    train, valid, test = division_temporal(datos)
    X_train, y_train = train[feature_cols].values, train[target_col].values
    X_valid, y_valid = valid[feature_cols].values, valid[target_col].values
    X_test, y_test = test[feature_cols].values, test[target_col].values

    f_valid = valid["venta_fecha"].values if "venta_fecha" in valid.columns else None
    f_test = test["venta_fecha"].values if "venta_fecha" in test.columns else None

    modelo = LinearRegression()
    algoritmo = "RL"

    version = siguiente_version(algoritmo, registro, modelo_tipo)
    modelo.fit(X_train, y_train)
    ruta = os.path.join(MODELOS_DIR, f"{version}.joblib")
    os.makedirs(MODELOS_DIR, exist_ok=True)
    joblib.dump(modelo, ruta)

    eval_valid = evaluar(modelo, X_valid, y_valid, f_valid)
    eval_test = evaluar(modelo, X_test, y_test, f_test)

    resultado = {
        "algoritmo": algoritmo,
        "version": version,
        "estado": "CANDIDATO",
        "modelo_tipo": modelo_tipo,
        "dataset": os.path.basename(os.environ.get("ML_RAW_CSV", "ventas.csv")),
        "registros": int(len(datos)),
        "variables": list(feature_cols),
        "parametros": str(modelo.get_params()),
        "mae": eval_test["mae"],
        "rmse": eval_test["rmse"],
        "mape": eval_test["mape"],
        "wmape": eval_test["mape"],
        "r2": eval_test["r2"],
        "ruta": ruta,
        "fecha": datetime.datetime.now().isoformat(timespec="seconds"),
        "fecha_datos": {
            "inicio": str(datos["venta_fecha"].min().date()),
            "fin": str(datos["venta_fecha"].max().date()),
        },
        "validacion": eval_valid,
        "prueba": eval_test,
    }
    registro[version] = resultado

    return {algoritmo: resultado}


def entrenar(conjuntos=None):
    """Train both sales and inventory models (Linear Regression only)"""
    registro = leer_registro()
    all_resultados = {}
    
    # Train sales model (demanda target)
    print("Training sales/demand model (Linear Regression)...")
    datos_ventas = conjuntos or preparar_dataset_ventas()
    datos_ventas = datos_ventas[datos_ventas["demanda"] >= 0].copy()
    resultados_ventas = entrenar_modelo(datos_ventas, SALES_FEATURE_COLS, "demanda", "ventas", registro)
    all_resultados["ventas"] = resultados_ventas

    # Train inventory model (stock_nuevo target)
    print("Training inventory/stock model (Linear Regression)...")
    datos_inventario = preparar_dataset_inventario()
    datos_inventario = datos_inventario[datos_inventario["stock_nuevo"] >= 0].copy()
    resultados_inventario = entrenar_modelo(datos_inventario, INVENTORY_FEATURE_COLS, "stock_nuevo", "inventario", registro)
    all_resultados["inventario"] = resultados_inventario

    # Guardian: solo activa el nuevo si mejora MAE
    def clave_orden(v):
        return v["prueba"]["mae"] if v["prueba"]["mae"] is not None else 1e9

    try:
        mejora_minima = float(os.environ.get("ML_MEJORA_MINIMA", "0.0") or 0.0)
    except (TypeError, ValueError):
        mejora_minima = 0.0

    activos_finales = {}
    cambios = {}
    for modelo_tipo in ["ventas", "inventario"]:
        candidatos = all_resultados[modelo_tipo]
        mejor_nuevo = min(candidatos.values(), key=clave_orden)

        previos = [
            v for v in registro.values()
            if v.get("estado") == "ACTIVO"
            and v.get("modelo_tipo") == modelo_tipo
            and v.get("version") not in candidatos
        ]
        previo = min(previos, key=clave_orden) if previos else None

        mae_nuevo = mejor_nuevo.get("prueba", {}).get("mae")
        mae_previo = previo.get("prueba", {}).get("mae") if previo else None

        cambia = True
        if previo is not None and mae_previo is not None and mae_nuevo is not None:
            cambia = mae_nuevo < mae_previo * (1.0 - mejora_minima)

        if cambia:
            for v in registro.values():
                if v.get("estado") == "ACTIVO" and v.get("modelo_tipo") == modelo_tipo:
                    v["estado"] = "INACTIVO"
            mejor_nuevo["estado"] = "ACTIVO"
            registro[mejor_nuevo["version"]]["estado"] = "ACTIVO"
            activos_finales[modelo_tipo] = mejor_nuevo["version"]
        else:
            activos_finales[modelo_tipo] = previo["version"]
        cambios[modelo_tipo] = cambia

    escribir_registro(registro)
    return {
        "entrenados": sum(len(v) for v in all_resultados.values()),
        "resultados": all_resultados,
        "mejor_ventas": activos_finales["ventas"],
        "mejor_inventario": activos_finales["inventario"],
        "cambio_ventas": cambios["ventas"],
        "cambio_inventario": cambios["inventario"],
    }


def modelo_activo(modelo_tipo=None):
    registro = leer_registro()
    if modelo_tipo:
        activos = [v for v in registro.values() if v.get("estado") == "ACTIVO" and v.get("modelo_tipo") == modelo_tipo]
    else:
        activos = [v for v in registro.values() if v.get("estado") == "ACTIVO"]
    if not activos:
        return None
    activos.sort(key=lambda v: v.get("mae") or 1e9)
    return activos[0]


def get_modelos_by_tipo(modelo_tipo):
    """Get all models of a specific type"""
    registro = leer_registro()
    return [v for v in registro.values() if v.get("modelo_tipo") == modelo_tipo]