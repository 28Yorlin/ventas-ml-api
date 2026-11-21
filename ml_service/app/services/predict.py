import joblib
import numpy as np
import pandas as pd

from app.services.preprocess import (
    cargar_dataset,
    generar_features_recientes_ventas,
    generar_features_recientes_inventario,
    preparar_dataset_ventas,
    preparar_dataset_inventario,
)
from app.services.db_source import firma_bd
from app.services.train import modelo_activo, get_modelos_by_tipo

_DATASET_CACHE_VENTAS = None
_DATASET_CACHE_INVENTARIO = None
_CATALOGO_CACHE = None
_CACHE_FIRMA = None


def _invalidar_cache_si_cambia_bd():
    """Descarta los datasets en cache cuando aparecen movimientos nuevos en la BD.

    Asi las predicciones usan ventas recientes sin reiniciar el servicio. Si la
    BD no esta disponible, se conserva lo que haya en cache (no rompe nada)."""
    global _CACHE_FIRMA, _DATASET_CACHE_VENTAS, _DATASET_CACHE_INVENTARIO, _CATALOGO_CACHE
    try:
        firma = firma_bd()
    except Exception:
        return
    if _CACHE_FIRMA is not None and firma != _CACHE_FIRMA:
        _DATASET_CACHE_VENTAS = None
        _DATASET_CACHE_INVENTARIO = None
        _CATALOGO_CACHE = None
    _CACHE_FIRMA = firma


def _catalogo_productos():
    """Catalogo unico por producto desde el CSV crudo: codigo, nombre, categoria,
    marca, modelo, minimos, reposicion, rotacion y stock actual (ultimo kardex)."""
    global _CATALOGO_CACHE
    _invalidar_cache_si_cambia_bd()
    if _CATALOGO_CACHE is None:
        df = cargar_dataset()
        df = df.sort_values(["producto_id", "venta_fecha"], kind="mergesort", ignore_index=True)
        cat = (
            df.groupby("producto_id", as_index=False)
            .agg(
                producto_codigo=("producto_codigo", "last"),
                producto_nombre=("producto_nombre", "last"),
                categoria_nombre=("categoria_nombre", "last"),
                producto_marca=("producto_marca", "last"),
                producto_modelo=("producto_modelo", "last"),
                producto_stock_minimo=("producto_stock_minimo", "last"),
                producto_stock_seguridad=("producto_stock_seguridad", "last"),
                producto_tiempo_reposicion_dias=("producto_tiempo_reposicion_dias", "last"),
                producto_rotacion=("producto_rotacion", "last"),
                stock_actual=("stock_nuevo", "last"),
            )
        )
        _CATALOGO_CACHE = cat
    return _CATALOGO_CACHE


def _meta_productos():
    """Mapa producto_id -> metadata de catalogo."""
    entries = {}
    for r in _catalogo_productos().to_dict("records"):
        entries[int(r["producto_id"])] = r
    return entries


def _stock_real(stock_map, pid):
    """Stock actual: prioriza el mapa provisto (BD/app), si no usa el ultimo kardex del CSV."""
    pid = int(pid)
    if stock_map:
        val = stock_map.get(str(pid), stock_map.get(pid))
        if val is not None:
            return int(val)
    cat = _catalogo_productos()
    fila = cat[cat["producto_id"] == pid]
    return int(fila["stock_actual"].iloc[0]) if not fila.empty else 0


def cargar_historial_ventas():
    global _DATASET_CACHE_VENTAS
    _invalidar_cache_si_cambia_bd()
    if _DATASET_CACHE_VENTAS is None:
        _DATASET_CACHE_VENTAS = preparar_dataset_ventas()
    return _DATASET_CACHE_VENTAS


def cargar_historial_inventario():
    global _DATASET_CACHE_INVENTARIO
    _invalidar_cache_si_cambia_bd()
    if _DATASET_CACHE_INVENTARIO is None:
        _DATASET_CACHE_INVENTARIO = preparar_dataset_inventario()
    return _DATASET_CACHE_INVENTARIO


def predecir_demanda(horizonte=30, stock_actual=None):
    """Predict sales/demand using the active ventas model"""
    entry = modelo_activo(modelo_tipo="ventas")
    if entry is None:
        return {"error": "No existe un modelo activo de ventas. Ejecute /train."}
    modelo = joblib.load(entry["ruta"])
    hist = cargar_historial_ventas()
    stock_actual = stock_actual or {}
    metas = _meta_productos()

    productos = [
        {
            "producto_id": int(pid),
            "producto_codigo": str(part["producto_codigo"].iloc[0]),
            "producto_nombre": str(part["producto_nombre"].iloc[0]),
        }
        for pid, part in hist.groupby("producto_id")
    ]

    ultima_fecha = hist["venta_fecha"].max()
    proxima_fecha = ultima_fecha
    resultado = []

    for prod in productos:
        pid = prod["producto_id"]
        features = generar_features_recientes_ventas(hist, pid, proxima_fecha)
        if features is None:
            continue
        demanda_diaria = float(max(modelo.predict(features)[0], 0))
        demanda_total = round(demanda_diaria * int(horizonte), 4)
        stock = _stock_real(stock_actual, pid)
        m = metas.get(pid) or {}
        resultado.append(
            {
                "producto_id": pid,
                "producto_codigo": prod["producto_codigo"],
                "producto_nombre": prod["producto_nombre"],
                "demanda_diaria_estimada": round(demanda_diaria, 4),
                "demanda_total_proyectada": demanda_total,
                "horizonte_dias": int(horizonte),
                "periodo_inicio": str((proxima_fecha + pd.Timedelta(days=1)).date()),
                "periodo_fin": str((proxima_fecha + pd.Timedelta(days=int(horizonte))).date()),
                "stock_actual": stock,
                "categoria_nombre": m.get("categoria_nombre") or "",
                "producto_marca": m.get("producto_marca") or "",
                "producto_modelo": m.get("producto_modelo") or "",
                "producto_stock_minimo": int(m.get("producto_stock_minimo") or 0),
                "producto_stock_seguridad": int(m.get("producto_stock_seguridad") or 0),
                "producto_tiempo_reposicion_dias": int(m.get("producto_tiempo_reposicion_dias") or 0),
                "producto_rotacion": m.get("producto_rotacion") or "SIN_MOVIMIENTO",
            }
        )

    resultado.sort(key=lambda r: r["demanda_total_proyectada"], reverse=True)
    return {
        "modelo": {"version": entry["version"], "algoritmo": entry["algoritmo"], "tipo": "ventas"},
        "registros": len(resultado),
        "predicciones": resultado,
    }


def _vector_ventas_serie(serie, fecha_pred, primera_fecha):
    """Construye el vector de features de ventas tal como se usa al predecir
    (mismas formulas que generar_features_recientes_ventas), para pronostico
    recursivo dia a dia. Requiere al menos 30 valores historicos."""
    if len(serie) < 30:
        return None
    from app.services.preprocess import FESTIVOS_SET
    festivo = 1.0 if pd.Timestamp(fecha_pred) in FESTIVOS_SET else 0.0
    fin_semana = 1.0 if fecha_pred.dayofweek >= 5 else 0.0
    # promo y descuento_pct: 0 por defecto (no hay datos futuros)
    promo = 0.0
    descuento_pct = 0.0
    promocion_id = 0.0
    promocion_descuento_pct = 0.0
    # One-hot encode promocion_tipo (todos 0 para futuro)
    promo_tipo_dict = {}
    for tipo in ["black_friday", "navidad", "cyber_monday", "aniversario", 
                 "dia_madre", "dia_padre", "san_valentin", "liquidacion",
                 "regreso_clases", "hot_sale", "otro", "ninguna"]:
        promo_tipo_dict[f"promo_tipo_{tipo}"] = 0.0
    promo_tipo_dict["promo_tipo_ninguna"] = 1.0
    # One-hot encode promocion_tipo_descuento
    promo_desc_dict = {}
    for td in ["porcentaje", "monto_fijo", "2x1", "3x2", "envio_gratis", "ninguna"]:
        promo_desc_dict[f"promo_desc_tipo_{td}"] = 0.0
    promo_desc_dict["promo_desc_tipo_ninguna"] = 1.0
    
    vector = [
        float(fecha_pred.dayofweek),
        float(fecha_pred.month),
        float((fecha_pred - primera_fecha).days),
        festivo,
        fin_semana,
        promo,
        descuento_pct,
        float(serie[-1]),
        float(serie[-8]),
        float(serie[-15]),
        float(serie[-29]),
        float(np.mean(serie[-8:])),
        float(np.mean(serie[-14:])),
        float(np.max(serie[-8:])),
    ]
    return vector


def predecir_demanda_serie(horizonte=30, stock_actual=None):
    """Pronostico recursivo dia a dia con el modelo activo de ventas.

    Genera una serie temporal diaria real (prediccion del modelo por cada dia
    del horizonte, usando los valores proyectados como lags del dia siguiente)
    ademas de la prediccion por producto. La demanda diaria estimada coincide
    con la primera prediccion (misma features que /predict).
    """
    entry = modelo_activo(modelo_tipo="ventas")
    if entry is None:
        return {"error": "No existe un modelo activo de ventas. Ejecute /train."}
    modelo = joblib.load(entry["ruta"])
    hist = cargar_historial_ventas()
    stock_actual = stock_actual or {}
    metas = _meta_productos()

    ultima_fecha = hist["venta_fecha"].max()
    horizonte = int(horizonte)
    if horizonte < 1:
        horizonte = 1

    preds_producto = []
    totales_dia = [0.0] * horizonte

    for pid, part in hist.groupby("producto_id"):
        part = part.sort_values("venta_fecha")
        if len(part) < 30:
            continue
        primera_fecha = part["venta_fecha"].min()
        serie = part["demanda"].astype(float).tolist()

        preds = []
        fecha = ultima_fecha
        for _ in range(horizonte):
            fecha = fecha + pd.Timedelta(days=1)
            vector = _vector_ventas_serie(serie, fecha, primera_fecha)
            if vector is None:
                break
            valor = float(max(modelo.predict(np.array([vector]))[0], 0))
            preds.append((fecha, valor))
            serie.append(valor)

        if not preds:
            continue

        demanda_diaria = float(preds[0][1])
        demanda_total = float(sum(v for _, v in preds))
        medio = max(1, len(preds) // 2)
        s1 = float(sum(v for _, v in preds[:medio]))
        s2 = float(sum(v for _, v in preds[medio:]))
        tendencia_pct = round(((s2 - s1) / max(s1, 1e-9)) * 100.0, 2) if s1 > 0 else 0.0

        for i, (_, valor) in enumerate(preds):
            totales_dia[i] += valor

        stock = _stock_real(stock_actual, int(pid))
        m = metas.get(int(pid)) or {}
        preds_producto.append(
            {
                "producto_id": int(pid),
                "producto_codigo": str(part["producto_codigo"].iloc[0]),
                "producto_nombre": str(part["producto_nombre"].iloc[0]),
                "demanda_diaria_estimada": round(demanda_diaria, 4),
                "demanda_total_proyectada": round(demanda_total, 4),
                "tendencia_pct": tendencia_pct,
                "stock_actual": stock,
                "categoria_nombre": m.get("categoria_nombre") or "",
                "producto_marca": m.get("producto_marca") or "",
                "producto_modelo": m.get("producto_modelo") or "",
                "producto_stock_minimo": int(m.get("producto_stock_minimo") or 0),
                "producto_stock_seguridad": int(m.get("producto_stock_seguridad") or 0),
                "producto_tiempo_reposicion_dias": int(m.get("producto_tiempo_reposicion_dias") or 0),
                "producto_rotacion": m.get("producto_rotacion") or "SIN_MOVIMIENTO",
            }
        )

    serie_diaria = [
        {
            "fecha": str((ultima_fecha + pd.Timedelta(days=i + 1)).date()),
            "demanda_total": round(totales_dia[i], 2),
        }
        for i in range(horizonte)
    ]

    dia_max = max(serie_diaria, key=lambda s: s["demanda_total"]) if serie_diaria else None

    preds_producto.sort(key=lambda r: r["demanda_total_proyectada"], reverse=True)
    return {
        "modelo": {"version": entry["version"], "algoritmo": entry["algoritmo"], "tipo": "ventas"},
        "registros": len(preds_producto),
        "horizonte_dias": horizonte,
        "fecha_inicio": str((ultima_fecha + pd.Timedelta(days=1)).date()),
        "fecha_fin": str((ultima_fecha + pd.Timedelta(days=horizonte)).date()),
        "serie_diaria": serie_diaria,
        "dia_mayor_demanda": dia_max["fecha"] if dia_max else None,
        "demanda_total_proyectada": round(sum(s["demanda_total"] for s in serie_diaria), 4),
        "promedio_diario": round(sum(s["demanda_total"] for s in serie_diaria) / max(1, len(serie_diaria)), 2),
        "predicciones": preds_producto,
    }


def predecir_inventario(horizonte=30):
    """Predict inventory/stock using the active inventario model"""
    entry = modelo_activo(modelo_tipo="inventario")
    if entry is None:
        return {"error": "No existe un modelo activo de inventario. Ejecute /train."}
    modelo = joblib.load(entry["ruta"])
    hist = cargar_historial_inventario()
    metas = _meta_productos()

    productos = [
        {
            "producto_id": int(pid),
            "producto_codigo": str(part["producto_codigo"].iloc[0]),
            "producto_nombre": str(part["producto_nombre"].iloc[0]),
            "stock_minimo": int(part["producto_stock_minimo"].iloc[0]),
            "stock_seguridad": int(part["producto_stock_seguridad"].iloc[0]),
            "tiempo_reposicion": int(part["producto_tiempo_reposicion_dias"].iloc[0]),
            "rotacion": str(part["producto_rotacion"].iloc[0]),
        }
        for pid, part in hist.groupby("producto_id")
    ]

    ultima_fecha = hist["venta_fecha"].max()
    proxima_fecha = ultima_fecha
    resultado = []

    for prod in productos:
        pid = prod["producto_id"]
        features = generar_features_recientes_inventario(hist, pid, proxima_fecha)
        if features is None:
            continue
        stock_proyectado = float(max(modelo.predict(features)[0], 0))
        # Project stock over horizon considering demand
        # Use average daily demand from sales model as approximation
        stock_diario = stock_proyectado
        stock_total = round(stock_diario, 4)
        m = metas.get(pid) or {}
        resultado.append(
            {
                "producto_id": pid,
                "producto_codigo": prod["producto_codigo"],
                "producto_nombre": prod["producto_nombre"],
                "categoria_nombre": m.get("categoria_nombre") or "",
                "producto_marca": m.get("producto_marca") or "",
                "producto_modelo": m.get("producto_modelo") or "",
                "stock_actual": _stock_real({}, pid),
                "stock_proyectado": stock_proyectado,
                "stock_minimo": prod["stock_minimo"],
                "stock_seguridad": prod["stock_seguridad"],
                "tiempo_reposicion_dias": prod["tiempo_reposicion"],
                "rotacion": prod["rotacion"],
                "horizonte_dias": int(horizonte),
                "periodo_inicio": str((proxima_fecha + pd.Timedelta(days=1)).date()),
                "periodo_fin": str((proxima_fecha + pd.Timedelta(days=int(horizonte))).date()),
                "riesgo_quiebre": stock_proyectado < prod["stock_seguridad"],
                "riesgo_exceso": stock_proyectado > prod["stock_minimo"] * 3,
            }
        )

    resultado.sort(key=lambda r: r["stock_proyectado"])
    return {
        "modelo": {"version": entry["version"], "algoritmo": entry["algoritmo"], "tipo": "inventario"},
        "registros": len(resultado),
        "predicciones": resultado,
    }


def predecir_ambos(horizonte=30, stock_actual=None):
    """Predict both sales and inventory"""
    demanda = predecir_demanda(horizonte, stock_actual)
    inventario = predecir_inventario(horizonte)
    return {
        "demanda": demanda,
        "inventario": inventario,
    }