import os

import numpy as np
import pandas as pd

RAW_CSV = os.environ.get(
    "ML_RAW_CSV",
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "ventas.csv"),
)

# Conjunto de fechas festivos para el modelo (mm-dd)
FESTIVOS_SET = {
    pd.Timestamp("2025-01-01"),  # Año Nuevo
    pd.Timestamp("2025-02-14"),  # San Valentín
    pd.Timestamp("2025-05-10"),  # Día de la Madre
    pd.Timestamp("2025-07-20"),  # Día de la Independencia (ejemplo)
    pd.Timestamp("2025-08-15"),  # Otra fecha festiva
    pd.Timestamp("2025-12-25"),  # Navidad
    pd.Timestamp("2025-11-30"),  # Otra fecha festiva
    pd.Timestamp("2025-12-31"),  # Fin de año
}

# Feature columns for sales/demand model (14 features, coinciden con el modelo RL-ventas-005)
SALES_FEATURE_COLS = [
    "dia_semana",
    "mes",
    "indice_fecha",
    "festivo",
    "fin_semana",
    "promo",
    "descuento_pct",
    "demanda_lag_1",
    "demanda_lag_7",
    "demanda_lag_14",
    "demanda_lag_28",
    "demanda_media_7",
    "demanda_media_14",
    "demanda_max_7",
]

# Feature columns for inventory/stock model
INVENTORY_FEATURE_COLS = [
    "dia_semana",
    "mes",
    "indice_fecha",
    "stock_lag_1",
    "stock_lag_7",
    "stock_lag_14",
    "stock_lag_28",
    "stock_media_7",
    "stock_media_14",
    "stock_minimo",
    "stock_seguridad",
    "tiempo_reposicion",
    "rotacion_alta",
    "rotacion_media",
    "rotacion_baja",
    "entrada_lag_1",
    "salida_lag_1",
]


def cargar_dataset(ruta=None):
    """Carga el dataset crudo. Por defecto lo lee EN VIVO de la base de datos
    (ML_DATA_SOURCE=db), de modo que el modelo aprenda de las ventas nuevas.
    Si la BD no esta disponible, o si se pasa una ruta explicita, usa el CSV
    como respaldo (data/raw/ventas.csv)."""
    if ruta is None and os.environ.get("ML_DATA_SOURCE", "db").strip().lower() == "db":
        try:
            from app.services.db_source import leer_dataset

            return leer_dataset()
        except Exception as exc:  # respaldo automatico al CSV si la BD falla
            print("[preprocess] No se pudo leer de la BD (%s). Se usa el CSV." % exc)
    ruta = ruta or RAW_CSV
    df = pd.read_csv(ruta, low_memory=False)
    df["venta_fecha"] = pd.to_datetime(df["venta_fecha"], format="%Y-%m-%d")
    return df


def agregar_diaria_ventas(df):
    """Aggregate daily demand per product from sales data"""
    df = df.copy()
    df["venta_fecha"] = pd.to_datetime(df["venta_fecha"], format="%Y-%m-%d")
    grupo = (
        df.groupby(["producto_id", "producto_codigo", "producto_nombre", "venta_fecha"], as_index=False)[
            "cantidad"
        ]
        .sum()
        .rename(columns={"cantidad": "demanda"})
    )
    grupo = grupo.sort_values(["producto_id", "venta_fecha"]).reset_index(drop=True)
    return grupo


def agregar_diaria_inventario(df):
    """Aggregate daily inventory state per product"""
    df = df.copy()
    df["venta_fecha"] = pd.to_datetime(df["venta_fecha"], format="%Y-%m-%d")
    # For inventory, we take the last stock_nuevo of each day per product
    grupo = (
        df.sort_values("venta_fecha")
        .groupby(["producto_id", "producto_codigo", "producto_nombre", "venta_fecha"], as_index=False)
        .last()[["producto_id", "producto_codigo", "producto_nombre", "venta_fecha",
                 "stock_nuevo", "producto_stock_minimo", "producto_stock_seguridad",
                 "producto_tiempo_reposicion_dias", "producto_rotacion",
                 "entrada", "salida"]]
    )
    grupo = grupo.sort_values(["producto_id", "venta_fecha"]).reset_index(drop=True)
    return grupo


def rellenar_series_ventas(grupo):
    """Fill missing dates with zero demand for sales model"""
    series = []
    for producto_id, part in grupo.groupby("producto_id"):
        nombre = part["producto_nombre"].iloc[0]
        codigo = part["producto_codigo"].iloc[0]
        indice = pd.date_range(part["venta_fecha"].min(), part["venta_fecha"].max(), freq="D")
        relleno = part.set_index("venta_fecha").reindex(indice, fill_value=0)
        relleno = relleno.reset_index().rename(columns={"index": "venta_fecha"})
        relleno["producto_id"] = producto_id
        relleno["producto_nombre"] = nombre
        relleno["producto_codigo"] = codigo
        series.append(relleno)
    return pd.concat(series, ignore_index=True)


def rellenar_series_inventario(grupo):
    """Fill missing dates by forward-filling stock for inventory model"""
    series = []
    for producto_id, part in grupo.groupby("producto_id"):
        nombre = part["producto_nombre"].iloc[0]
        codigo = part["producto_codigo"].iloc[0]
        indice = pd.date_range(part["venta_fecha"].min(), part["venta_fecha"].max(), freq="D")
        relleno = part.set_index("venta_fecha").reindex(indice)
        # Forward fill stock values, fill other columns appropriately
        relleno["producto_id"] = producto_id
        relleno["producto_nombre"] = nombre
        relleno["producto_codigo"] = codigo
        # Forward fill stock_nuevo and other inventory params
        relleno["stock_nuevo"] = relleno["stock_nuevo"].ffill()
        relleno["producto_stock_minimo"] = relleno["producto_stock_minimo"].ffill()
        relleno["producto_stock_seguridad"] = relleno["producto_stock_seguridad"].ffill()
        relleno["producto_tiempo_reposicion_dias"] = relleno["producto_tiempo_reposicion_dias"].ffill()
        relleno["producto_rotacion"] = relleno["producto_rotacion"].ffill()
        # Fill entrada/salida with 0 for missing days
        relleno["entrada"] = relleno["entrada"].fillna(0)
        relleno["salida"] = relleno["salida"].fillna(0)
        relleno = relleno.reset_index().rename(columns={"index": "venta_fecha"})
        series.append(relleno)
    return pd.concat(series, ignore_index=True)


def construir_features_ventas(grupo):
    """Build features for sales/demand prediction model"""
    base = grupo.copy().sort_values(["producto_id", "venta_fecha"])
    base["fecha_norm"] = pd.to_datetime(base["venta_fecha"])
    base["dia_semana"] = base["fecha_norm"].dt.dayofweek
    base["mes"] = base["fecha_norm"].dt.month
    inicio = base.groupby("producto_id")["fecha_norm"].transform("min")
    base["indice_fecha"] = (base["fecha_norm"] - inicio).dt.days.astype(float)

    # festivo, fin_semana, promo, descuento_pct vienen de la BD
    if "festivo" not in base.columns:
        base["festivo"] = 0
    if "fin_semana" not in base.columns:
        base["fin_semana"] = (base["dia_semana"] >= 5).astype(int)
    if "promo" not in base.columns:
        base["promo"] = 0
    if "descuento_pct" not in base.columns:
        base["descuento_pct"] = 0.0
    # Nuevas features de promociones
    if "promocion_id" not in base.columns:
        base["promocion_id"] = 0
    if "promocion_tipo" not in base.columns:
        base["promocion_tipo"] = "ninguna"
    if "promocion_descuento_pct" not in base.columns:
        base["promocion_descuento_pct"] = 0.0
    if "promocion_tipo_descuento" not in base.columns:
        base["promocion_tipo_descuento"] = "ninguna"
    
    # One-hot encode promocion_tipo
    for tipo in ["black_friday", "navidad", "cyber_monday", "aniversario", 
                 "dia_madre", "dia_padre", "san_valentin", "liquidacion",
                 "regreso_clases", "hot_sale", "otro", "ninguna"]:
        base[f"promo_tipo_{tipo}"] = (base["promocion_tipo"] == tipo).astype(int)
    
    # One-hot encode promocion_tipo_descuento
    for td in ["porcentaje", "monto_fijo", "2x1", "3x2", "envio_gratis", "ninguna"]:
        base[f"promo_desc_tipo_{td}"] = (base["promocion_tipo_descuento"] == td).astype(int)

    for lag in (1, 7, 14, 28):
        base[f"demanda_lag_{lag}"] = base.groupby("producto_id")["demanda"].shift(lag)
    base["demanda_media_7"] = base.groupby("producto_id")["demanda"].transform(lambda s: s.shift(1).rolling(7).mean())
    base["demanda_media_14"] = base.groupby("producto_id")["demanda"].transform(lambda s: s.shift(1).rolling(14).mean())
    base["demanda_max_7"] = base.groupby("producto_id")["demanda"].transform(lambda s: s.shift(1).rolling(7).max())

    base = base.dropna(subset=SALES_FEATURE_COLS).reset_index(drop=True)
    return base


def construir_features_inventario(grupo):
    """Build features for inventory/stock prediction model"""
    base = grupo.copy().sort_values(["producto_id", "venta_fecha"])
    base["fecha_norm"] = pd.to_datetime(base["venta_fecha"])
    base["dia_semana"] = base["fecha_norm"].dt.dayofweek
    base["mes"] = base["fecha_norm"].dt.month
    inicio = base.groupby("producto_id")["fecha_norm"].transform("min")
    base["indice_fecha"] = (base["fecha_norm"] - inicio).dt.days.astype(float)

    # Stock lags
    for lag in (1, 7, 14, 28):
        base[f"stock_lag_{lag}"] = base.groupby("producto_id")["stock_nuevo"].shift(lag)
    base["stock_media_7"] = base.groupby("producto_id")["stock_nuevo"].transform(lambda s: s.shift(1).rolling(7).mean())
    base["stock_media_14"] = base.groupby("producto_id")["stock_nuevo"].transform(lambda s: s.shift(1).rolling(14).mean())

    # Static product features
    base["stock_minimo"] = base["producto_stock_minimo"]
    base["stock_seguridad"] = base["producto_stock_seguridad"]
    base["tiempo_reposicion"] = base["producto_tiempo_reposicion_dias"]
    
    # One-hot encode rotacion
    base["rotacion_alta"] = (base["producto_rotacion"] == "Alta").astype(int)
    base["rotacion_media"] = (base["producto_rotacion"] == "Media").astype(int)
    base["rotacion_baja"] = (base["producto_rotacion"] == "Baja").astype(int)

    # Entrada/salida lags
    base["entrada_lag_1"] = base.groupby("producto_id")["entrada"].shift(1)
    base["salida_lag_1"] = base.groupby("producto_id")["salida"].shift(1)

    base = base.dropna(subset=INVENTORY_FEATURE_COLS).reset_index(drop=True)
    return base


def preparar_dataset_ventas(ruta=None):
    """Prepare dataset for sales/demand model"""
    df = cargar_dataset(ruta)
    agg = agregar_diaria_ventas(df)
    relleno = rellenar_series_ventas(agg)
    feat = construir_features_ventas(relleno)
    return feat


def preparar_dataset_inventario(ruta=None):
    """Prepare dataset for inventory/stock model"""
    df = cargar_dataset(ruta)
    agg = agregar_diaria_inventario(df)
    relleno = rellenar_series_inventario(agg)
    feat = construir_features_inventario(relleno)
    return feat


def preparar_dataset(ruta=None):
    """Default: prepare sales dataset (backward compatibility)"""
    return preparar_dataset_ventas(ruta)


def generar_features_recientes_ventas(hist, producto_id, ultima_fecha, n_dias=30):
    part = hist[hist["producto_id"] == producto_id].sort_values("venta_fecha")
    if part.empty or len(part) < 30:
        return None
    ultimos = part.tail(max(n_dias, 30))
    
    # festivo, fin_semana, promo, descuento_pct, promocion_id, promocion_tipo, promocion_descuento_pct, promocion_tipo_descuento
    festivo = 0.0
    fin_semana = 0.0
    promo = 0.0
    descuento_pct = 0.0
    promocion_id = 0
    promocion_tipo = "ninguna"
    promocion_descuento_pct = 0.0
    promocion_tipo_descuento = "ninguna"
    
    if "festivo" in part.columns:
        festivo = float(part["festivo"].iloc[-1])
    if "fin_semana" in part.columns:
        fin_semana = float(part["fin_semana"].iloc[-1])
    else:
        fin_semana = 1.0 if ultima_fecha.dayofweek >= 5 else 0.0
    if "promo" in part.columns:
        promo = float(part["promo"].iloc[-1])
    if "descuento_pct" in part.columns:
        descuento_pct = float(part["descuento_pct"].iloc[-1])
    if "promocion_id" in part.columns:
        promocion_id = float(part["promocion_id"].iloc[-1])
    if "promocion_tipo" in part.columns:
        promocion_tipo = part["promocion_tipo"].iloc[-1]
    if "promocion_descuento_pct" in part.columns:
        promocion_descuento_pct = float(part["promocion_descuento_pct"].iloc[-1])
    if "promocion_tipo_descuento" in part.columns:
        promocion_tipo_descuento = part["promocion_tipo_descuento"].iloc[-1]
    
    # One-hot encode promocion_tipo
    promo_tipo_dict = {}
    for tipo in ["black_friday", "navidad", "cyber_monday", "aniversario", 
                 "dia_madre", "dia_padre", "san_valentin", "liquidacion",
                 "regreso_clases", "hot_sale", "otro", "ninguna"]:
        promo_tipo_dict[f"promo_tipo_{tipo}"] = 1.0 if promocion_tipo == tipo else 0.0
    
    # One-hot encode promocion_tipo_descuento
    promo_desc_dict = {}
    for td in ["porcentaje", "monto_fijo", "2x1", "3x2", "envio_gratis", "ninguna"]:
        promo_desc_dict[f"promo_desc_tipo_{td}"] = 1.0 if promocion_tipo_descuento == td else 0.0
    
    vector = {
        "dia_semana": float(ultima_fecha.dayofweek),
        "mes": float(ultima_fecha.month),
        "indice_fecha": float((ultima_fecha - part["venta_fecha"].min()).days),
        "festivo": festivo,
        "fin_semana": fin_semana,
        "promo": promo,
        "descuento_pct": descuento_pct,
        "demanda_lag_1": float(ultimos["demanda"].iloc[-1]),
        "demanda_lag_7": float(ultimos["demanda"].iloc[-8]),
        "demanda_lag_14": float(ultimos["demanda"].iloc[-15]),
        "demanda_lag_28": float(ultimos["demanda"].iloc[-29]),
        "demanda_media_7": float(ultimos["demanda"].iloc[-8:].mean()),
        "demanda_media_14": float(ultimos["demanda"].iloc[-14:].mean()),
        "demanda_max_7": float(ultimos["demanda"].iloc[-8:].max()),
    }
    return np.array([list(vector.values())])


def generar_features_recientes_inventario(hist, producto_id, ultima_fecha, n_dias=30):
    part = hist[hist["producto_id"] == producto_id].sort_values("venta_fecha")
    if part.empty or len(part) < 30:
        return None
    ultimos = part.tail(max(n_dias, 30))
    # Get static features
    static = part.iloc[-1]
    vector = {
        "dia_semana": float(ultima_fecha.dayofweek),
        "mes": float(ultima_fecha.month),
        "indice_fecha": float((ultima_fecha - part["venta_fecha"].min()).days),
        "stock_lag_1": float(ultimos["stock_nuevo"].iloc[-1]),
        "stock_lag_7": float(ultimos["stock_nuevo"].iloc[-8]),
        "stock_lag_14": float(ultimos["stock_nuevo"].iloc[-15]),
        "stock_lag_28": float(ultimos["stock_nuevo"].iloc[-29]),
        "stock_media_7": float(ultimos["stock_nuevo"].iloc[-8:].mean()),
        "stock_media_14": float(ultimos["stock_nuevo"].iloc[-14:].mean()),
        "stock_minimo": float(static["producto_stock_minimo"]),
        "stock_seguridad": float(static["producto_stock_seguridad"]),
        "tiempo_reposicion": float(static["producto_tiempo_reposicion_dias"]),
        "rotacion_alta": float(static["rotacion_alta"]),
        "rotacion_media": float(static["rotacion_media"]),
        "rotacion_baja": float(static["rotacion_baja"]),
        "entrada_lag_1": float(ultimos["entrada"].iloc[-1]),
        "salida_lag_1": float(ultimos["salida"].iloc[-1]),
    }
    return np.array([list(vector.values())])


# Backward compatibility
FEATURE_COLS = SALES_FEATURE_COLS
agregar_diaria = agregar_diaria_ventas
rellenar_series = rellenar_series_ventas
construir_features = construir_features_ventas
generar_features_recientes = generar_features_recientes_ventas