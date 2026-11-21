# -*- coding: utf-8 -*-
"""Origen de datos del servicio ML: lee el dataset directamente de MySQL.

Antes el servicio leia un CSV congelado (data/raw/ventas.csv) generado a mano,
por lo que no aprendia de las ventas nuevas. Este modulo reproduce exactamente
el mismo dataset (una fila por movimiento de kardex) mediante una consulta SQL,
de modo que el modelo siempre entrena con los datos vivos de la base de datos.

No crea ni modifica tablas: solo hace SELECT.
"""

import os

import pandas as pd

# Consulta que replica el dataset del CSV (validada 1:1 contra ventas.csv):
#  - una fila por cada movimiento de kardex
#  - "cantidad"/"salida" = cantidad del movimiento cuando es Salida (venta)
#  - "entrada"            = cantidad del movimiento cuando es Entrada (compra)
#  - rotacion en formato amigable (Alta/Media/Baja) como esperaba el CSV
#  - NUEVO: join con calendario para festivo / fin_semana
#  - NUEVO: promo y descuento_pct desde kardex
#  - NUEVO: join con promociones para promocion_id, tipo, descuento, etc.
SQL_DATASET = """
SELECT
  k.kardex_fecha                                    AS venta_fecha,
  k.producto_id                                     AS producto_id,
  p.producto_codigo                                 AS producto_codigo,
  p.producto_nombre                                 AS producto_nombre,
  c.categoria_nombre                                AS categoria_nombre,
  p.producto_marca                                  AS producto_marca,
  p.producto_modelo                                 AS producto_modelo,
  CASE WHEN k.kardex_tipo = 'Salida'  THEN k.kardex_cantidad ELSE 0 END AS cantidad,
  CASE WHEN k.kardex_tipo = 'Entrada' THEN k.kardex_cantidad ELSE 0 END AS entrada,
  CASE WHEN k.kardex_tipo = 'Salida'  THEN k.kardex_cantidad ELSE 0 END AS salida,
  k.kardex_stock_anterior                           AS stock_anterior,
  k.kardex_stock_nuevo                              AS stock_nuevo,
  p.producto_stock_minimo                           AS producto_stock_minimo,
  p.producto_stock_seguridad                        AS producto_stock_seguridad,
  p.producto_tiempo_reposicion_dias                 AS producto_tiempo_reposicion_dias,
  CASE p.producto_rotacion
    WHEN 'ALTA_ROTACION'  THEN 'Alta'
    WHEN 'MEDIA_ROTACION' THEN 'Media'
    WHEN 'BAJA_ROTACION'  THEN 'Baja'
    ELSE p.producto_rotacion
  END                                               AS producto_rotacion,
  cal.festivo                                       AS festivo,
  cal.fin_semana                                    AS fin_semana,
  k.promo                                           AS promo,
  k.descuento_pct                                   AS descuento_pct,
  k.promocion_id                                    AS promocion_id,
  pr.tipo                                           AS promocion_tipo,
  pr.descuento_pct                                  AS promocion_descuento_pct,
  pr.tipo_descuento                                 AS promocion_tipo_descuento
FROM kardex k
JOIN producto p  ON p.producto_id  = k.producto_id
JOIN categoria c ON c.categoria_id = p.categoria_id
LEFT JOIN calendario cal ON cal.fecha = k.kardex_fecha
LEFT JOIN promociones pr ON pr.promocion_id = k.promocion_id
ORDER BY k.producto_id, k.kardex_fecha, k.kardex_id
"""


def configuracion_bd():
    """Configuracion de conexion (configurable por variables de entorno)."""
    return {
        "host": os.environ.get("ML_DB_HOST", "127.0.0.1"),
        "port": int(os.environ.get("ML_DB_PORT", "3306")),
        "user": os.environ.get("ML_DB_USER", "root"),
        "password": os.environ.get("ML_DB_PASS", ""),
        "database": os.environ.get("ML_DB_NAME", "ventas"),
        "charset": "utf8mb4",
    }


def _conectar():
    import pymysql

    return pymysql.connect(**configuracion_bd())


def leer_dataset():
    """Devuelve el dataset de movimientos (producto/dia) leido de la BD."""
    conn = _conectar()
    try:
        cur = conn.cursor()
        try:
            cur.execute(SQL_DATASET)
            columnas = [d[0] for d in cur.description]
            filas = cur.fetchall()
        finally:
            cur.close()
    finally:
        conn.close()

    df = pd.DataFrame(filas, columns=columnas)
    df["venta_fecha"] = pd.to_datetime(df["venta_fecha"])
    for col in ("cantidad", "entrada", "salida", "stock_anterior", "stock_nuevo",
                "producto_stock_minimo", "producto_stock_seguridad",
                "producto_tiempo_reposicion_dias"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    return df


def firma_bd():
    """Firma ligera para saber si aparecieron movimientos nuevos (cache).

    Devuelve (cantidad_de_movimientos, ultimo_kardex_id) o lanza excepcion si
    la BD no esta disponible (el llamador decide usar el respaldo en cache/CSV).
    """
    conn = _conectar()
    try:
        cur = conn.cursor()
        try:
            cur.execute("SELECT COUNT(*), COALESCE(MAX(kardex_id), 0) FROM kardex")
            fila = cur.fetchone()
        finally:
            cur.close()
    finally:
        conn.close()
    return int(fila[0]), int(fila[1])