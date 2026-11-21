from app.services.preprocess import agregar_diaria, cargar_dataset

UMBRAL_ALTA_ROTACION = 1.5
UMBRAL_MEDIA_ROTACION = 0.5


def clasificar_rotacion(demanda_diaria_media):
    if demanda_diaria_media >= UMBRAL_ALTA_ROTACION:
        return "ALTA_ROTACION"
    if demanda_diaria_media >= UMBRAL_MEDIA_ROTACION:
        return "MEDIA_ROTACION"
    if demanda_diaria_media > 0:
        return "BAJA_ROTACION"
    return "SIN_MOVIMIENTO"


def historico_diario_medio(dataset=None):
    data = dataset if dataset is not None else cargar_dataset()
    agg = agregar_diaria(data)
    medio = agg.groupby("producto_id")["demanda"].mean().to_dict()
    return medio


def generar_recomendaciones(predicciones, params=None):
    params = params or {}
    resultados = []
    stock_minimo_map = params.get("stock_minimo") or {}
    stock_seguridad_map = params.get("stock_seguridad") or {}
    rotacion_map = params.get("rotacion") or {}
    for p in predicciones:
        pid = p["producto_id"]
        demanda_total = float(p["demanda_total_proyectada"])
        stock = int(p.get("stock_actual", 0))
        stock_minimo = int(stock_minimo_map.get(str(pid), 0))
        stock_seguridad = int(stock_seguridad_map.get(str(pid), 0))
        rotacion = rotacion_map.get(str(pid), "SIN_MOVIMIENTO")

        sugerencia = None
        riesgo = "STOCK_ADECUADO"
        motivo = ""

        if demanda_total <= 0:
            riesgo = "SIN_DEMANDA"
            motivo = "Sin demanda proyectada en el horizonte."
        elif stock <= 0:
            riesgo = "AGOTADO_QUIEBRE"
            sugerencia = int(demanda_total + stock_seguridad)
            motivo = "Producto agotado con demanda proyectada."
        elif demanda_total + stock_seguridad > stock:
            riesgo = "RIESGO_QUIEBRE_STOCK"
            sugerencia = int(max(0, demanda_total + stock_seguridad - stock))
            motivo = f"Demanda proyectada {demanda_total:.1f} supera stock {stock} (margen {stock_seguridad})."
        elif stock > 0 and stock > demanda_total * 3:
            if rotacion in ("BAJA_ROTACION", "SIN_MOVIMIENTO"):
                riesgo = "EXCESO_INVENTARIO"
                sugerencia = 0
                motivo = "Stock alto con baja rotacion; evaluar promocion."
            else:
                riesgo = "STOCK_ADECUADO"
                motivo = "Stock suficiente para la demanda proyectada."
        else:
            riesgo = "STOCK_ADECUADO"
            motivo = "Stock suficiente para la demanda proyectada."

        resultados.append(
            {
                "producto_id": pid,
                "producto_codigo": p["producto_codigo"],
                "producto_nombre": p["producto_nombre"],
                "stock_actual": stock,
                "demanda_total_proyectada": round(demanda_total, 2),
                "stock_minimo": stock_minimo,
                "stock_seguridad": stock_seguridad,
                "rotacion": rotacion,
                "riesgo": riesgo,
                "cantidad_sugerida": sugerencia if sugerencia else 0,
                "motivo": motivo,
            }
        )
    resultados.sort(
        key=lambda r: (0 if "QUIEBRE" in r["riesgo"] else 1 if "AGOTADO" in r["riesgo"] else 2 if r["riesgo"] == "EXCESO_INVENTARIO" else 3),
    )
    return resultados