def simular(prediccion, compra_simulada=0, variacion_demanda_pct=0):
    stock_actual = int(prediccion.get("stock_actual", 0))
    demanda = float(prediccion.get("demanda_total_proyectada", 0))
    horizonte = int(prediccion.get("horizonte_dias", 30))

    factor = 1.0 + (float(variacion_demanda_pct) / 100.0)
    if factor < 0:
        factor = 0.0
    demanda_efectiva = round(demanda * factor, 4)

    stock_simulado = int(stock_actual - demanda_efectiva + int(compra_simulada))

    if stock_simulado < 0:
        riesgo = "RIESGO_QUIEBRE_STOCK"
        observacion = (
            f"La simulacion deja stock negativo (-{abs(stock_simulado)}). "
            f"Recomendado cubrir al menos {abs(stock_simulado)} unidades adicionales."
        )
        stock_simulado = 0
    elif stock_simulado == 0:
        riesgo = "STOCK_AGOTADO"
        observacion = "La simulacion dejaria el stock en cero exactamente al final del horizonte."
    elif stock_simulado < stock_actual:
        riesgo = "STOCK_PARCIAL"
        observacion = (
            f"Stock final ({stock_simulado}) menor al actual ({stock_actual}): la compra simulada "
            "no cubre la demanda proyectada del horizonte."
        )
    elif stock_simulado >= demanda_efectiva * 3 and compra_simulada > 0:
        riesgo = "EXCESO_STOCK"
        observacion = "La compra simulada genera exceso de inventario frente a la demanda proyectada."
    else:
        riesgo = "STOCK_SUFICIENTE"
        observacion = "La simulacion cubre la demanda proyectada sin dejar el stock agotado."

    return {
        "stock_actual": stock_actual,
        "demanda_proyectada_original": demanda,
        "variacion_demanda_pct": float(variacion_demanda_pct),
        "demanda_efectiva": demanda_efectiva,
        "compra_simulada": int(compra_simulada),
        "stock_simulado": stock_simulado,
        "riesgo": riesgo,
        "observacion": observacion,
        "horizonte_dias": horizonte,
        "es_simulacion": True,
        "stock_real_sin_cambios": stock_actual,
    }