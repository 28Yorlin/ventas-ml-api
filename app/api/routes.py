from fastapi import APIRouter, HTTPException

from app.services.backtest import BacktestError, predecir_backtest
from app.services.gemelo import simular
from app.services.predict import (
    predecir_ambos,
    predecir_demanda,
    predecir_demanda_serie,
    predecir_inventario,
)
from app.services.recommendations import generar_recomendaciones, historico_diario_medio
from app.services.train import entrenar, leer_registro, modelo_activo, get_modelos_by_tipo
from app.schemas.schemas import (
    BacktestRequest,
    PredictBothRequest,
    PredictInventarioRequest,
    PredictRequest,
    RecommendationsRequest,
    SimulateRequest,
    TrainResponse,
)

router = APIRouter()


@router.get("/health")
def health():
    entry_ventas = modelo_activo(modelo_tipo="ventas")
    entry_inventario = modelo_activo(modelo_tipo="inventario")
    return {
        "status": "ok",
        "servicio": "ml_service",
        "modelo_activo_ventas": entry_ventas["version"] if entry_ventas else None,
        "modelo_activo_inventario": entry_inventario["version"] if entry_inventario else None,
    }


@router.get("/model-status")
def model_status():
    registro = leer_registro()
    activo_ventas = modelo_activo(modelo_tipo="ventas")
    activo_inventario = modelo_activo(modelo_tipo="inventario")
    return {
        "activo_ventas": activo_ventas if activo_ventas else None,
        "activo_inventario": activo_inventario if activo_inventario else None,
        "modelos_ventas": sorted(
            [v for v in registro.values() if v.get("modelo_tipo") == "ventas"],
            key=lambda v: v.get("fecha", ""), reverse=True
        ) if registro else [],
        "modelos_inventario": sorted(
            [v for v in registro.values() if v.get("modelo_tipo") == "inventario"],
            key=lambda v: v.get("fecha", ""), reverse=True
        ) if registro else [],
    }


@router.post("/train", response_model=TrainResponse)
def entrenar_modelos():
    try:
        return entrenar()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/predict")
def predecir(req: PredictRequest):
    try:
        return predecir_demanda(horizonte=req.horizonte, stock_actual=req.stock_actual)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/predict-serie")
def predecir_serie(req: PredictRequest):
    try:
        return predecir_demanda_serie(horizonte=req.horizonte, stock_actual=req.stock_actual)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/predict-inventario")
def predecir_inventario_endpoint(req: PredictInventarioRequest):
    try:
        return predecir_inventario(horizonte=req.horizonte)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/predict-both")
def predecir_both(req: PredictBothRequest):
    try:
        return predecir_ambos(horizonte=req.horizonte, stock_actual=req.stock_actual)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/recommendations")
def recomendaciones(req: RecommendationsRequest):
    try:
        return generar_recomendaciones(req.predicciones, req.model_dump())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/simulate")
def gemelo(req: SimulateRequest):
    try:
        return simular(
            req.prediccion,
            compra_simulada=req.compra_simulada,
            variacion_demanda_pct=req.variacion_demanda_pct,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/rotation")
def rotacion():
    return historico_diario_medio()


@router.post("/backtest")
def backtest(req: BacktestRequest):
    """Compara el modelo activo contra ventas REALES en una ventana pasada."""
    try:
        return predecir_backtest(req.fecha_inicio, req.fecha_fin)
    except BacktestError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))