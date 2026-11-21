from typing import Dict, List, Optional

from pydantic import BaseModel


class PredictRequest(BaseModel):
    horizonte: int = 30
    stock_actual: Optional[Dict] = None


class PredictInventarioRequest(BaseModel):
    horizonte: int = 30


class PredictBothRequest(BaseModel):
    horizonte: int = 30
    stock_actual: Optional[Dict] = None


class SimulateRequest(BaseModel):
    prediccion: Dict
    compra_simulada: int = 0
    variacion_demanda_pct: float = 0.0


class RecommendationsRequest(BaseModel):
    predicciones: List[Dict]
    stock_minimo: Optional[Dict] = None
    stock_seguridad: Optional[Dict] = None
    rotacion: Optional[Dict] = None


class BacktestRequest(BaseModel):
    fecha_inicio: str
    fecha_fin: str


class TrainResponse(BaseModel):
    entrenados: int
    mejor_ventas: Optional[str] = None
    mejor_inventario: Optional[str] = None
    cambio_ventas: Optional[bool] = None
    cambio_inventario: Optional[bool] = None
    resultados: Optional[Dict] = None


class HealthResponse(BaseModel):
    status: str
    servicio: str
    version: Optional[str] = None