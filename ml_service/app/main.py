
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router

app = FastAPI(
    title="SERVTEC ml_service",
    description=(
        "Predicción de demanda, recomendaciones y gemelo digital "
        "(solo consulta, nunca escribe stock real)."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


if __name__ == "__main__":
    import uvicorn

    puerto = int(os.environ.get("PORT", os.environ.get("ML_PORT", "8000")))
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=puerto,
        reload=False,
    )
