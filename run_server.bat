@echo off
cd /d "%~dp0"
rem --- Origen de datos: base de datos viva (sin exportar a CSV) ---
set ML_DATA_SOURCE=db
set ML_DB_HOST=127.0.0.1
set ML_DB_PORT=3306
set ML_DB_NAME=ventas
set ML_DB_USER=root
set ML_DB_PASS=
rem --- Respaldo automatico si la BD no esta disponible ---
set ML_RAW_CSV=%~dp0data\raw\ventas.csv
set ML_MODELS_DIR=%~dp0models
if not exist logs mkdir logs
python -X utf8 -m uvicorn app.main:app --host 127.0.0.1 --port 8000 >> logs\server.log 2>&1