@echo off
set ML_DATA_SOURCE=db
set ML_DB_HOST=127.0.0.1
set ML_DB_PORT=3306
set ML_DB_NAME=ventas
set ML_DB_USER=root
set ML_DB_PASS=
cd /d C:\xampp\htdocs\VENTAS\ml_service
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000