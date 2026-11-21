# Reentrenamiento mensual del modelo ML (VENTAS).
#
# El servicio ML lee el dataset EN VIVO de la base de datos (ver db_source.py),
# asi que este script NO exporta CSV: solo se asegura de que el servicio este
# encendido y llama a POST /train. No crea ni modifica tablas.
#
# Programado por el Programador de tareas de Windows:
#   "VENTAS - Reentrenar modelo ML (mensual)"  (dia 1 de cada mes)

$ErrorActionPreference = "Stop"
$raiz = "C:\xampp\htdocs\VENTAS\ml_service"
$log = Join-Path $raiz "logs\retrain.log"
$base = "http://127.0.0.1:8000"

if (-not (Test-Path (Join-Path $raiz "logs"))) {
    New-Item -ItemType Directory -Path (Join-Path $raiz "logs") -Force | Out-Null
}

function Escribir-Log($mensaje) {
    $linea = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $mensaje
    Add-Content -Path $log -Value $linea -Encoding UTF8
}

function Probar-Salud {
    try {
        $r = Invoke-WebRequest -Uri "$base/health" -TimeoutSec 5 -UseBasicParsing
        return ($r.StatusCode -eq 200)
    } catch {
        return $false
    }
}

try {
    Escribir-Log "=== Inicio de reentrenamiento mensual ==="

    if (-not (Probar-Salud)) {
        Escribir-Log "El servicio ML no responde; iniciandolo..."
        Start-Process -FilePath "cmd.exe" -ArgumentList '/c', "`"$raiz\run_server.bat`"" -WindowStyle Hidden
        $arriba = $false
        for ($i = 0; $i -lt 40; $i++) {
            Start-Sleep -Seconds 3
            if (Probar-Salud) { $arriba = $true; break }
        }
        if (-not $arriba) {
            Escribir-Log "ERROR: el servicio ML no arranco a tiempo; se cancela el reentrenamiento."
            exit 1
        }
        Escribir-Log "Servicio ML iniciado correctamente."
    } else {
        Escribir-Log "El servicio ML ya estaba en ejecucion."
    }

    Escribir-Log "Llamando a POST /train (puede tardar varios minutos)..."
    $r = Invoke-RestMethod -Uri "$base/train" -Method Post -TimeoutSec 7200
    Escribir-Log ("OK. entrenados={0} | ventas={1} (cambio={2}) | inventario={3} (cambio={4})" -f $r.entrenados, $r.mejor_ventas, $r.cambio_ventas, $r.mejor_inventario, $r.cambio_inventario)
    Escribir-Log ("Detalle: " + ($r.resultados | ConvertTo-Json -Depth 10 -Compress))
    Escribir-Log "=== Fin de reentrenamiento (OK) ==="
    exit 0
} catch {
    Escribir-Log ("ERROR: " + $_.Exception.Message)
    exit 1
}
