# Arranca el servicio ML solo si no esta respondiendo.
# Idempotente: pensado para ejecutarse al iniciar sesion (sin duplicar el servicio).
$raiz = "C:\xampp\htdocs\VENTAS\ml_service"
$base = "http://127.0.0.1:8000"

try {
    $r = Invoke-WebRequest -Uri "$base/health" -TimeoutSec 4 -UseBasicParsing
    if ($r.StatusCode -eq 200) { exit 0 }
} catch { }

Start-Process -FilePath "cmd.exe" -ArgumentList '/c', "`"$raiz\run_server.bat`"" -WindowStyle Hidden
exit 0
