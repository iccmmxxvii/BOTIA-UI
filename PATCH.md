# PATCH propuesto para loop del bot (mínimo)

Objetivo: que el loop principal de paper trading lea `data/botia_control.json` y respete:
- `auto_trade`
- `risk_level`
- `frequency`

## 1) Archivo a tocar (sugerido)
- Donde viva el loop principal de ejecución de señales/órdenes (ej: `bot/main.py`, `engine/loop.py`, o equivalente).

## 2) Diff mínimo sugerido
```diff
+ import json
+ from pathlib import Path
+
+ CONTROL_PATH = Path("data/botia_control.json")
+
+ def read_control():
+     defaults = {
+         "auto_trade": False,
+         "risk_level": "moderate",
+         "frequency": "normal",
+         "refresh_rate": 2,
+     }
+     try:
+         payload = json.loads(CONTROL_PATH.read_text())
+         defaults.update(payload)
+     except Exception:
+         pass
+     return defaults
@@
- while True:
+ while True:
+     control = read_control()
+     if not control["auto_trade"]:
+         time.sleep(1)
+         continue
+
+     # mapear frequency -> sleep o intervalo de evaluación
+     freq_map = {"slow": 5, "normal": 2, "fast": 1, "turbo": 0.5}
+     loop_delay = freq_map.get(control["frequency"], 2)
+
+     # mapear risk_level -> sizing/rules de riesgo
+     risk_multiplier = {
+         "conservative": 0.5,
+         "moderate": 1.0,
+         "aggressive": 1.5,
+         "degen": 2.0,
+     }.get(control["risk_level"], 1.0)
+
+     # usar risk_multiplier en sizing de paper
+     # size = base_size * risk_multiplier
+
      run_one_cycle()
-     time.sleep(2)
+     time.sleep(loop_delay)
```

## 3) Nota
Si el proyecto ya tiene lectura de control runtime, sólo enlazar claves y documentar mapeos de `frequency/risk_level`.
