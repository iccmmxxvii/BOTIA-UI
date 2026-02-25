# RUN UI VISUAL

## 1) Iniciar
```bash
cd /workspace/BOTIA-UI
scripts/ui_visual_start.sh
```

Por defecto arranca en `8502`. Para cambiar puerto:
```bash
UI_PORT=8510 scripts/ui_visual_start.sh
```

## 2) Abrir en navegador local
Si estás en la misma máquina:
- `http://localhost:8502`

Si estás remoto por SSH, usa túnel:
```bash
ssh -L 8502:localhost:8502 <usuario>@<host>
```
Luego abre:
- `http://localhost:8502`

## 3) Logs
```bash
tail -f logs/ui_visual.log
```

## 4) Stop
```bash
scripts/ui_visual_stop.sh
```

## Datos reales (paper)
La UI busca DB en este orden:
1. `BOTIA_DB_PATH`
2. `*.sqlite/*.sqlite3/*.db` dentro del repo
3. Ruta manual en sidebar

Control UI se guarda en:
- `data/botia_control.json`
