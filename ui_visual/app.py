import json
import os
import sqlite3
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import altair as alt
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
LOGS_DIR = ROOT / "logs"
CONTROL_PATH = DATA_DIR / "botia_control.json"

DEFAULT_CONTROL = {
    "auto_trade": False,
    "risk_level": "moderate",
    "frequency": "normal",
    "refresh_rate": 2,
}

RISK_OPTIONS = ["conservative", "moderate", "aggressive", "degen"]
FREQUENCY_OPTIONS = ["slow", "normal", "fast", "turbo"]

TS_CANDIDATES = ["ts", "timestamp", "datetime", "created_at", "time", "date"]
PRICE_CANDIDATES = ["price", "mark", "last", "close", "value"]
PNL_CANDIDATES = ["pnl", "realized_pnl", "pnl_usd", "profit", "pl"]
PTB_CANDIDATES = ["price_to_beat", "ptb", "start_price", "open_price", "target_price"]


@dataclass
class SchemaHints:
    ticks_table: Optional[str] = None
    ticks_ts_col: Optional[str] = None
    ticks_price_col: Optional[str] = None
    trades_table: Optional[str] = None
    trades_ts_col: Optional[str] = None
    trades_pnl_col: Optional[str] = None
    trades_side_col: Optional[str] = None
    trades_price_col: Optional[str] = None
    trades_size_col: Optional[str] = None
    rounds_table: Optional[str] = None
    rounds_ptb_col: Optional[str] = None
    rounds_end_col: Optional[str] = None


def fmt_money(value: float) -> str:
    return f"${value:,.2f}"


def load_css() -> None:
    css_path = Path(__file__).parent / "assets" / "terminal.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)


def keyboard_shortcuts() -> None:
    components.html(
        """
        <script>
            const handler = (e) => {
                const map = {s: 'sound', c: 'settings', t: 'theme', '?': 'help'};
                const k = e.key.toLowerCase();
                if (map[k]) {
                    const url = new URL(window.parent.location.href);
                    url.searchParams.set('shortcut', map[k]);
                    url.searchParams.set('shortcut_ts', Date.now().toString());
                    window.parent.location.href = url.toString();
                }
            };
            window.addEventListener('keydown', handler);
        </script>
        """,
        height=0,
    )


def ensure_control_file() -> Dict:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not CONTROL_PATH.exists():
        write_control(DEFAULT_CONTROL)
    try:
        return {**DEFAULT_CONTROL, **json.loads(CONTROL_PATH.read_text())}
    except Exception:
        return DEFAULT_CONTROL.copy()


def write_control(payload: Dict) -> None:
    CONTROL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=CONTROL_PATH.parent, encoding="utf-8") as tmp:
        json.dump(payload, tmp, indent=2)
        tmp.flush()
        os.fsync(tmp.fileno())
    os.replace(tmp.name, CONTROL_PATH)


def detect_db_candidates() -> List[Path]:
    env_path = os.getenv("BOTIA_DB_PATH")
    candidates = []
    if env_path:
        p = Path(env_path).expanduser()
        if p.exists():
            candidates.append(p)
    patterns = ["*.sqlite", "*.sqlite3", "*.db"]
    for pattern in patterns:
        candidates.extend(ROOT.rglob(pattern))
    uniq = []
    seen = set()
    for c in candidates:
        rc = c.resolve()
        if rc not in seen:
            seen.add(rc)
            uniq.append(rc)
    return uniq


def inspect_schema(conn: sqlite3.Connection) -> SchemaHints:
    hints = SchemaHints()
    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]

    for table in tables:
        cols = [r[1].lower() for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]
        if not hints.ticks_table and ("tick" in table.lower() or any(c in cols for c in PRICE_CANDIDATES)):
            ts = next((c for c in cols if c in TS_CANDIDATES), None)
            pc = next((c for c in cols if c in PRICE_CANDIDATES), None)
            if ts and pc:
                hints.ticks_table, hints.ticks_ts_col, hints.ticks_price_col = table, ts, pc

        if not hints.trades_table and "trade" in table.lower():
            ts = next((c for c in cols if c in TS_CANDIDATES), None)
            pnl = next((c for c in cols if c in PNL_CANDIDATES), None)
            side = next((c for c in cols if c in ["side", "direction"]), None)
            price = next((c for c in cols if c in PRICE_CANDIDATES), None)
            size = next((c for c in cols if c in ["size", "qty", "quantity", "amount"]), None)
            hints.trades_table, hints.trades_ts_col, hints.trades_pnl_col = table, ts, pnl
            hints.trades_side_col, hints.trades_price_col, hints.trades_size_col = side, price, size

        if not hints.rounds_table and "round" in table.lower():
            ptb = next((c for c in cols if c in PTB_CANDIDATES), None)
            end_col = next((c for c in cols if c in ["end_time", "close_time", "ends_at", "resolve_time"]), None)
            if ptb:
                hints.rounds_table, hints.rounds_ptb_col, hints.rounds_end_col = table, ptb, end_col
    return hints


@st.cache_data(ttl=2)
def read_db(db_path: str) -> Tuple[SchemaHints, pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    notices: List[str] = []
    conn = sqlite3.connect(db_path)
    hints = inspect_schema(conn)

    ticks_df = pd.DataFrame()
    trades_df = pd.DataFrame()
    rounds_df = pd.DataFrame()

    if hints.ticks_table and hints.ticks_ts_col and hints.ticks_price_col:
        ticks_df = pd.read_sql_query(
            f"SELECT {hints.ticks_ts_col} as ts, {hints.ticks_price_col} as price FROM {hints.ticks_table} ORDER BY 1 DESC LIMIT 1000",
            conn,
        )
    else:
        notices.append("No se encontró tabla/columnas de ticks compatibles.")

    if hints.trades_table:
        select_cols = []
        for src, dst in [
            (hints.trades_ts_col, "ts"),
            (hints.trades_pnl_col, "pnl"),
            (hints.trades_side_col, "side"),
            (hints.trades_price_col, "price"),
            (hints.trades_size_col, "size"),
        ]:
            if src:
                select_cols.append(f"{src} as {dst}")
        if select_cols:
            trades_df = pd.read_sql_query(
                f"SELECT {', '.join(select_cols)} FROM {hints.trades_table} ORDER BY 1 DESC LIMIT 500",
                conn,
            )
        if not hints.trades_pnl_col:
            notices.append("Trades detectados, pero sin columna P/L reconocible.")
    else:
        notices.append("No se encontró tabla de trades compatible.")

    if hints.rounds_table and hints.rounds_ptb_col:
        cols = [f"{hints.rounds_ptb_col} as price_to_beat"]
        if hints.rounds_end_col:
            cols.append(f"{hints.rounds_end_col} as end_time")
        rounds_df = pd.read_sql_query(
            f"SELECT {', '.join(cols)} FROM {hints.rounds_table} ORDER BY ROWID DESC LIMIT 100",
            conn,
        )
    else:
        notices.append("No se encontró tabla de rounds compatible para Price To Beat.")

    conn.close()
    return hints, ticks_df, trades_df, rounds_df, notices


def parse_time_series(df: pd.DataFrame, ts_col: str = "ts") -> pd.DataFrame:
    if df.empty or ts_col not in df.columns:
        return df
    out = df.copy()
    out[ts_col] = pd.to_datetime(out[ts_col], errors="coerce", utc=True)
    out = out.dropna(subset=[ts_col]).sort_values(ts_col)
    return out


def compute_countdown() -> Tuple[int, str, str]:
    now = datetime.now(timezone.utc)
    bucket_seconds = 5 * 60
    next_boundary = int((now.timestamp() // bucket_seconds + 1) * bucket_seconds)
    remaining = max(0, next_boundary - int(now.timestamp()))
    next_dt = datetime.fromtimestamp(next_boundary, tz=timezone.utc)
    et = next_dt.astimezone(ZoneInfo("America/New_York")).strftime("%Y-%m-%d %H:%M:%S %Z")
    utc = next_dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    return remaining, et, utc


def read_log_tail(path: Path, lines: int = 30) -> List[str]:
    if not path.exists():
        return []
    return path.read_text(errors="ignore").splitlines()[-lines:]


def main() -> None:
    st.set_page_config(page_title="BOTIA Visual Terminal", layout="wide")
    load_css()
    keyboard_shortcuts()

    qp = st.query_params
    shortcut = qp.get("shortcut", "")
    if shortcut == "settings":
        st.session_state["show_settings"] = True
    elif shortcut == "theme":
        st.session_state["show_theme"] = True
    elif shortcut == "sound":
        st.session_state["sound_on"] = not st.session_state.get("sound_on", False)
    elif shortcut == "help":
        st.session_state["show_help"] = True

    control = ensure_control_file()

    st.markdown('<div class="crt-overlay"></div>', unsafe_allow_html=True)
    st.markdown('<div class="topbar"><div class="title">BOTIA // VISUAL TERMINAL</div></div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns([1, 1, 1, 3])
    if c1.button("SETTINGS"):
        st.session_state["show_settings"] = True
    if c2.button("THEME"):
        st.session_state["show_theme"] = True
    if c3.button("SOUND"):
        st.session_state["sound_on"] = not st.session_state.get("sound_on", False)

    with st.sidebar:
        st.subheader("Connection")
        candidates = detect_db_candidates()
        options = [str(c) for c in candidates]
        selected = st.selectbox("Detected DB", options=["(none)"] + options)
        manual = st.text_input("Manual DB path", value="")
        db_path = manual.strip() or ("" if selected == "(none)" else selected)

        st.subheader("Control (paper)")
        auto_trade = st.toggle("auto_trade", value=bool(control.get("auto_trade", False)))
        risk = st.selectbox("risk_level", RISK_OPTIONS, index=RISK_OPTIONS.index(control.get("risk_level", "moderate")))
        frequency = st.selectbox(
            "frequency", FREQUENCY_OPTIONS, index=FREQUENCY_OPTIONS.index(control.get("frequency", "normal"))
        )
        refresh_rate = st.slider("refresh_rate (s)", 1, 5, int(control.get("refresh_rate", 2)))

        new_control = {
            "auto_trade": auto_trade,
            "risk_level": risk,
            "frequency": frequency,
            "refresh_rate": refresh_rate,
        }
        if new_control != control:
            write_control(new_control)
            st.success("Control guardado en data/botia_control.json")
        st.caption("Atajos: S sonido, C settings, T theme, ? ayuda")

    db_ok = db_path and Path(db_path).exists()
    hints = SchemaHints()
    ticks_df = trades_df = rounds_df = pd.DataFrame()
    notices: List[str] = []
    if db_ok:
        hints, ticks_df, trades_df, rounds_df, notices = read_db(db_path)
    else:
        notices.append("No hay DB disponible. Selecciona una ruta válida para datos reales.")

    ticks_df = parse_time_series(ticks_df, "ts")
    trades_df = parse_time_series(trades_df, "ts") if "ts" in trades_df.columns else trades_df

    total_pnl = float(trades_df["pnl"].fillna(0).sum()) if "pnl" in trades_df.columns else 0.0
    trades_count = int(len(trades_df))
    starting_balance = 10000.0
    equity = starting_balance + total_pnl
    now_et = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d %H:%M:%S %Z")

    hdr1, hdr2 = st.columns([4, 1])
    with hdr1:
        st.markdown(
            f"### Live Visual Terminal  <span class='badge {'ok' if control['auto_trade'] else 'warn'}'>{'AUTO_TRADING_ACTIVE' if control['auto_trade'] else 'PAUSED'}</span>",
            unsafe_allow_html=True,
        )
    with hdr2:
        st.markdown("<span class='dot'></span>CONNECTED", unsafe_allow_html=True)

    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("EQUITY BALANCE", fmt_money(equity))
    s2.metric("SESSION P/L", fmt_money(total_pnl))
    s3.metric("MARKET FOCUS", "BTC/USD 5M")
    s4.metric("TRADES", f"{trades_count:,}")
    s5.metric("ET TIME", now_et)

    for n in notices:
        st.warning(n)

    left, center, right = st.columns([1.45, 1.1, 1.45])

    with left:
        st.markdown("#### BALANCE GROWTH HISTORY")
        if not trades_df.empty and "pnl" in trades_df.columns and "ts" in trades_df.columns:
            g = trades_df.sort_values("ts").copy()
            g["equity"] = starting_balance + g["pnl"].fillna(0).cumsum()
            ch = alt.Chart(g).mark_line().encode(x="ts:T", y="equity:Q")
            st.altair_chart(ch.properties(height=220), use_container_width=True)
        else:
            st.info("Sin datos de equity aún.")

        st.markdown("#### LIVE VOLATILITY FEED")
        if not ticks_df.empty:
            vf = ticks_df.tail(300)
            ch2 = alt.Chart(vf).mark_line().encode(x="ts:T", y="price:Q")
            st.altair_chart(ch2.properties(height=220), use_container_width=True)
            current_price = float(vf["price"].iloc[-1])
        else:
            st.info("Sin ticks disponibles.")
            current_price = 0.0

        ptb = float(rounds_df["price_to_beat"].dropna().iloc[-1]) if not rounds_df.empty else 0.0
        delta = current_price - ptb
        st.markdown("#### PRICE TO BEAT")
        st.progress(0.5 if ptb == 0 else min(1.0, max(0.0, current_price / ptb)))
        st.write(f"PTB: {fmt_money(ptb)} | Current: {fmt_money(current_price)} | Δ: {fmt_money(delta)}")

    with center:
        st.markdown("#### ROUND COUNTDOWN")
        rem, et, utc = compute_countdown()
        st.metric("Next 5m close", f"{rem}s")
        st.write(f"Resolution ET: {et}")
        st.write(f"Resolution UTC: {utc}")

        show_oracle = st.toggle("Show oracle iframe", value=False)
        if show_oracle:
            st.components.v1.html(
                '<iframe src="about:blank" style="width:100%;height:220px;border:1px solid #333;"></iframe>',
                height=230,
            )

        if st.session_state.get("show_help", False):
            with st.expander("Keyboard Shortcuts", expanded=True):
                st.write("S: Sound | C: Settings | T: Theme | ?: Help")

    with right:
        st.markdown("#### EXECUTION_LOG_STREAM")
        log_lines = read_log_tail(LOGS_DIR / "botia.log", lines=30)
        if log_lines:
            st.code("\n".join(log_lines), language="bash")
        elif not trades_df.empty:
            st.write("Fallback: últimas 20 trades")
            st.dataframe(trades_df.head(20), use_container_width=True)
        else:
            st.info("Sin logs ni trades recientes.")

    duration = "N/A"
    if "ts" in trades_df.columns and len(trades_df) > 1:
        duration_td = trades_df["ts"].max() - trades_df["ts"].min()
        duration = str(duration_td).split(".")[0]

    win_rate = 0.0
    avg_pl = best = worst = 0.0
    if "pnl" in trades_df.columns and not trades_df.empty:
        pnl_ser = trades_df["pnl"].fillna(0)
        win_rate = float((pnl_ser > 0).mean() * 100)
        avg_pl = float(pnl_ser.mean())
        best = float(pnl_ser.max())
        worst = float(pnl_ser.min())

    btc_delta = 0.0
    if not ticks_df.empty and len(ticks_df) > 1:
        btc_delta = float(ticks_df["price"].iloc[-1] - ticks_df["price"].iloc[0])

    st.markdown(
        f"""
        <div class='footer'>
          SYSTEM STATUS | WIN RATE: {win_rate:.2f}% | AVG P/L: {fmt_money(avg_pl)} | BEST: {fmt_money(best)} |
          WORST: {fmt_money(worst)} | SESSION DURATION: {duration} | BTC 5m Δ: {fmt_money(btc_delta)} |
          WS: LOCAL | LATENCY: {control['refresh_rate']}s | THEME: TERMINAL
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.get("show_settings", False):
        with st.modal("Settings"):
            st.write("Configura conexión DB y parámetros de paper control desde el sidebar.")
            if st.button("Close settings"):
                st.session_state["show_settings"] = False

    if st.session_state.get("show_theme", False):
        with st.modal("Theme Selector"):
            st.write("Tema activo: Terminal Green.")
            st.button("Terminal Green")
            st.button("Amber Mono")
            if st.button("Close theme"):
                st.session_state["show_theme"] = False

    st.caption(
        f"DB: {db_path if db_ok else 'N/A'} | ticks={hints.ticks_table or '-'} trades={hints.trades_table or '-'} rounds={hints.rounds_table or '-'}"
    )

    refresh_ms = int(control.get("refresh_rate", 2)) * 1000
    components.html(f"<script>setTimeout(() => window.parent.location.reload(), {refresh_ms});</script>", height=0)


if __name__ == "__main__":
    main()
