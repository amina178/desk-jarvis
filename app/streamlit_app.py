"""Desk Jarvis - Streamlit demo.

    streamlit run app/streamlit_app.py

Tab 1 analyses an uploaded video (works everywhere, incl. Hugging Face
Spaces). Tab 2 runs live in the browser through WebRTC (needs camera
permission; on some networks WebRTC is blocked - then use tab 1).
"""
from __future__ import annotations

import tempfile
import threading
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

import _bootstrap  # noqa: F401  (adds src/ to sys.path)
from deskjarvis import config as cfg
from deskjarvis.pipeline import Analyzer, Settings
from deskjarvis.video import analyze_video, blinks_per_minute, summarize

# Categorical slots for the four postures (validated palette, light mode)
POSTURE_COLORS = {"upright": "#2a78d6", "slouch": "#eb6834",
                  "forward_head": "#1baf7a", "side_lean": "#eda100"}
POSTURE_RU = {"upright": "ровно", "slouch": "сутулость",
              "forward_head": "голова вперёд", "side_lean": "наклон вбок"}
GESTURE_RU = {"like": "👍 следующий слайд", "dislike": "👎 предыдущий",
              "palm": "✋ микрофон", "peace": "✌️ скриншот"}

st.set_page_config(page_title="Desk Jarvis", page_icon="🖐", layout="wide")


@st.cache_resource(show_spinner="Downloading MediaPipe models (first run)…")
def ensure_mediapipe_models() -> None:
    """Cloud hosts start from the git repo, which does not contain the
    MediaPipe .task files: fetch them once per server process."""
    import urllib.request

    cfg.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in cfg.MODEL_URLS.items():
        target = cfg.MODELS_DIR / f"{name}.task"
        if not target.exists():
            urllib.request.urlretrieve(url, target)


ensure_mediapipe_models()
st.title("Desk Jarvis")
st.caption("CV-ассистент рабочего места: жесты, осанка и усталость глаз "
           "по обычной веб-камере. Кадры обрабатываются в памяти и не "
           "сохраняются.")

with st.sidebar:
    st.header("Настройки")
    calib = st.slider("Калибровка осанки, с", 2, 10, 5,
                      help="Первые секунды видео человек сидит ровно")
    hold = st.slider("Плохая поза считается после, с", 5, 60, 20)
    max_minutes = st.slider("Анализировать первые N минут", 1, 10, 3)
    settings = Settings(calibration_s=calib, bad_posture_hold_s=hold)
    probe = Analyzer.with_default_models(settings=settings)
    st.markdown(
        f"**Жесты:** {'модель загружена' if probe.gestures else 'нет модели'}"
        f"  \n**Осанка:** {probe.posture_kind}")

tab_video, tab_live = st.tabs(["Анализ видео", "Live (веб-камера)"])

# ------------------------------------------------------------- video


def posture_timeline(df: pd.DataFrame) -> alt.Chart:
    judged = df[df.posture_status == "ok"].copy()
    judged["поза"] = judged.posture.map(POSTURE_RU)
    step = judged.t_s.diff().median()
    judged["t_end"] = judged.t_s + (step if step == step else 0.04)
    judged["track"] = "поза"
    scale = alt.Scale(domain=[POSTURE_RU[k] for k in POSTURE_COLORS],
                      range=list(POSTURE_COLORS.values()))
    return alt.Chart(judged).mark_bar(height=36).encode(
        x=alt.X("t_s:Q", title="время, с"), x2="t_end:Q",
        y=alt.Y("track:N", axis=None),
        color=alt.Color("поза:N", scale=scale,
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=[alt.Tooltip("t_s:Q", title="с", format=".1f"), "поза:N"],
    ).properties(height=135)


def ear_chart(df: pd.DataFrame) -> alt.Chart:
    data = df.dropna(subset=["ear"])
    line = alt.Chart(data).mark_line(strokeWidth=2, color="#2a78d6").encode(
        x=alt.X("t_s:Q", title="время, с"),
        y=alt.Y("ear:Q", title="EAR (открытость глаз)"),
        tooltip=[alt.Tooltip("t_s:Q", format=".2f"),
                 alt.Tooltip("ear:Q", format=".3f")])
    blinks = alt.Chart(data[data.blinked]).mark_rule(
        color="#52514e", strokeDash=[3, 3]).encode(x="t_s:Q")
    return (line + blinks).properties(height=220)


with tab_video:
    st.write("Загрузи короткое видео за рабочим столом (камера спереди, "
             "плечи в кадре; первые секунды — ровная посадка).")
    upload = st.file_uploader("Видео", type=["mp4", "mov", "avi", "webm"])
    if upload and st.button("Анализировать", type="primary"):
        suffix = Path(upload.name).suffix
        with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
            tmp.write(upload.read())
            tmp.flush()
            bar = st.progress(0.0, text="Обработка кадров…")
            df = analyze_video(
                tmp.name, Analyzer.with_default_models(settings=settings),
                max_seconds=max_minutes * 60,
                progress=lambda p: bar.progress(p, text="Обработка кадров…"))
            bar.empty()
        st.session_state["report"] = df

    df = st.session_state.get("report")
    if df is not None and not df.empty:
        s = summarize(df)
        c = st.columns(4)
        c[0].metric("Длительность", f"{s['duration_s']:.0f} с")
        c[1].metric("Морганий в минуту", f"{s['blinks_per_min']:.1f}",
                    help="В покое обычно около 15–20")
        upright = s["posture_shares"].get("upright", 0.0)
        c[2].metric("Время ровной посадки", f"{upright:.0%}")
        c[3].metric("Жестов распознано",
                    sum(s["gesture_events"].values()))

        st.subheader("Осанка во времени")
        if s["posture_judged"] > 0:
            st.altair_chart(posture_timeline(df), width="stretch")
            shares = pd.DataFrame(
                [(POSTURE_RU.get(k, k), f"{v:.0%}")
                 for k, v in s["posture_shares"].items()],
                columns=["поза", "доля времени"])
            st.dataframe(shares, hide_index=True)
        else:
            st.info("Плечи и уши не были видны достаточно чётко — осанку "
                    "оценить нельзя.")
        st.caption(f"Осанка оценена в {s['posture_judged']:.0%} кадров "
                   "(калибровка и кадры без видимых плеч не оцениваются).")

        st.subheader("Глаза")
        if df.ear.notna().any():
            st.altair_chart(ear_chart(df), width="stretch")
            st.caption("Пунктир — зафиксированные моргания.")
            st.dataframe(blinks_per_minute(df), hide_index=True)

        if s["gesture_events"]:
            st.subheader("Жесты")
            events = df.dropna(subset=["event"])[["t_s", "event"]]
            events["действие"] = events.event.map(GESTURE_RU)
            st.dataframe(events, hide_index=True)

        st.download_button("Скачать покадровый отчёт (CSV)",
                           df.to_csv(index=False), "desk_jarvis_report.csv")

# -------------------------------------------------------------- live

with tab_live:
    try:
        import av
        import cv2
        from streamlit_webrtc import webrtc_streamer

        from deskjarvis.landmarks import LandmarkExtractor
    except ImportError:
        st.warning("Для live-режима установи streamlit-webrtc.")
    else:
        lock = threading.Lock()
        state = {"an": None, "ex": None}

        def callback(frame: av.VideoFrame) -> av.VideoFrame:
            img = cv2.flip(frame.to_ndarray(format="bgr24"), 1)
            with lock:  # the callback runs in a worker thread
                if state["an"] is None:
                    state["an"] = Analyzer.with_default_models(
                        settings=settings)
                    state["ex"] = LandmarkExtractor(
                        hand=state["an"].gestures is not None)
                t_ms = int(frame.time * 1000) if frame.time else 0
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                s = state["an"].step(state["ex"].process(rgb, t_ms), rgb)
            lines = [f"Blinks/min {s.blink_rate:.0f}",
                     {"ok": f"Posture: {s.posture}",
                      "calibrating": "Calibrating: sit upright",
                      "not_visible": "Shoulders not visible"}
                     [s.posture_status]]
            if state["an"].gestures is not None:
                lines.insert(0, f"Gesture: {s.gesture}")
            if s.action_text:
                lines.append(s.action_text)
            for k, text in enumerate(lines):
                y = 30 + 30 * k
                cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 0, 0), 4)
                cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (255, 255, 255), 2)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        st.write("Нажми START и разреши доступ к камере. Первые секунды "
                 "сиди ровно — идёт калибровка осанки. Действия жестов "
                 "только показываются на экране.")
        webrtc_streamer(
            key="desk-jarvis", video_frame_callback=callback,
            media_stream_constraints={"video": True, "audio": False},
            rtc_configuration={"iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]}]})
