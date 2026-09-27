"""EmergencyAI — one-page Streamlit dashboard for simulated emergency traffic analysis."""

from __future__ import annotations

import hashlib
import io
import json
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image, UnidentifiedImageError

load_dotenv(Path(__file__).resolve().parent / ".env")

import streamlit as st

from models.scenario import (
    EmergencyScenario,
    ImageObservations,
    TrafficAnalysis,
    format_eta,
)
from services.gemini_service import (
    GeminiCallError,
    MissingApiKeyError,
    analyze_image,
    analyze_scenario,
    analyze_with_image,
    api_key_configured,
    ask_question,
    model_name,
    set_request_listener,
    structured_from_cache,
    structured_to_cache,
    TEMPORARY_DEMAND_MESSAGE,
)
from services.report_service import build_report
from utils.scenarios import (
    APPROACHES,
    DEFAULT_SCENARIO,
    EXAMPLE_QUESTIONS,
    INCIDENT_OPTIONS,
    PRESETS,
    ROAD_OPTIONS,
    SIGNAL_OPTIONS,
    VEHICLES,
    WEATHER_OPTIONS,
)

st.set_page_config(
    page_title="EmergencyAI",
    page_icon="🚑",
    layout="wide",
    initial_sidebar_state="collapsed",
)

VEHICLE_ICON = {"Ambulance": "🚑", "Fire Truck": "🚒", "Police": "🚓"}


def init_state() -> None:
    st.session_state.setdefault("gemini_request_count", 0)
    st.session_state.setdefault("analysis_cache", {})
    st.session_state.setdefault("image_cache", {})
    st.session_state.setdefault("combined_cache", {})
    st.session_state.setdefault("question_cache", {})
    if st.session_state.get("_ready"):
        st.session_state.setdefault("analysis_request_number", 0)
        st.session_state.setdefault("analyzed_signature", None)
        return
    for key, value in DEFAULT_SCENARIO.items():
        st.session_state[key] = value
    st.session_state.chat_history = []
    st.session_state.event_log = []
    st.session_state.analysis = None
    st.session_state.analysis_raw = None
    st.session_state.analysis_source = None
    st.session_state.image_observations = None
    st.session_state.image_raw = None
    st.session_state.report_text = None
    st.session_state.analysis_request_number = 0
    st.session_state.analyzed_signature = None
    st.session_state.gemini_request_count = 0
    st.session_state.analysis_cache = {}
    st.session_state.image_cache = {}
    st.session_state.combined_cache = {}
    st.session_state.question_cache = {}
    st.session_state._ready = True
    log_event("Session started")


def log_event(message: str) -> None:
    st.session_state.event_log.append(
        {"time": datetime.now().strftime("%H:%M:%S"), "message": message}
    )


SCENARIO_FIELDS = (
    "emergency_vehicle",
    "approach",
    "distance_meters",
    "speed_mph",
    "north_traffic",
    "south_traffic",
    "east_traffic",
    "west_traffic",
    "weather",
    "road_condition",
    "incident",
    "current_signal",
)


def scenario_signature() -> tuple:
    return tuple(st.session_state[key] for key in SCENARIO_FIELDS)


def inputs_changed_since_analysis() -> bool:
    if not (st.session_state.analysis or st.session_state.analysis_raw):
        return False
    saved = st.session_state.get("analyzed_signature")
    if saved is None:
        return False
    return saved != scenario_signature()


def format_analysis_debug(scenario: EmergencyScenario, request_number: int) -> str:
    eta = scenario.eta_seconds()
    if eta is None:
        eta_text = format_eta(None)
    else:
        eta_text = f"{format_eta(eta)} ({eta:.1f} seconds)"
    counts = scenario.traffic_by_approach()
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return "\n".join(
        [
            "Current scenario sent to Gemini:",
            f"- timestamp: {stamp}",
            f"- request number: {request_number}",
            f"- vehicle: {scenario.emergency_vehicle}",
            f"- approach: {scenario.approach}",
            f"- distance: {scenario.distance_meters:.0f} meters",
            f"- speed: {scenario.speed_mph:.0f} mph",
            f"- calculated ETA: {eta_text}",
            f"- north traffic: {counts['North']}",
            f"- south traffic: {counts['South']}",
            f"- east traffic: {counts['East']}",
            f"- west traffic: {counts['West']}",
            f"- weather: {scenario.weather}",
            f"- road condition: {scenario.road_condition}",
            f"- incident: {scenario.incident}",
            f"- current signal: {scenario.current_signal}",
            f"- GEMINI_MODEL: {model_name()}",
        ]
    )


def current_scenario() -> EmergencyScenario:
    return EmergencyScenario(
        emergency_vehicle=st.session_state.emergency_vehicle,
        approach=st.session_state.approach,
        distance_meters=float(st.session_state.distance_meters),
        speed_mph=float(st.session_state.speed_mph),
        north_traffic=int(st.session_state.north_traffic),
        south_traffic=int(st.session_state.south_traffic),
        east_traffic=int(st.session_state.east_traffic),
        west_traffic=int(st.session_state.west_traffic),
        weather=st.session_state.weather,
        road_condition=st.session_state.road_condition,
        incident=st.session_state.incident,
        current_signal=st.session_state.current_signal,
    )


def apply_preset(preset: dict) -> None:
    for key, value in preset["values"].items():
        st.session_state[key] = value
    st.session_state.analysis = None
    st.session_state.analysis_raw = None
    st.session_state.analysis_source = None
    st.session_state.report_text = None
    st.session_state.analyzed_signature = None
    st.session_state.chat_history = []
    log_event(preset["log"])
    st.rerun()


def show_gemini_error(exc: Exception) -> None:
    message = str(exc)
    preserved = message == TEMPORARY_DEMAND_MESSAGE or message.startswith(
        "Gemini API quota is temporarily limited."
    )
    if isinstance(exc, MissingApiKeyError) or preserved:
        st.warning(message)
    else:
        st.error(message)


def _stable_key(payload) -> str:
    return json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False)


def scenario_cache_key(scenario: EmergencyScenario) -> str:
    return _stable_key(scenario.model_dump())


def combined_cache_key(scenario: EmergencyScenario, observations, raw_text) -> str:
    return _stable_key(
        {
            "scenario": scenario.model_dump(),
            "observations": observations,
            "raw_text": raw_text,
        }
    )


def question_cache_key(scenario: EmergencyScenario, question: str, analysis, analysis_raw, image_notes) -> str:
    return _stable_key(
        {
            "scenario": scenario.model_dump(),
            "question": question,
            "analysis": analysis,
            "analysis_raw": analysis_raw,
            "image_notes": image_notes,
        }
    )


def _record_gemini_request() -> None:
    st.session_state.gemini_request_count = int(st.session_state.get("gemini_request_count", 0)) + 1


def apply_image_result(result) -> None:
    if result.parsed_ok and result.data is not None:
        st.session_state.image_observations = result.data.model_dump()
        st.session_state.image_raw = None
    else:
        st.session_state.image_observations = None
        st.session_state.image_raw = result.raw_text


def store_analysis(result, source: str) -> None:
    if result.parsed_ok and result.data is not None:
        st.session_state.analysis = result.data.model_dump()
        st.session_state.analysis_raw = None
    else:
        st.session_state.analysis = None
        st.session_state.analysis_raw = result.raw_text
    st.session_state.analysis_source = source
    st.session_state.analyzed_signature = scenario_signature()


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
            background: #ffffff;
        }
        .ea-hero {
            background: linear-gradient(90deg, #dc2626 0%, #eab308 50%, #16a34a 100%);
            color: #ffffff;
            border-radius: 16px;
            padding: 1.15rem 1.6rem 1.15rem 1.6rem;
            margin-bottom: 0.8rem;
            text-align: center;
        }
        .ea-hero h1, .ea-hero p, .ea-note {
            text-align: center;
            color: #ffffff;
            text-shadow: 0 1px 2px rgba(0, 0, 0, 0.35);
        }
        .ea-hero h1 { margin: 0.35rem 0 0 0; font-size: 2.1rem; letter-spacing: -0.03em; }
        .ea-hero p { margin: 0.35rem 0 0 0; font-size: 1.02rem; }
        .ea-note { font-size: 0.92rem; margin-top: 0.55rem; }
        .ea-signal {
            display: block;
            margin: 0 auto;
        }
        .ea-board {
            background: #1e293b;
            border-radius: 16px;
            padding: 0.9rem;
            color: #f8fafc;
        }
        .ea-grid {
            display: grid;
            grid-template-columns: 1fr 168px 1fr;
            grid-template-rows: auto 150px auto;
            gap: 8px;
            align-items: stretch;
        }
        .ea-arm, .ea-center {
            border-radius: 12px;
            padding: 0.55rem 0.7rem;
        }
        .ea-arm { background: #334155; min-height: 78px; }
        .ea-arm strong { display: block; font-size: 0.95rem; }
        .ea-meta { color: #cbd5e1; font-size: 0.82rem; margin-top: 0.2rem; }
        .ea-center {
            background: #0f172a;
            border: 2px solid #f59e0b;
            display: flex;
            flex-direction: column;
            justify-content: center;
            text-align: center;
        }
        .ea-light {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 6px;
        }
        .ea-green { background: #22c55e; box-shadow: 0 0 8px #22c55e; }
        .ea-red { background: #ef4444; }
        .ea-ev { margin-top: 0.35rem; color: #fde68a; font-weight: 600; }
        div[data-testid="stVerticalBlockBorderWrapper"],
        [data-testid="stLayoutWrapper"] > [data-testid="stVerticalBlock"] {
            border: 1.5px solid #94a3b8 !important;
            border-radius: 14px !important;
            background: #F7F4EF !important;
            margin-bottom: 0.85rem;
            padding-top: 0.35rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def signal_class(direction: str, phase: str) -> str:
    north_south = phase == "North-South Green"
    if direction in {"North", "South"}:
        return "ea-green" if north_south else "ea-red"
    return "ea-red" if north_south else "ea-green"


def signal_word(direction: str, phase: str) -> str:
    return "Green" if signal_class(direction, phase) == "ea-green" else "Red"


def arm_html(direction: str, count: int, scenario: EmergencyScenario) -> str:
    light = signal_class(direction, scenario.current_signal)
    word = signal_word(direction, scenario.current_signal)
    ev = ""
    if scenario.approach == direction:
        icon = VEHICLE_ICON.get(scenario.emergency_vehicle, "🚨")
        ev = (
            f"<div class='ea-ev'>{icon} {scenario.emergency_vehicle} "
            f"· {scenario.distance_meters:.0f} m</div>"
        )
    return (
        f"<div class='ea-arm'><strong>{direction}</strong>"
        f"<div class='ea-meta'><span class='ea-light {light}'></span>"
        f"{word} · {count} vehicles</div>{ev}</div>"
    )


def render_intersection(scenario: EmergencyScenario) -> None:
    counts = scenario.traffic_by_approach()
    empty = "<div></div>"
    html = (
        "<div class='ea-board'><div class='ea-grid'>"
        f"{empty}{arm_html('North', counts['North'], scenario)}{empty}"
        f"{arm_html('West', counts['West'], scenario)}"
        f"<div class='ea-center'><strong>Intersection</strong>"
        f"<div class='ea-meta'>{scenario.current_signal}</div></div>"
        f"{arm_html('East', counts['East'], scenario)}"
        f"{empty}{arm_html('South', counts['South'], scenario)}{empty}"
        "</div></div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def render_bullets(items: list[str] | None) -> None:
    if not items:
        st.caption("None listed.")
        return
    for item in items:
        st.markdown(f"- {item}")


def render_analysis_body(analysis: dict) -> None:
    st.metric("Traffic level", analysis.get("traffic_level", "—"))
    st.markdown(analysis.get("situation_summary", ""))
    left, right = st.columns(2)
    with left:
        st.markdown("**Key concerns**")
        render_bullets(analysis.get("key_concerns"))
        st.markdown("**Responder considerations**")
        render_bullets(analysis.get("responder_considerations"))
        st.markdown("**Weather and road considerations**")
        render_bullets(analysis.get("weather_considerations"))
    with right:
        st.markdown("**Operator considerations**")
        render_bullets(analysis.get("operator_considerations"))
        st.markdown("**Needs human verification**")
        render_bullets(analysis.get("verification_needed"))


def render_raw_fallback(raw_text: str | None) -> None:
    st.info(
        "Gemini replied, but the response did not match the expected structure. "
        "The readable text is shown below."
    )
    st.write(raw_text or "The response was empty.")


def image_notes_for_prompt() -> str | None:
    if st.session_state.image_observations:
        obs = st.session_state.image_observations
        return "\n".join(f"- {key}: {value}" for key, value in obs.items())
    return st.session_state.image_raw


def read_upload(uploaded) -> tuple[bytes, str]:
    if uploaded is None:
        raise GeminiCallError("Choose a JPG, JPEG, or PNG image first.")
    mime = (uploaded.type or "").lower()
    if mime not in {"image/jpeg", "image/jpg", "image/png"}:
        raise GeminiCallError("That file type is not supported. Upload a JPG, JPEG, or PNG image.")
    raw = uploaded.getvalue()
    if not raw:
        raise GeminiCallError("The uploaded file is empty. Choose another image.")
    try:
        with Image.open(io.BytesIO(raw)) as img:
            img.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise GeminiCallError("That file could not be read as an image. Upload a JPG, JPEG, or PNG.") from exc
    return raw, "image/jpeg" if mime == "image/jpg" else mime


def render_header() -> None:
    key_state = "Gemini key detected" if api_key_configured() else "Add GEMINI_API_KEY to .env before analysis"
    st.markdown(
        f"""
        <div class="ea-hero">
          <svg class="ea-signal" width="46" height="78" viewBox="0 0 46 78" aria-hidden="true">
            <rect x="6" y="1" width="34" height="76" rx="10" fill="#111827"/>
            <circle cx="23" cy="18" r="9" fill="#ef4444"/>
            <circle cx="23" cy="39" r="9" fill="#facc15"/>
            <circle cx="23" cy="60" r="9" fill="#22c55e"/>
          </svg>
          <h1>EmergencyAI</h1>
          <p>Gemini Powered Emergency Traffic Scenario Assistant</p>
          <div class="ea-note">
            AI-powered analysis of simulated emergency traffic situations. Combines scenario data and image observations to support human decision-making. It does not control traffic signals.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_quick_demo() -> EmergencyScenario:
    with st.container(border=True):
        st.markdown("### Quick Demo")
        scenario_cols = st.columns(3)
        for column, preset, name in zip(scenario_cols, PRESETS, ("Scenario 1", "Scenario 2", "Scenario 3")):
            with column:
                if st.button(name, key=f"preset_{preset['id']}", type="primary", use_container_width=True, help=preset["blurb"]):
                    apply_preset(preset)
                st.caption(preset["label"])
        left, right = st.columns([1.05, 1], gap="large")
        with left:
            vehicle_col, approach_col = st.columns(2)
            with vehicle_col:
                st.selectbox("Emergency vehicle", VEHICLES, key="emergency_vehicle")
            with approach_col:
                st.selectbox("Approach", APPROACHES, key="approach")
            st.markdown("**Environmental condition**")
            distance_col, speed_col = st.columns(2)
            with distance_col:
                st.slider("Distance (meters)", 50, 2000, step=10, key="distance_meters")
            with speed_col:
                st.slider("Speed (mph)", 0, 100, step=1, key="speed_mph")
            st.markdown("**Environment**")
            weather_col, road_col, incident_col = st.columns(3)
            with weather_col:
                st.selectbox("Weather", WEATHER_OPTIONS, key="weather")
            with road_col:
                st.selectbox("Road condition", ROAD_OPTIONS, key="road_condition")
            with incident_col:
                st.selectbox("Incident", INCIDENT_OPTIONS, key="incident")
            scenario = current_scenario()
            eta = scenario.eta_seconds()
            eta_col, approach_metric, signal_metric = st.columns(3)
            eta_col.metric("Calculated ETA", format_eta(eta))
            approach_metric.metric(
                "Approach",
                f"{VEHICLE_ICON.get(scenario.emergency_vehicle, '')} {scenario.approach}",
            )
            signal_metric.metric("Signal", scenario.current_signal)
            render_analyze_button()
        with right:
            render_intersection(current_scenario())
            st.markdown("**Traffic count**")
            north_col, south_col = st.columns(2)
            with north_col:
                st.slider("North", 0, 120, key="north_traffic")
            with south_col:
                st.slider("South", 0, 120, key="south_traffic")
            east_col, west_col = st.columns(2)
            with east_col:
                st.slider("East", 0, 120, key="east_traffic")
            with west_col:
                st.slider("West", 0, 120, key="west_traffic")
            st.selectbox("Current signal", SIGNAL_OPTIONS, key="current_signal")
        render_analyze_output()
    return current_scenario()


def render_analyze_button() -> None:
    if st.button("Analyze with Gemini", type="primary", use_container_width=True):
        scenario = current_scenario()
        st.session_state.analysis_request_number += 1
        request_number = st.session_state.analysis_request_number
        cache_key = scenario_cache_key(scenario)
        cached = st.session_state.analysis_cache.get(cache_key)
        debug_text = format_analysis_debug(scenario, request_number)
        if cached is not None:
            debug_text += "\n- Gemini request: reused session cache"
        st.code(debug_text, language="text")
        log_event(
            f"Analysis request #{request_number}: {scenario.emergency_vehicle} from {scenario.approach}, "
            f"{scenario.distance_meters:.0f} m at {scenario.speed_mph:.0f} mph, model {model_name()}"
        )
        if cached is not None:
            store_analysis(structured_from_cache(cached, TrafficAnalysis), "Scenario inputs")
            log_event("Reused cached scenario analysis")
        else:
            with st.spinner("Gemini is reading the scenario..."):
                try:
                    result = analyze_scenario(scenario)
                except (MissingApiKeyError, GeminiCallError) as exc:
                    show_gemini_error(exc)
                    log_event("Analysis failed")
                else:
                    st.session_state.analysis_cache[cache_key] = structured_to_cache(result)
                    store_analysis(result, "Scenario inputs")
                    log_event("Analysis completed")
                    if not result.parsed_ok:
                        log_event("Analysis response stored as readable text")


def render_analyze_output() -> None:
    if inputs_changed_since_analysis():
        st.warning("Scenario inputs have changed. Run Analyze with Gemini to refresh the AI analysis.")

    st.markdown("**Output**")
    with st.container(border=True):
        if st.session_state.analysis or st.session_state.analysis_raw:
            source = st.session_state.analysis_source or "Scenario inputs"
            st.caption(f"Gemini analysis · {source}")
            if st.session_state.analysis:
                render_analysis_body(st.session_state.analysis)
            else:
                render_raw_fallback(st.session_state.analysis_raw)
        else:
            st.info("Run Analyze with Gemini to see the situation summary, concerns, and considerations.")


def render_vision(scenario: EmergencyScenario) -> None:
    with st.container(border=True):
        st.markdown("### Gemini Vision")
        upload_col, action_col = st.columns([1.15, 1], gap="large")
        with upload_col:
            uploaded = st.file_uploader(
                "Upload traffic scene image (JPG, JPEG, or PNG)",
                type=["jpg", "jpeg", "png"],
            )
            if uploaded is not None:
                st.image(uploaded, caption="Uploaded intersection image", use_container_width=True)
        with action_col:
            analyze_clicked = st.button("Analyze image", type="primary", use_container_width=True)
            combine_clicked = st.button("Use image observations in analysis", type="primary", use_container_width=True)

        if analyze_clicked:
            try:
                raw, mime = read_upload(uploaded)
            except GeminiCallError as exc:
                st.error(str(exc))
            else:
                log_event("Image analysis requested")
                digest = hashlib.sha256(raw).hexdigest()
                cached = st.session_state.image_cache.get(digest)
                if cached is not None:
                    apply_image_result(structured_from_cache(cached, ImageObservations))
                    log_event("Reused cached image analysis")
                else:
                    with st.spinner("Gemini is looking at the image..."):
                        try:
                            result = analyze_image(raw, mime)
                        except (MissingApiKeyError, GeminiCallError) as exc:
                            show_gemini_error(exc)
                            log_event("Image analysis failed")
                        else:
                            st.session_state.image_cache[digest] = structured_to_cache(result)
                            apply_image_result(result)
                            log_event("Image analyzed")

        if combine_clicked:
            if not st.session_state.image_observations and not st.session_state.image_raw:
                st.warning("Analyze an image first. Then combine those observations with the scenario.")
            else:
                scenario_now = current_scenario()
                log_event("Combined scenario and image analysis requested")
                cache_key = combined_cache_key(
                    scenario_now,
                    st.session_state.image_observations,
                    st.session_state.image_raw,
                )
                cached = st.session_state.combined_cache.get(cache_key)
                if cached is not None:
                    store_analysis(
                        structured_from_cache(cached, TrafficAnalysis),
                        "Scenario inputs + image observations",
                    )
                    log_event("Reused cached combined analysis")
                    st.rerun()
                else:
                    with st.spinner("Gemini is combining the scenario with the image notes..."):
                        try:
                            result = analyze_with_image(
                                scenario_now,
                                st.session_state.image_observations,
                                st.session_state.image_raw,
                            )
                        except (MissingApiKeyError, GeminiCallError) as exc:
                            show_gemini_error(exc)
                            log_event("Combined analysis failed")
                        else:
                            if result.parsed_ok and isinstance(result.data, TrafficAnalysis):
                                st.session_state.combined_cache[cache_key] = structured_to_cache(result)
                                store_analysis(result, "Scenario inputs + image observations")
                                log_event("Combined analysis completed")
                            else:
                                st.session_state.analysis = None
                                st.session_state.analysis_raw = result.raw_text
                                st.session_state.analysis_source = "Scenario inputs + image observations"
                                log_event("Combined analysis stored as readable text")
                            st.rerun()

        output_left, output_right = st.columns(2, gap="medium")
        with output_left:
            st.markdown("**Output 1**")
            with st.container(border=True):
                st.caption("Gemini-generated observations from the uploaded image. These observations do not replace the operator-entered scenario.")
                if st.session_state.image_observations or st.session_state.image_raw:
                    obs = st.session_state.image_observations
                    if obs:
                        pairs = [
                            ("Traffic density", "traffic_density"),
                            ("Congestion", "congestion"),
                            ("Road conditions", "road_conditions"),
                            ("Weather / visibility", "weather_visibility"),
                            ("Pedestrians", "pedestrians"),
                            ("Emergency vehicles", "emergency_vehicles"),
                            ("Obstructions", "obstructions"),
                        ]
                        for label, key in pairs:
                            st.markdown(f"**{label}:** {obs.get(key, '')}")
                        st.markdown("**Other observations**")
                        render_bullets(obs.get("other_observations"))
                        st.markdown("**Uncertainty**")
                        render_bullets(obs.get("uncertainty_notes"))
                    else:
                        render_raw_fallback(st.session_state.image_raw)
                else:
                    st.info("Upload and analyze an image to view Gemini’s traffic observations here.")
        with output_right:
            st.markdown("**Output 2**")
            with st.container(border=True):
                st.caption("Combined analysis using the current scenario and Gemini image observations.")
                combined = st.session_state.get("analysis_source") == "Scenario inputs + image observations"
                if combined and (st.session_state.analysis or st.session_state.analysis_raw):
                    if st.session_state.analysis:
                        render_analysis_body(st.session_state.analysis)
                    else:
                        render_raw_fallback(st.session_state.analysis_raw)
                else:
                    st.info("Use image observations in analysis to see the combined result here.")


def render_chat(scenario: EmergencyScenario) -> None:
    with st.container(border=True):
        st.markdown("### Ask EmergencyAI")
        st.caption("Each submitted question uses the current scenario, the latest analysis, image observations when available, and this session's chat.")
        question_cols = st.columns(2)
        for column, example in zip(question_cols, EXAMPLE_QUESTIONS):
            with column:
                if st.button(example, key=f"q_{example}", type="primary", use_container_width=True):
                    _ask(scenario, example)

        question = st.text_input("Your question", key="question_text", placeholder="Ask about this simulated scenario")
        _, ask_col, _ = st.columns([1.4, 1, 1.4])
        with ask_col:
            asked = st.button("Ask", type="primary", use_container_width=True)
        if asked:
            _ask(scenario, question)

        st.markdown("**Output**")
        with st.container(border=True):
            if st.session_state.chat_history:
                for turn in st.session_state.chat_history:
                    role = "user" if turn.get("role") == "user" else "assistant"
                    with st.chat_message(role):
                        st.markdown(turn.get("content", ""))
            else:
                st.info("Ask a question after you set a scenario. Example prompts are above.")


def _ask(scenario: EmergencyScenario, question: str) -> None:
    cleaned = (question or "").strip()
    if not cleaned:
        st.warning("Type a question first. An empty question cannot be sent.")
        return
    scenario = current_scenario()
    image_notes = image_notes_for_prompt()
    cache_key = question_cache_key(
        scenario,
        cleaned,
        st.session_state.analysis,
        st.session_state.analysis_raw,
        image_notes,
    )
    cached = st.session_state.question_cache.get(cache_key)
    if cached is not None:
        answer = cached
        log_event("Reused cached answer")
    else:
        log_event("Contextual question asked")
        with st.spinner("Gemini is answering..."):
            try:
                answer = ask_question(
                    scenario,
                    cleaned,
                    st.session_state.analysis,
                    st.session_state.analysis_raw,
                    st.session_state.chat_history,
                    image_notes,
                )
            except (MissingApiKeyError, GeminiCallError) as exc:
                show_gemini_error(exc)
                log_event("Contextual question failed")
                return
        st.session_state.question_cache[cache_key] = answer
        log_event("Contextual answer received")
    st.session_state.chat_history.append({"role": "user", "content": cleaned})
    st.session_state.chat_history.append({"role": "assistant", "content": answer})


def render_report(scenario: EmergencyScenario) -> None:
    with st.container(border=True):
        st.markdown("### Incident report")
        st.caption("Builds a .txt summary in Python from the current scenario and analysis. This step does not call Gemini.")
        _, report_col, _ = st.columns([1.4, 1, 1.4])
        with report_col:
            create_report = st.button("Create Report", type="primary", use_container_width=True)
        if create_report:
            scenario = current_scenario()
            st.session_state.report_text = build_report(
                scenario,
                analysis=st.session_state.analysis,
                analysis_raw=st.session_state.analysis_raw,
                image_observations=st.session_state.image_observations,
                image_raw=st.session_state.image_raw,
                analysis_source=st.session_state.analysis_source,
            )
            log_event("Incident report generated")

        if st.session_state.report_text:
            st.download_button(
                "Download .txt report",
                type="primary",
                data=st.session_state.report_text,
                file_name="emergencyai_incident_summary.txt",
                mime="text/plain",
                use_container_width=True,
            )
            with st.expander("Preview report", expanded=False):
                st.text(st.session_state.report_text)


def render_timeline() -> None:
    """Kept for later. Not shown on the page right now."""
    with st.container(border=True):
        st.markdown("### Session timeline")
        if not st.session_state.event_log:
            st.caption("Events will appear here as you load scenarios and call Gemini.")
            return
        for event in st.session_state.event_log:
            st.markdown(f"`{event['time']}` {event['message']}")


def main() -> None:
    init_state()
    set_request_listener(_record_gemini_request)
    inject_css()
    render_header()
    scenario = render_quick_demo()
    render_vision(scenario)
    render_chat(scenario)
    render_report(scenario)
    st.divider()
    st.caption(
        "EmergencyAI."
    )


main()
