"""Google Gemini calls for analysis, vision, chat.

The API key is read only from the environment (typically loaded from .env).
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import TypeVar

from google import genai
from google.genai import types
from pydantic import BaseModel, ValidationError

from models.scenario import (
    EmergencyScenario,
    ImageObservations,
    TrafficAnalysis,
)

T = TypeVar("T", bound=BaseModel)

PLACEHOLDER_KEYS = {"", "your_key_here", "changeme", "paste_your_key_here"}

SYSTEM_INSTRUCTION = """You are EmergencyAI, an assistant for analyzing simulated urban emergency traffic scenarios in a hackathon and classroom setting.

Rules you must follow:
- This is a simulation. You do not control traffic signals, dispatch units, or operate vehicles.
- You do not override traffic engineers, emergency responders, or public-safety agencies.
- Phrase guidance as considerations and situational awareness. Never issue automatic control commands or signal timing plans.
- Do not invent or apply proprietary preemption algorithms, research equations, readiness scores, or recovery algorithms.
- Use only the scenario facts, image observations, and conversation provided. If something is unknown, say it needs human verification.
- The ETA is the calculated ETA from the operator-entered distance and speed. Do not replace that calculated ETA with your own travel-time estimate.
- Keep the response concise enough to read during a live demo.
"""


class MissingApiKeyError(Exception):
    """Raised when GEMINI_API_KEY is absent or still the example placeholder."""


class GeminiCallError(Exception):
    """Raised when the Gemini request fails or the SDK is unavailable."""


TEMPORARY_DEMAND_MESSAGE = (
    "Gemini is experiencing temporary high demand. Your scenario has "
    "been preserved. Please try Analyze again."
)

_request_listener = None


def set_request_listener(listener) -> None:
    """App hook incremented once per real generate_content call, including 503 retries."""
    global _request_listener
    _request_listener = listener


def _note_api_request() -> None:
    if _request_listener is not None:
        _request_listener()


@dataclass
class StructuredResult:
    data: BaseModel | None
    raw_text: str
    parsed_ok: bool


def api_key_configured() -> bool:
    return _read_api_key() is not None


def _read_api_key() -> str | None:
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if key.lower() in PLACEHOLDER_KEYS:
        return None
    return key


def _require_api_key() -> str:
    key = _read_api_key()
    if key is None:
        raise MissingApiKeyError(
            "Add your Gemini API key to a local .env file as GEMINI_API_KEY, then restart the app. "
            "Copy .env.example to .env and paste the key from Google AI Studio."
        )
    return key


def model_name() -> str:
    name = os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip()
    return name or "gemini-3.8-flash"


def _redact(message: str) -> str:
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if key:
        message = message.replace(key, "[REDACTED]")
    message = re.sub(r"AIza[0-9A-Za-z\-_]{10,}", "[REDACTED]", message)
    return message.strip()


def _is_model_access_error(lowered: str) -> bool:
    """True only when the API text itself says the model is missing or not allowed."""
    mentions_model = "model" in lowered
    not_found = any(
        token in lowered
        for token in (
            "not found",
            "404",
            "does not exist",
            "unknown model",
            "invalid model",
            "is not supported for generatecontent",
        )
    )
    access_denied = any(
        token in lowered
        for token in (
            "not have access",
            "does not have access",
            "permission denied",
            "caller does not have permission",
        )
    )
    return mentions_model and (not_found or access_denied)


def _friendly_api_error(exc: Exception) -> str:
    text = _redact(str(exc))
    lowered = text.lower()
    detail = f"{exc.__class__.__name__}: {text}" if text else exc.__class__.__name__
    if any(token in lowered for token in ("api key", "api_key", "unauthenticated", "401")):
        return (
            "Gemini rejected the API key. Check GEMINI_API_KEY in .env and restart the app. "
            f"Details: {detail}"
        )
    if _is_model_access_error(lowered):
        return (
            f"Gemini could not use the model '{model_name()}'. "
            "Set GEMINI_MODEL in .env to a model your key can access, then restart. "
            f"Details: {detail}"
        )
    if "deadline" in lowered or "timeout" in lowered or "503" in lowered or "unavailable" in lowered:
        return f"Gemini is temporarily unavailable. Wait a moment and try the request again. Details: {detail}"
    return f"Gemini request failed. {detail}"


def _extract_json(text: str) -> str:
    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        cleaned = fence.group(1).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        return cleaned[start : end + 1]
    return cleaned


def _parse_delay_seconds(value) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        if value > 0:
            return max(1, int(round(float(value))))
        return None
    if isinstance(value, str):
        match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)s?\s*", value, flags=re.IGNORECASE)
        if match:
            number = float(match.group(1))
            if number > 0:
                return max(1, int(round(number)))
    return None


def _find_retry_delay(value, found: list[int]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in {"retrydelay", "retry_delay"}:
                seconds = _parse_delay_seconds(item)
                if seconds is not None:
                    found.append(seconds)
            else:
                _find_retry_delay(item, found)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _find_retry_delay(item, found)


def _retry_delay_seconds(exc: Exception) -> int | None:
    found: list[int] = []
    details = getattr(exc, "details", None)
    if details is not None:
        _find_retry_delay(details, found)
    if found:
        return found[0]
    text = str(exc)
    match = re.search(r"retryDelay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)s", text, flags=re.IGNORECASE)
    if not match:
        match = re.search(r"retry in\s+(\d+(?:\.\d+)?)\s*s", text, flags=re.IGNORECASE)
    if match:
        return max(1, int(round(float(match.group(1)))))
    return None


def _is_429(exc: Exception) -> bool:
    if getattr(exc, "code", None) == 429:
        return True
    status = str(getattr(exc, "status", "") or "").upper()
    if status == "RESOURCE_EXHAUSTED":
        return True
    text = str(exc)
    return "RESOURCE_EXHAUSTED" in text.upper() or re.search(r"\b429\b", text) is not None


def quota_message(exc: Exception) -> str:
    seconds = _retry_delay_seconds(exc)
    if seconds is None:
        wait = "Please wait and try again."
    elif seconds == 1:
        wait = "Please wait about 1 second and try again."
    else:
        wait = f"Please wait about {seconds} seconds and try again."
    return (
        "Gemini API quota is temporarily limited. "
        f"{wait} Your scenario and image analysis have been preserved."
    )


def _is_503(exc: Exception) -> bool:
    if _is_429(exc):
        return False
    code = getattr(exc, "code", None)
    if code == 503:
        return True
    return re.search(r"\b503\b", str(exc)) is not None


def structured_to_cache(result: StructuredResult) -> dict:
    data = None
    if result.parsed_ok and result.data is not None:
        data = result.data.model_dump()
    return {"parsed_ok": data is not None, "data": data, "raw_text": result.raw_text}


def structured_from_cache(entry: dict, schema: type[T]) -> StructuredResult:
    data = None
    if entry.get("parsed_ok") and entry.get("data") is not None:
        data = schema.model_validate(entry["data"])
    return StructuredResult(
        data=data,
        raw_text=entry.get("raw_text") or "",
        parsed_ok=data is not None,
    )


def _generate(contents, schema: type[BaseModel] | None) -> object:
    _require_api_key()
    client = genai.Client(api_key=_require_api_key())
    config_kwargs: dict = {
        "system_instruction": SYSTEM_INSTRUCTION,
        "temperature": 0.4,
    }
    if schema is not None:
        config_kwargs["response_mime_type"] = "application/json"
        config_kwargs["response_schema"] = schema
    request = {
        "model": model_name(),
        "contents": contents,
        "config": types.GenerateContentConfig(**config_kwargs),
    }
    # Four attempts. After a 503, wait 1s, then 2s, then 4s, and reuse this same request.
    retry_waits = (1, 2, 4)
    for attempt in range(4):
        try:
            _note_api_request()
            return client.models.generate_content(**request)
        except Exception as exc:
            if _is_429(exc):
                raise GeminiCallError(quota_message(exc)) from exc
            if _is_503(exc) and attempt < 3:
                time.sleep(retry_waits[attempt])
                continue
            if _is_503(exc):
                raise GeminiCallError(TEMPORARY_DEMAND_MESSAGE) from exc
            raise GeminiCallError(_friendly_api_error(exc)) from exc
    raise GeminiCallError(TEMPORARY_DEMAND_MESSAGE)


def _as_model(response: object, schema: type[T]) -> StructuredResult:
    raw_text = getattr(response, "text", None) or ""
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, schema):
        return StructuredResult(data=parsed, raw_text=raw_text, parsed_ok=True)
    if isinstance(parsed, dict):
        try:
            return StructuredResult(data=schema.model_validate(parsed), raw_text=raw_text, parsed_ok=True)
        except ValidationError:
            pass
    try:
        data = schema.model_validate_json(_extract_json(raw_text))
        return StructuredResult(data=data, raw_text=raw_text, parsed_ok=True)
    except (ValidationError, json.JSONDecodeError, ValueError):
        return StructuredResult(data=None, raw_text=raw_text or "Gemini returned an empty response.", parsed_ok=False)


def _history_block(chat_history: list[dict] | None) -> str:
    if not chat_history:
        return "No earlier questions in this session."
    lines = ["Earlier questions in this session:"]
    for turn in chat_history[-8:]:
        role = "Operator" if turn.get("role") == "user" else "EmergencyAI"
        lines.append(f"{role}: {turn.get('content', '')}")
    return "\n".join(lines)


def _analysis_block(analysis: dict | None, raw_text: str | None) -> str:
    if analysis:
        return "Previous Gemini analysis (JSON):\n" + json.dumps(analysis, indent=2)
    if raw_text:
        return "Previous Gemini analysis could not be structured. Raw text:\n" + raw_text
    return "No Gemini analysis has been generated yet for this scenario."


def analyze_scenario(scenario: EmergencyScenario) -> StructuredResult:
    prompt = "\n".join(
        [
            "Analyze this simulated emergency traffic scenario.",
            scenario.prompt_block(),
            "",
            "Return structured JSON with:",
            "- situation_summary",
            "- traffic_level",
            "- key_concerns",
            "- responder_considerations",
            "- weather_considerations (include road-condition effects here)",
            "- operator_considerations",
            "- verification_needed",
            "Keep every item a consideration for a human. Do not prescribe signal changes.",
        ]
    )
    response = _generate(prompt, TrafficAnalysis)
    return _as_model(response, TrafficAnalysis)


def ask_question(
    scenario: EmergencyScenario,
    question: str,
    analysis: dict | None,
    analysis_raw: str | None,
    chat_history: list[dict] | None,
    image_notes: str | None = None,
) -> str:
    prompt = "\n".join(
        [
            "Answer the operator's question about the current simulated scenario.",
            "Be concise. Separate what comes from the entered scenario, from an image, and from uncertainty.",
            "",
            scenario.prompt_block(),
            "",
            _analysis_block(analysis, analysis_raw),
            "",
            "Image observations:" if image_notes else "Image observations: none in this session.",
            image_notes or "",
            "",
            _history_block(chat_history),
            "",
            f"Operator question: {question}",
        ]
    )
    response = _generate(prompt, None)
    text = (getattr(response, "text", None) or "").strip()
    if not text:
        raise GeminiCallError("Gemini returned an empty answer. Try asking the question again.")
    return text


def analyze_image(image_bytes: bytes, mime_type: str) -> StructuredResult:
    allowed = {"image/jpeg", "image/jpg", "image/png"}
    if mime_type.lower() not in allowed:
        raise GeminiCallError("Upload a JPG, JPEG, or PNG image.")
    if mime_type.lower() == "image/jpg":
        mime_type = "image/jpeg"
    prompt = "\n".join(
        [
            "Look at this intersection image and describe only what is reasonably observable.",
            "Cover approximate traffic density, congestion, road conditions, weather or visibility,",
            "pedestrians, clearly identifiable emergency vehicles, possible obstructions, and other relevant traffic observations.",
            "Express uncertainty. If a detail is not visible, say it is not clear from the image.",
            "Do not invent vehicle counts, speeds, identities, or events that cannot be seen.",
            "Return JSON with traffic_density, congestion, road_conditions, weather_visibility,",
            "pedestrians, emergency_vehicles, obstructions, other_observations, and uncertainty_notes.",
        ]
    )
    image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
    response = _generate([prompt, image_part], ImageObservations)
    return _as_model(response, ImageObservations)


def _serialize_image_observations(image_observations: dict | None, image_raw: str | None) -> str:
    """Turn stored image notes into plain text. This path never attaches image bytes."""
    if image_observations:
        payload = image_observations.model_dump() if hasattr(image_observations, "model_dump") else image_observations
        try:
            return json.dumps(payload, indent=2, ensure_ascii=False, default=str)
        except TypeError:
            return str(payload)
    if image_raw and str(image_raw).strip():
        return str(image_raw).strip()
    raise GeminiCallError("Analyze an image first, then combine it with the scenario.")


def analyze_with_image(
    scenario: EmergencyScenario,
    image_observations: dict | None,
    image_raw: str | None,
) -> StructuredResult:
    observations_text = _serialize_image_observations(image_observations, image_raw)
    prompt = "\n".join(
        [
            "Analyze this simulated emergency traffic scenario.",
            "Use the operator-entered scenario together with text observations that were already extracted from an image.",
            "The image is not attached. Do not ask for image bytes.",
            "Keep operator-entered facts and image-derived observations distinguishable in situation_summary.",
            "If they conflict, say so in verification_needed.",
            scenario.prompt_block(),
            "",
            "Text observations already extracted from the uploaded image:",
            observations_text,
            "",
            "Return structured JSON with:",
            "- situation_summary",
            "- traffic_level",
            "- key_concerns",
            "- responder_considerations",
            "- weather_considerations (include road-condition effects here)",
            "- operator_considerations",
            "- verification_needed",
            "Keep every item a consideration for a human. Do not prescribe signal changes.",
        ]
    )
    response = _generate(prompt, TrafficAnalysis)
    return _as_model(response, TrafficAnalysis)
