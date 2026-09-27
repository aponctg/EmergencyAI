"""Plain-text incident summaries built locally from scenario data and prior analysis."""

from models.scenario import EmergencyScenario


def _bullets(items: list[str] | None) -> list[str]:
    if not items:
        return ["- None recorded."]
    return [f"- {item}" for item in items]


def _eta_line(scenario: EmergencyScenario) -> str:
    eta = scenario.eta_seconds()
    if eta is None:
        return "Unavailable because speed is 0"
    return f"{int(round(eta))} seconds"


def build_report(
    scenario: EmergencyScenario,
    analysis: dict | None = None,
    analysis_raw: str | None = None,
    image_observations: dict | None = None,
    image_raw: str | None = None,
    analysis_source: str | None = None,
) -> str:
    """Format a TXT report from data already in the session. This function does not call Gemini."""
    del analysis_source
    lines = [
        "EMERGENCYAI INCIDENT SUMMARY",
        "",
        "Scenario",
        f"Emergency Vehicle: {scenario.emergency_vehicle}",
        f"Approach: {scenario.approach}",
        f"Distance: {scenario.distance_meters:.0f} meters",
        f"Speed: {scenario.speed_mph:.0f} mph",
        f"Calculated ETA: {_eta_line(scenario)}",
        f"North traffic: {scenario.north_traffic}",
        f"South traffic: {scenario.south_traffic}",
        f"East traffic: {scenario.east_traffic}",
        f"West traffic: {scenario.west_traffic}",
        f"Weather: {scenario.weather}",
        f"Road condition: {scenario.road_condition}",
        f"Incident: {scenario.incident}",
        f"Current Signal: {scenario.current_signal}",
        "",
        "Traffic Level",
        (analysis or {}).get("traffic_level") or "Not available until Analyze with Gemini has been run.",
        "",
        "Situation Summary",
    ]
    if analysis and analysis.get("situation_summary"):
        lines.append(analysis["situation_summary"])
    elif analysis_raw:
        lines.append(analysis_raw)
    else:
        lines.append("Not available until Analyze with Gemini has been run.")

    lines.extend(["", "Key Concerns", *_bullets((analysis or {}).get("key_concerns"))])
    lines.extend(["", "Responder Considerations", *_bullets((analysis or {}).get("responder_considerations"))])
    lines.extend(["", "Operator Considerations", *_bullets((analysis or {}).get("operator_considerations"))])
    lines.extend(["", "Weather / Road Considerations", *_bullets((analysis or {}).get("weather_considerations"))])
    lines.extend(["", "Human Verification Needed", *_bullets((analysis or {}).get("verification_needed"))])

    if image_observations or (image_raw and str(image_raw).strip()):
        lines.extend(["", "Image Observations"])
        if image_observations:
            lines.extend(
                [
                    f"Traffic density: {image_observations.get('traffic_density', '')}",
                    f"Congestion: {image_observations.get('congestion', '')}",
                    f"Road conditions: {image_observations.get('road_conditions', '')}",
                    f"Weather / visibility: {image_observations.get('weather_visibility', '')}",
                    f"Pedestrians: {image_observations.get('pedestrians', '')}",
                    f"Emergency vehicles: {image_observations.get('emergency_vehicles', '')}",
                    f"Obstructions: {image_observations.get('obstructions', '')}",
                ]
            )
            other = image_observations.get("other_observations") or []
            uncertainty = image_observations.get("uncertainty_notes") or []
            if other:
                lines.append("Other observations:")
                lines.extend(_bullets(other))
            if uncertainty:
                lines.append("Uncertainty:")
                lines.extend(_bullets(uncertainty))
        else:
            lines.append(str(image_raw).strip())

    lines.extend(
        [
            "",
            "This is an AI-assisted simulated report. It does not control traffic signals.",
            "Qualified people must verify the situation before any real-world action.",
        ]
    )
    return "\n".join(lines) + "\n"
