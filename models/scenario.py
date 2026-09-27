"""Scenario facts and the structured responses Gemini is asked to return."""

from typing import Literal

from pydantic import BaseModel, Field

MPH_TO_MPS = 0.44704

Vehicle = Literal["Ambulance", "Fire Truck", "Police"]
Approach = Literal["North", "South", "East", "West"]
Weather = Literal["Clear", "Rain", "Fog"]
RoadCondition = Literal["Normal", "Construction", "Lane Blocked"]
Incident = Literal["None", "Crash", "Stalled Vehicle", "Pedestrian Activity"]
SignalPhase = Literal["North-South Green", "East-West Green"]


def calculate_eta_seconds(distance_meters: float, speed_mph: float) -> float | None:
    """ETA is computed locally. Gemini never calculates it.

    speed_mps = speed_mph * 0.44704
    eta_seconds = distance_meters / speed_mps
    """
    if speed_mph <= 0 or distance_meters < 0:
        return None
    speed_mps = speed_mph * MPH_TO_MPS
    if speed_mps <= 0:
        return None
    return distance_meters / speed_mps


def format_eta(seconds: float | None) -> str:
    if seconds is None:
        return "Unavailable — set speed above 0"
    total = max(0, int(round(seconds)))
    minutes, secs = divmod(total, 60)
    if minutes:
        return f"{minutes} min {secs} sec"
    return f"{secs} sec"


class EmergencyScenario(BaseModel):
    emergency_vehicle: Vehicle
    approach: Approach
    distance_meters: float = Field(ge=0)
    speed_mph: float = Field(ge=0)
    north_traffic: int = Field(ge=0)
    south_traffic: int = Field(ge=0)
    east_traffic: int = Field(ge=0)
    west_traffic: int = Field(ge=0)
    weather: Weather
    road_condition: RoadCondition
    incident: Incident
    current_signal: SignalPhase

    def eta_seconds(self) -> float | None:
        return calculate_eta_seconds(self.distance_meters, self.speed_mph)

    def traffic_by_approach(self) -> dict[str, int]:
        return {
            "North": self.north_traffic,
            "South": self.south_traffic,
            "East": self.east_traffic,
            "West": self.west_traffic,
        }

    def prompt_block(self) -> str:
        eta = self.eta_seconds()
        eta_line = (
            f"{format_eta(eta)} ({eta:.1f} seconds)"
            if eta is not None
            else "Unavailable because speed is 0. Do not estimate a replacement ETA."
        )
        counts = self.traffic_by_approach()
        return "\n".join(
            [
                "Simulated scenario inputs (facts entered by the operator, not observed from a camera):",
                f"- Emergency vehicle: {self.emergency_vehicle}",
                f"- Approach direction: {self.approach}",
                f"- Distance: {self.distance_meters:.0f} meters",
                f"- Speed: {self.speed_mph:.0f} mph",
                f"- Calculated ETA: {eta_line}",
                f"- North traffic count: {counts['North']}",
                f"- South traffic count: {counts['South']}",
                f"- East traffic count: {counts['East']}",
                f"- West traffic count: {counts['West']}",
                f"- Weather: {self.weather}",
                f"- Road condition: {self.road_condition}",
                f"- Incident: {self.incident}",
                f"- Current signal phase: {self.current_signal}",
            ]
        )


class TrafficAnalysis(BaseModel):
    situation_summary: str = Field(description="Short plain-language summary of the simulated situation.")
    traffic_level: str = Field(description="Overall traffic level, such as Light, Moderate, or Heavy.")
    key_concerns: list[str] = Field(description="Main situational concerns. Not control commands.")
    responder_considerations: list[str] = Field(
        description="Situational notes for emergency responders. Not orders."
    )
    weather_considerations: list[str] = Field(
        description="How weather and road condition may affect the situation."
    )
    operator_considerations: list[str] = Field(
        description="Short considerations for a human operator. Not signal-control commands."
    )
    verification_needed: list[str] = Field(
        description="Facts a person should verify before acting."
    )


class ImageObservations(BaseModel):
    traffic_density: str
    congestion: str
    road_conditions: str
    weather_visibility: str
    pedestrians: str
    emergency_vehicles: str
    obstructions: str
    other_observations: list[str]
    uncertainty_notes: list[str]

