"""One-click demo scenarios for a live judging walkthrough."""

VEHICLES = ["Ambulance", "Fire Truck", "Police"]
APPROACHES = ["North", "South", "East", "West"]
WEATHER_OPTIONS = ["Clear", "Rain", "Fog"]
ROAD_OPTIONS = ["Normal", "Construction", "Lane Blocked"]
INCIDENT_OPTIONS = ["None", "Crash", "Stalled Vehicle", "Pedestrian Activity"]
SIGNAL_OPTIONS = ["North-South Green", "East-West Green"]

DEFAULT_SCENARIO = {
    "emergency_vehicle": "Ambulance",
    "approach": "East",
    "distance_meters": 500,
    "speed_mph": 40,
    "north_traffic": 15,
    "south_traffic": 15,
    "east_traffic": 15,
    "west_traffic": 15,
    "weather": "Clear",
    "road_condition": "Normal",
    "incident": "None",
    "current_signal": "North-South Green",
}

PRESETS = [
    {
        "id": "A",
        "label": "Scenario A: Ambulance + Heavy Traffic",
        "log": "Scenario loaded: A — Ambulance + Heavy Traffic",
        "blurb": "Ambulance from the East, 400 m, 35 mph, heavy east traffic, clear, normal road, no incident.",
        "values": {
            "emergency_vehicle": "Ambulance",
            "approach": "East",
            "distance_meters": 400,
            "speed_mph": 35,
            "north_traffic": 25,
            "south_traffic": 20,
            "east_traffic": 85,
            "west_traffic": 20,
            "weather": "Clear",
            "road_condition": "Normal",
            "incident": "None",
            "current_signal": "East-West Green",
        },
    },
    {
        "id": "B",
        "label": "Scenario B: Fire Truck + Crash",
        "log": "Scenario loaded: B — Fire Truck + Crash",
        "blurb": "Fire truck from the North, 550 m, 40 mph, medium traffic, clear, normal road, crash.",
        "values": {
            "emergency_vehicle": "Fire Truck",
            "approach": "North",
            "distance_meters": 550,
            "speed_mph": 40,
            "north_traffic": 40,
            "south_traffic": 40,
            "east_traffic": 40,
            "west_traffic": 40,
            "weather": "Clear",
            "road_condition": "Normal",
            "incident": "Crash",
            "current_signal": "North-South Green",
        },
    },
    {
        "id": "C",
        "label": "Scenario C: Ambulance + Bad Weather",
        "log": "Scenario loaded: C — Ambulance + Bad Weather",
        "blurb": "Ambulance from the South, 300 m, 30 mph, heavy traffic, rain, lane blocked, pedestrian activity.",
        "values": {
            "emergency_vehicle": "Ambulance",
            "approach": "South",
            "distance_meters": 300,
            "speed_mph": 30,
            "north_traffic": 75,
            "south_traffic": 75,
            "east_traffic": 70,
            "west_traffic": 70,
            "weather": "Rain",
            "road_condition": "Lane Blocked",
            "incident": "Pedestrian Activity",
            "current_signal": "North-South Green",
        },
    },
]

EXAMPLE_QUESTIONS = [
    "What is the biggest concern?",
    "What should the operator verify?",
]
