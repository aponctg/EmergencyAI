# EmergencyAI

EmergencyAI is a Gemini-powered decision-support application for simulated emergency traffic situations. It combines operator-entered emergency traffic information with AI-based traffic image observations to create a structured situation assessment. The application is designed for human decision support and does not directly control traffic signals.

## Features

### Emergency Scenario Simulation

Users can configure:

- emergency vehicle type
- approach direction
- distance
- speed
- traffic counts
- current traffic signal
- weather conditions
- road conditions
- incident conditions

EmergencyAI calculates the vehicle's estimated arrival time locally in Python.

### Gemini Scenario Analysis

Google Gemini analyzes the structured scenario and returns:

- traffic level
- situation summary
- key concerns
- responder considerations
- operator considerations
- weather and road considerations
- information requiring human verification

### Gemini Vision

Users can upload a traffic/intersection image.

Gemini extracts observations about:

- traffic density
- congestion
- road conditions
- weather and visibility
- pedestrians
- emergency vehicles when clearly visible
- obstructions
- other observations
- uncertainty

### Combined Scenario + Image Analysis

EmergencyAI can combine the operator-entered scenario with Gemini's image observations.

The application preserves the distinction between:

- operator-entered information
- image-derived observations

This allows Gemini to identify relevant concerns as well as information that may require human verification.

### Ask EmergencyAI

Users can ask contextual questions about the current scenario and latest analysis.

Example:

> What is the biggest concern?

Gemini receives the current scenario and analysis as context before answering.

### Incident Report

EmergencyAI creates a downloadable text incident summary using the existing structured scenario and analysis.

Report generation is performed locally and does not require an additional Gemini API request.

### Built With
- Python
- Streamlit
- Google Gemini API
- Google GenAI Python SDK
- Pydantic
- Pillow
- python-dotenv

### Installation:
Clone the repository
Create a virtual environment if desired and install the dependencies.
Create a .env file
    GEMINI_API_KEY=your_gemini_api_key
    GEMINI_MODEL=your_supported_gemini_model
Run the application

The End
