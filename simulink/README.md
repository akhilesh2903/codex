# Simulink Telemedicine Simulation

This folder is designated for `DR_Telemedicine.slx`.

## Workflow Overview
The Simulink model simulates the rural clinic telemedicine workflow:

1. **Patient Source**: Represents patient arrivals (e.g., poisson distribution).
2. **Image Acquisition Server**: Fundus camera capture time (normally distributed).
3. **Quality Assessment**: Rejection probability routing.
4. **Network Transmission**: Latency and bandwidth simulation.
5. **AI Processing Server**: Queue for the Python backend.
6. **Ophthalmologist Review Queue**: The backlog of images requiring human confirmation.
7. **Simulation Dashboard**: Throughput rates, server utilization, doctor wait time.

## Parameters to configure:
- `patients_per_day`: 100
- `images_per_hour`: 12
- `ai_processing_seconds`: 4
- `doctor_review_seconds`: 60
- `number_of_doctors`: 2
- `number_of_servers`: 1

## Running
Load SIMEVENTS toolbox in MATLAB to run the discrete-event simulation model `DR_Telemedicine.slx`.
