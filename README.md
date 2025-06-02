# Placement-Interview-Task-Patient-health-dashboard
Firebase-powered health data dashboard with ML classification


# Patient Health Dashboard

A comprehensive dashboard for visualizing and analyzing patient health data with predictive capabilities.

## Features

- **Data Visualization**: Interactive charts for health metrics (heart rate, steps, sleep, nutrition)
- **Health Classification**: Machine learning model to categorize patient health status (Good/Moderate/Poor)
- **Data Export**: Export charts as images or full data as CSV/PDF
- **Responsive Design**: Works on desktop and mobile devices

## Technologies

- Frontend: HTML, CSS, JavaScript, Firebase SDK, Chart.js
- Backend: Python, scikit-learn, pandas, numpy

## Setup

1. **Frontend**:
   - Open `index.html` in a modern browser
   - No additional setup required (Firebase config included)

2. **Machine Learning Model**:
   - Install requirements: `pip install pandas numpy scikit-learn`
   - Run the Python script: `python health_classifier.py`

## Data Sources

- Patient data is fetched from Firebase Firestore
- Fallback CSV data can be used if Firebase is unavailable

## Evaluation Metrics

- Model Accuracy: 1.0 (varies by dataset)
- F1 Score: 1.0 (weighted average)
