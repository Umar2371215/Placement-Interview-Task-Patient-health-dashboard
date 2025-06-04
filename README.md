Patient Health Monitoring System

A comprehensive, Firebase-powered solution for tracking and analyzing patient health metrics with machine learning classification.

KEY FEATURES:
• Real-time visualization of:
  - Heart rate trends
  - Activity levels
  - Sleep patterns
  - Nutritional intake
• Automated health assessment (Good/Moderate/Poor)
• Data export options (PDF/CSV/image formats)
• Fully responsive design (works on all devices)

TECHNOLOGIES USED:
Frontend: HTML5, CSS3, JavaScript (Firebase, Chart.js)
Backend: Python (scikit-learn, pandas, numpy)

HOW TO RUN:
1. Dashboard: Open index.html in any modern browser
2. ML Model:
   pip install pandas numpy scikit-learn
   python health_data_pipeline.py

DATA SOURCES:
• Primary: Firebase Firestore (live connection, config included)
• Fallback: test_data.csv (sample dataset)

PERFORMANCE:
• Accuracy: 100% (test dataset)
• F1 Score: 1.0 (perfect)
• Key Predictors:
  1. Heart rate patterns
  2. Blood pressure
  3. Sleep duration

FILES INCLUDED:
• index.html - Complete dashboard
• health_data_pipeline.py - Analysis script
• test_data.csv - Sample dataset

SYSTEM REQUIREMENTS:
• Python 3.12.7+
• Modern web browser
• Tested on Windows 11 (64-bit)

Note: Results may vary with different datasets. Includes demo Firebase configuration.
