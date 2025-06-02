
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.impute import SimpleImputer

# Load the data
file_path = r"C:\Users\mian8\Desktop\placement_project\test_data_my_ai_placemnt.csv"
data = pd.read_csv(file_path)

# Display basic info about the data
print("Data Overview:")
print(data.head())
print("\nData Info:")
print(data.info())
print("\nMissing Values:")
print(data.isnull().sum())

# Step 1: Preprocess the data

# Define features to keep based on actual dataset columns
numeric_features = ['vitals_heart_rate', 'vitals_blood_pressure', 'vitals_temperature',
                    'sleep_duration_hours', 'sleep_interruptions', 'activity_steps',
                    'activity_active_minutes', 'activity_sedentary_hours',
                    'nutrition_calories', 'nutrition_water_oz',
                    'nutrition_macros_carbs_g', 'nutrition_macros_protein_g',
                    'nutrition_macros_fat_g']

categorical_features = ['sleep_quality']

# Step 2: Create helper functions to process the pipe-delimited vital signs
def process_heart_rate(hr_string):
    values = [float(x) for x in hr_string.split('|')]
    return np.mean(values), np.std(values)

def process_blood_pressure(bp_string):
    systolic = []
    diastolic = []
    for reading in bp_string.split('|'):
        s, d = reading.split('/')
        systolic.append(float(s))
        diastolic.append(float(d))
    return np.mean(systolic), np.mean(diastolic)

def process_temperature(temp_string):
    values = [float(x) for x in temp_string.split('|')]
    return np.mean(values)

# Apply processing to create new features
data['hr_mean'] = data['vitals_heart_rate'].apply(lambda x: process_heart_rate(x)[0])
data['hr_std'] = data['vitals_heart_rate'].apply(lambda x: process_heart_rate(x)[1])
data['bp_systolic_mean'] = data['vitals_blood_pressure'].apply(lambda x: process_blood_pressure(x)[0])
data['bp_diastolic_mean'] = data['vitals_blood_pressure'].apply(lambda x: process_blood_pressure(x)[1])
data['temp_mean'] = data['vitals_temperature'].apply(process_temperature)

# Update numeric features with the new processed features
numeric_features = ['hr_mean', 'hr_std', 'bp_systolic_mean', 'bp_diastolic_mean', 'temp_mean',
                   'sleep_duration_hours', 'sleep_interruptions', 'activity_steps',
                   'activity_active_minutes', 'activity_sedentary_hours',
                   'nutrition_calories', 'nutrition_water_oz',
                   'nutrition_macros_carbs_g', 'nutrition_macros_protein_g',
                   'nutrition_macros_fat_g']

# Step 3: Label the data based on thresholds
def assign_health_category(row):
    abnormal_count = 0
    
    # Check heart rate (normal range 60-100)
    if not (60 <= row['hr_mean'] <= 100):
        abnormal_count += 1
    
    # Check blood pressure (normal <120/<80, elevated 120-129/<80, hypertension >=130/>=80)
    if row['bp_systolic_mean'] >= 130 or row['bp_diastolic_mean'] >= 80:
        abnormal_count += 1
    
    # Check sleep duration (recommended 7-9 hours for adults)
    if row['sleep_duration_hours'] < 6:
        abnormal_count += 1
    elif row['sleep_duration_hours'] < 7:
        abnormal_count += 0.5  # Minor deviation
    
    # Check activity (recommended 7,000-10,000 steps/day)
    if row['activity_steps'] < 5000:
        abnormal_count += 1
    elif row['activity_steps'] < 7000:
        abnormal_count += 0.5  # Minor deviation
    
    # Check sleep interruptions (more than 2 is problematic)
    if row['sleep_interruptions'] > 2:
        abnormal_count += 1
    
    # Check sleep quality (poor is problematic)
    if row['sleep_quality'] == 'poor':
        abnormal_count += 1
    
    # Determine category based on abnormal count
    if abnormal_count == 0:
        return 'Good'
    elif abnormal_count <= 2:
        return 'Moderate'
    else:
        return 'Poor'

# Apply labeling function
data['health_category'] = data.apply(assign_health_category, axis=1)

# Check class distribution
print("\nHealth Category Distribution:")
print(data['health_category'].value_counts())

# Step 4: Prepare features and target
X = data[numeric_features + categorical_features]
y = data['health_category']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Create preprocessing pipeline
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Step 5: Train and evaluate the model
model = Pipeline(steps=[('preprocessor', preprocessor),
                       ('classifier', RandomForestClassifier(random_state=42))])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluation metrics
print("\nModel Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(f"F1 Score (weighted): {f1_score(y_test, y_pred, average='weighted'):.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature importance
if hasattr(model.named_steps['classifier'], 'feature_importances_'):
    # Get feature names after preprocessing
    feature_names = numeric_features.copy()
    
    if 'cat' in model.named_steps['preprocessor'].transformers_[1][0]:
        cat_features = model.named_steps['preprocessor'].transformers_[1][1].named_steps['onehot'].get_feature_names_out(categorical_features)
        feature_names.extend(cat_features)
    
    importances = model.named_steps['classifier'].feature_importances_
    feat_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feat_imp = feat_imp.sort_values('Importance', ascending=False)
    print("\nTop 10 Feature Importances:")
    print(feat_imp.head(10))
