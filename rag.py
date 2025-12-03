import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error
import json

# ----------------------------------------------------
# 1. LOAD DATA
# ----------------------------------------------------

df = pd.read_csv("training_data.csv")

# Features (inputs to the model)
feature_cols = ["pH","N","P2O5","K2O","OC","EC","region","season","stage",
                "acreage_ha","forecast_rain_mm"]

# Convert categorical → numeric
df_encoded = pd.get_dummies(df, columns=["region","season","stage"], drop_first=True)

X = df_encoded.drop([
    "apply_dolomite","add_urea","add_TSP","add_MOP","split_3","delay_due_to_rain",
    "dose_dolomite","dose_urea","dose_TSP","dose_MOP"
], axis=1)

# Action labels (classification)
actions = ["apply_dolomite","add_urea","add_TSP","add_MOP","split_3","delay_due_to_rain"]

# Dose labels (regression)
doses = ["dose_dolomite","dose_urea","dose_TSP","dose_MOP"]

y_actions = df[actions]
y_doses = df[doses]

# ----------------------------------------------------
# 2. TRAIN MODELS
# ----------------------------------------------------

# Train 6 classifiers (one for each action)
classifiers = {}
for label in actions:
    model = RandomForestClassifier(n_estimators=300, random_state=42)
    model.fit(X, y_actions[label])
    classifiers[label] = model

# Train 4 regressors (one for each fertilizer dose)
regressors = {}
for label in doses:
    model = RandomForestRegressor(n_estimators=300, random_state=42)
    model.fit(X, y_doses[label])
    regressors[label] = model

print("Training complete! Models are ready.")


# ----------------------------------------------------
# 3. FUNCTION TO MAKE PREDICTIONS
# ----------------------------------------------------

def recommend_plan(input_json):
    # Convert to dataframe
    user_df = pd.DataFrame([input_json])

    # One-hot encode like training data
    user_df_encoded = pd.get_dummies(user_df)
    user_df_encoded = user_df_encoded.reindex(columns=X.columns, fill_value=0)

    # Action predictions
    action_outputs = {}
    for label, model in classifiers.items():
        action_outputs[label] = int(model.predict(user_df_encoded)[0])

    # Dose predictions
    dose_outputs = {}
    for label, model in regressors.items():
        predicted = model.predict(user_df_encoded)[0]
        dose_outputs[label] = round(float(predicted), 2)

    # Build final response
    result = {
        "actions": action_outputs,
        "doses_kg_ha": dose_outputs
    }

    return result


# ----------------------------------------------------
# 4. TEST THE ENGINE WITH A SAMPLE INPUT
# ----------------------------------------------------

sample_input = {
    "pH": 4.3,
    "N": 1.4,
    "P2O5": 0.2,
    "K2O": 1.1,
    "OC": 1.8,
    "EC": 0.2,
    "region": "Kandy",
    "season": "SW_monsoon",
    "stage": "mature",
    "acreage_ha": 1.5,
    "forecast_rain_mm": 28
}

print(json.dumps(recommend_plan(sample_input), indent=4))
