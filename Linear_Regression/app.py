
# Auto MPG Chat Assistant (with km/L + Evaluation Metrics)

! pip install gradio
! pip install pandas scikit-learn matplotlib seaborn
import gradio as gr
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import re
import os

# Load and Clean Dataset
data = pd.read_csv("/content/Auto-MPG-Linear-Regression-project/Linear_Regression/auto-mpg[1].csv")

# Fix missing values
data['horsepower'] = data['horsepower'].replace('?', pd.NA)
data['horsepower'] = pd.to_numeric(data['horsepower'], errors='coerce')
data['horsepower'] = data['horsepower'].fillna(data['horsepower'].mean())
data = data.dropna()

# Normalize column names
data.columns = [c.lower().strip() for c in data.columns]
data["car name"] = data["car name"].str.lower()

# Select only required features
required_features = ["cylinders", "displacement", "horsepower", "weight", "acceleration", "model year"]
X = data[required_features]
y = data["mpg"]

# Split for Evaluation
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate Model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

evaluation_summary = f"""Model Evaluation Summary

R² Score (Explained Variance): **{r2:.3f}**
MAE (Mean Absolute Error): **{mae:.3f}**
RMSE (Root Mean Square Error): **{rmse:.3f}**

Interpretation:
- R² close to 1 → Model fits data well.
- Lower MAE/RMSE → Better prediction accuracy.
"""

# Helper: Explain topics like “What is MPG?” or “How does weight affect MPG?”
def explain_topic(message):
    msg = message.lower()

    if "mpg" in msg and ("what" in msg or "mean" in msg):
        return (
            " **MPG (Miles per Gallon)** means how many miles a car can travel using one gallon of fuel.\n"
            " Higher MPG = Better fuel efficiency.\n"
            " Lower MPG = More fuel consumption.\n\n"
            "Example: 30 MPG ≈ 12.75 km/L."
        )
    elif "affect" in msg or "impact" in msg or "influence" in msg:
        if "weight" in msg:
            return " Heavier cars usually have **lower MPG** because the engine works harder to move more mass."
        elif "horsepower" in msg:
            return " Higher horsepower often means **lower MPG**, since more powerful engines consume more fuel."
        elif "cylinders" in msg:
            return " More cylinders → larger engine → **lower MPG**, generally."
        elif "acceleration" in msg:
            return " Faster acceleration can slightly reduce MPG since more fuel is burned for performance."
        elif "model year" in msg:
            return " Newer models often have **better MPG** due to improved technology."
        elif "displacement" in msg:
            return " Higher engine displacement (bigger engine size) → typically **lower MPG**."
        else:
            return " Many factors affect MPG — mainly weight, horsepower, and engine size."
    elif "dataset" in msg or "data" in msg:
        return (
            f"The dataset has {len(data)} cars with 8 features.\n"
            "It includes cylinders, displacement, horsepower, weight, acceleration, model year, origin, and car name."
        )
    elif "who are you" in msg or "what can you do" in msg:
        return (
            " I'm the **Auto MPG Chat Assistant**!\n"
            "I can estimate a car's fuel efficiency (MPG or km/L) and explain how features affect it."
        )
    elif "accurate" in msg or "evaluate" in msg or "performance" in msg or "score" in msg:
        return evaluation_summary
    return None

# Main Chat Function
def respond(message, history):
    message_lower = message.lower()
    explain_response = explain_topic(message)
    if explain_response:
        return explain_response

    matched_cars = [car for car in data["car name"].unique() if car in message_lower]
    if matched_cars:
        car = matched_cars[0]
        car_data = data[data["car name"] == car].iloc[0]
        input_data = pd.DataFrame({col: [car_data[col]] for col in required_features}).astype(float)
        mpg_pred = model.predict(input_data)[0]
        km_per_liter = mpg_pred * 0.425144
        reply = f" I found **{car.title()}** in my dataset!\n\n"
        reply += f" Estimated Fuel Efficiency:\n **{mpg_pred:.2f} MPG**\n **{km_per_liter:.2f} km/L**\n\n"
        reply += (
            f" Features → Cylinders: {car_data['cylinders']}, Displacement: {car_data['displacement']}, "
            f"Horsepower: {car_data['horsepower']}, Weight: {car_data['weight']}, "
            f"Acceleration: {car_data['acceleration']}, Model Year: {car_data['model year']}"
        )
        return reply

    import re
    try:
        manual_features_list = [float(x.strip()) for x in re.split(r'[,\s]+', message) if x.strip()]
        if len(manual_features_list) == len(required_features):
            input_data = pd.DataFrame([manual_features_list], columns=required_features).astype(float)
            mpg_pred = model.predict(input_data)[0]
            km_per_liter = mpg_pred * 0.425144
            reply = (
                f"Based on your input features ({', '.join(map(str, manual_features_list))}):\n"
                f"**{mpg_pred:.2f} MPG**\n **{km_per_liter:.2f} km/L**\n\n"
                f" Details → Cylinders: {manual_features_list[0]}, Displacement: {manual_features_list[1]}, "
                f"Horsepower: {manual_features_list[2]}, Weight: {manual_features_list[3]}, "
                f"Acceleration: {manual_features_list[4]}, Model Year: {manual_features_list[5]}"
            )
            return reply
    except ValueError:
        pass

    # Fallback message
    return (
        " Car not found. Please enter the features manually like:\n"
        "`8 cyl, 302 engine, 140 hp, 3449 lbs, acceleration 10.5, model year 75`\n\n"
        "Alternatively, you can provide the 6 numerical feature values separated by commas or spaces in this order: Cylinders, Displacement, Horsepower, Weight, Acceleration, Model Year.\n"
        "For example: `4, 97, 68, 2025, 18.2, 82`\n\n"
        " I'll then estimate both **MPG** and **km/L**."
    )

# Gradio Chat Interface
chatbot = gr.ChatInterface(
    fn=respond,
    title=" Auto MPG Chat With Atheek Fareez ",
    description=(
        "Chat with me to estimate car fuel efficiency (MPG and km/L), learn how features affect it, "
        "and check model accuracy.\n\n"
        " Try asking:\n"
        "- 'What is MPG?'\n"
        "- 'How does weight affect MPG?'\n"
        "- 'Predict MPG for 1970 Toyota Corolla'\n"
        "- 'Show model evaluation'\n"
        " You can also provide the 6 numerical feature values separated by commas or spaces in this order: Cylinders, Displacement, Horsepower, Weight, Acceleration, Model Year."
    ),
    examples=[
        ["What is MPG?"],
        ["How does horsepower affect MPG?"],
        ["Predict MPG for 1970 Toyota Corolla"],
        ["8 cyl, 302 engine, 140 hp, 3449 lbs, acceleration 10.5, model year 75"],
        ["4, 97, 68, 2025, 18.2, 82"],
        ["How accurate is your model?"]
    ]
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8500))
    chatbot.launch(server_name="0.0.0.0", server_port=port, share=False)
