import os
import json
import pandas as pd
import joblib
from pathlib import Path
from fpdf import FPDF
from io import BytesIO
from datetime import datetime
from groq import Groq

# PDF Class
class PDF(FPDF):
    def __init__(self, logo_path):
        super().__init__()
        self.logo_path = logo_path

    def header(self):
        if self.logo_path.exists() and hasattr(self, 'local_context'):
            with self.local_context(fill_opacity=0.3):
                self.image(str(self.logo_path), x=45, y=80, w=120)
        elif self.logo_path.exists():
            self.image(str(self.logo_path), x=45, y=80, w=120)

        if self.page_no() == 1 and self.logo_path.exists():
            self.image(str(self.logo_path), x=10, y=8, w=50)

# Register Helper methods
def get_models_dir():
    BASE_DIR = Path(__file__).resolve().parent.parent
    return BASE_DIR / "models"

def get_registry():
    """Load the model registry JSON. Returns dict with active_model and models list."""
    registry_path = get_models_dir() / "model_registry.json"
    if registry_path.exists():
        with open(registry_path, "r") as f:
            return json.load(f)
    return {"active_model": None, "models": []}

def get_active_model_metadata():
    """
    Returns metadata dict for the currently active model.
    Keys: filename, training_years, registered_at
    Returns None if no active model set.
    """
    registry = get_registry()
    active_filename = registry.get("active_model")
    if not active_filename:
        return None
    return next((m for m in registry["models"] if m["filename"] == active_filename), None)


def get_sweet_spot_year():
    """
    Returns the optimal prediction year for the current active model.
    e.g. if trained on [2023, 2024, 2025] → returns 2026
    Returns None if no active model.
    """
    meta = get_active_model_metadata()
    if meta and meta.get("training_years"):
        return max(meta["training_years"]) + 1
    return None

def get_valid_year_range():
    """
    Returns (sweet_spot_year, max_allowed_year) for the year dropdown.
    sweet_spot_year = max(training_years) + 1  → most accurate
    Years before sweet_spot should be hidden (already trained on them).
    Years after sweet_spot are allowed but with a warning.
    Returns (None, None) if no active model.
    """
    meta = get_active_model_metadata()
    if meta and meta.get("training_years"):
        sweet_spot = max(meta["training_years"]) + 1
        return sweet_spot, sweet_spot + 4  # show 5 years starting from sweet spot
    # Fallback: no registry yet (original model)
    return 2026, 2030

# MODEL LOADING
def load_model(model_path=None):
    """
    Load the active model from the registry.
    Falls back to the original hardcoded model if no registry exists.
    """
    models_dir = get_models_dir()

    # Try registry first
    registry = get_registry()
    active_filename = registry.get("active_model")

    if active_filename:
        active_path = models_dir / active_filename
        if active_path.exists():
            return joblib.load(active_path)

    # Fallback: original model
    fallback_path = models_dir / "Final_Ridge_Model_23_24_25.pkl"
    if fallback_path.exists():
        return joblib.load(fallback_path)

    # Last resort: use provided path
    if model_path and os.path.exists(model_path):
        return joblib.load(model_path)

    return None

# Input Helper methods
def create_input_df(T200, U709, Epson, Dolomite, Urea, BioShield,
                    temperature_2m_mean, rain_sum, wind_speed_10m_mean, et0_fao_evapotranspiration,
                    Dolomite_moisture, YPH_lag1, YPH_lag12, Compost_lag1):
    return pd.DataFrame([[T200, U709, Epson, Dolomite, Urea, BioShield,
                          temperature_2m_mean, rain_sum, wind_speed_10m_mean, et0_fao_evapotranspiration,
                          Dolomite_moisture, YPH_lag1, YPH_lag12, Compost_lag1]],
                        columns=['T200', 'U709', 'Epson Salt', 'Dolomite', 'Urea', 'BioShield',
                                 'temperature_2m_mean', 'rain_sum', 'wind_speed_10m_mean',
                                 'et0_fao_evapotranspiration', 'Dolomite_moisture',
                                 'YPH_lag1', 'YPH_lag12', 'Compost_lag1'])


def find_dolomite_moisture(dolomite, rain_sum, et0_fao_evapotranspiration):
    return dolomite * (rain_sum - et0_fao_evapotranspiration)

# PDF REPORT

def generate_pdf_report(T200, U709, Epson, Dolomite, Urea, BioShield,
                        temperature_2m_mean, rain_sum, wind_speed_10m_mean, et0_fao_evapotranspiration,
                        Dolomite_moisture, YPH_lag1, YPH_lag12, Compost_lag1,
                        predicted_yph, explanation_text, year=None, month=None):

    BASE_DIR = Path(__file__).resolve().parent.parent
    logo_path = BASE_DIR / "assets" / "logo_white_cropped.png"

    pdf = PDF(logo_path)
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    pdf.set_font("Arial", size=10)
    current_date = datetime.now().strftime("%d/%m/%Y")
    current_time = datetime.now().strftime("%H:%M:%S")
    pdf.set_y(8)
    pdf.cell(0, 5, f"Date: {current_date}", ln=1, align='R')
    pdf.cell(0, 5, f"Time: {current_time}", ln=1, align='R')
    pdf.ln(15)

    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Tea Yield Prediction Report", ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", 'B', 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 10, f"Predicted Yield Per Hectare (YPH): {predicted_yph:.2f} kg", ln=1)
    pdf.ln(2)

    pdf.set_font("Arial", size=11)
    pdf.cell(95, 8, f"Year - {year if year else 'N/A'}", ln=0)
    pdf.cell(95, 8, f"Month - {month if month else 'N/A'}", ln=1)
    pdf.ln(5)

    # Model info
    meta = get_active_model_metadata()
    if meta:
        train_yrs = ', '.join(str(y) for y in meta.get("training_years", []))
        pdf.set_font("Arial", size=10)
        pdf.cell(0, 8, f"Model Trained On: {train_yrs}", ln=1)
        pdf.ln(3)

    # WEATHER
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(0, 8, "Weather Data", ln=1)
    pdf.set_font("Arial", size=10)
    pdf.cell(95, 8, f"Temperature (Mean, 2m): {temperature_2m_mean} C", ln=0)
    pdf.cell(95, 8, f"Wind Speed (10m): {wind_speed_10m_mean} m/s", ln=1)
    pdf.cell(95, 8, f"Rainfall (Total): {rain_sum} mm", ln=0)
    pdf.cell(95, 8, f"Evapotranspiration: {et0_fao_evapotranspiration} mm/day", ln=1)
    pdf.ln(5)

    # FERTILIZER
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(0, 8, "Fertilizer Data", ln=1)
    pdf.set_font("Arial", size=10)
    pdf.cell(95, 8, f"T200: {T200} kg", ln=0)
    pdf.cell(95, 8, f"Epson Salt: {Epson} kg", ln=1)
    pdf.cell(95, 8, f"U709: {U709} kg", ln=0)
    pdf.cell(95, 8, f"Dolomite: {Dolomite} kg", ln=1)
    pdf.cell(95, 8, f"BioShield: {BioShield} kg", ln=0)
    pdf.cell(95, 8, f"Urea: {Urea} kg", ln=1)
    pdf.ln(5)

    # LAG
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(0, 8, "Historical (Lag) Data", ln=1)
    pdf.set_font("Arial", size=10)
    pdf.cell(95, 8, f"Last Month Compost: {Compost_lag1} kg", ln=0)
    pdf.cell(95, 8, f"Last Year Same Month YPH: {YPH_lag12}", ln=1)
    pdf.cell(95, 8, f"Last Month YPH: {YPH_lag1}", ln=1)
    pdf.ln(5)

    # EXPLANATION
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(0, 8, "Explanation & Insights", ln=1)
    pdf.set_font("Arial", size=10)
    pdf.multi_cell(0, 8, explanation_text)

    pdf_bytes = pdf.output()
    if isinstance(pdf_bytes, str):
        pdf_bytes = pdf_bytes.encode('latin-1')

    pdf_buffer = BytesIO(pdf_bytes)
    pdf_buffer.seek(0)

    filename = f"Tea_Yield_Report_{month}_{year}.pdf" if month and year else "Tea_Yield_Report.pdf"
    return pdf_buffer, filename

# AI EXPLANATION
def generate_ai_explanation(pipeline, input_df):
    try:
        imputer = pipeline.named_steps['imputer']
        scaler = pipeline.named_steps['scaler']
        ridge = pipeline.named_steps['ridge']

        X_imputed = imputer.transform(input_df)
        X_scaled = scaler.transform(X_imputed)
        contributions = X_scaled[0] * ridge.coef_

        explanation_df = pd.DataFrame({
            "Feature": input_df.columns,
            "Value": input_df.iloc[0].values,
            "Contribution": contributions
        }).sort_values(by="Contribution", key=abs, ascending=False)

        feature_summary = ""
        for _, row in explanation_df.iterrows():
            feature_summary += (
                f"{row['Feature']} = {row['Value']} "
                f"(impact on yield: {row['Contribution']:.2f})\n"
            )

        prompt = f"""You are GPTea AI, an agricultural assistant that explains tea yield predictions to farmers in simple, easy-to-understand English. 
            Your explanation must include: Do not copy the example output. The following is for reference only indentation and everything must be same. Generate explanation only for the input below.

            IMPORTANT NOTES TO CONSIDER
            1. Keep explanations tight. Do not add extra blank lines. Display each bullet consecutively without extra spacing. Leave spacing like how its mentioned in the example input.
            2. Do not copy the example output. The following is for reference only. Generate explanation only for the input below.

            1. Introduction of yourself.
            2. Past yield information (last month and same month last year).
            3. Weather variables impact.
            4. Fertilizer variables impact (including T200, U709, Epson, Dolomite, Urea, BioShield, Compost_lag1).
            5. Recommendations to improve yield.
            6. Do NOT repeat the same sentence exactly; make the sentences varied.
            7. Use **human-friendly names** for features:
               - temperature_2m_mean → Temperature
               - rain_sum → Rainfall
               - wind_speed_10m_mean → Wind speed
               - et0_fao_evapotranspiration → Evapotranspiration
               - Dolomite_moisture → Dolomite Moisture
               - Other features use the same name
            8. Avoid technical jargon; be simple and conversational.
            9. Explain the contribution of each variable in a **positive or negative way**.

            Do not copy the example output. The following is for reference only. Generate explanation only for the input below.
            ---
            Below is a example output for a given input.
            # Example Input:
            Predicted Yield: 340
            Past Yield Last Month: 320
            Past Yield Same Month Last Year: 300

            Feature Contributions:
            Temperature = 28.5 (impact: +0.40)
            Rainfall = 150 (impact: +1.30)
            Wind speed = 3.2 (impact: -0.25)
            Evapotranspiration = 5.8 (impact: +0.10)
            Dolomite Moisture = 16 (impact: -0.50)
            T200 = 6 (impact: +0.75)
            U709 = 3 (impact: +0.50)
            Epson = 2 (impact: +0.20)
            Dolomite = 8 (impact: -0.60)
            Urea = 5 (impact: +0.35)
            BioShield = 1 (impact: -0.10)
            Compost_lag1 = 20 (impact: +0.50)

            # Example Output:
            Hi, I am GPTea AI. I am going to explain why I predicted this yield.
            In the same month of last year, the yield was 300 and last month it was 320, which explains part of the current prediction.\n            
            Weather Variables:
            1. The temperature is 28.5, which slightly increased the yield.
            2. The rainfall was 150 mm, contributing strongly to a higher yield.
            3. Wind speed of 3.2 slightly reduced the yield.
            4. Evapotranspiration of 5.8 had a small positive effect.
            5. Dolomite Moisture of 16 slightly decreased the yield.\n
            Fertilizer Variables:
            1. T200 fertilizer significantly boosted the yield.
            2. U709 also contributed positively.
            3. Epson had a small positive impact.
            4. Dolomite slightly lowered the yield.
            5. Urea improved the yield moderately.
            6. BioShield had a minor negative effect.
            7. Compost_lag1 helped improve the yield slightly.\n
            Recommendations:
            - Consider optimizing the Dolomite and BioShield application to minimize negative impact.
            - Maintain proper irrigation and monitor rainfall patterns.
            - Keep applying beneficial fertilizers like T200, U709, Urea, and Compost regularly to improve yield further.
            - Ensure temperature and wind conditions are favorable during growth periods.
            Do not copy the example output. The following is for reference only. Generate explanation only for the input below.
            ---
            Feature Contributions:
            {feature_summary}
            """

        client = Groq(api_key="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")
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}]
        )

        explanation_text = response.choices[0].message.content
    except Exception as e:
        explanation_text = f"AI explanation unavailable: {str(e)}"

    return explanation_text