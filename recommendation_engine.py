class RecommendationEngine:
    def __init__(self):
        # Define thresholds and rules
        self.rules = {
            "sodium_low": {
                "threshold": 0.5,
                "message": "Increase sodium using gypsum or sodium nitrate irrigation."
            },
            "soil_pH_low": {
                "threshold": 4.5,
                "message": "Apply dolomite to raise soil pH gradually."
            },
            "soil_pH_high": {
                "threshold": 5.5,
                "message": "Avoid over-liming; use acid-forming fertilizers."
            },
            "brown_blight": {
                "message": "Apply copper-based fungicides, prune infected shoots, and improve field sanitation."
            }
        }

    def recommend_from_soil(self, soil_data: dict):
        recommendations = []

        if "sodium" in soil_data and soil_data["sodium"] < self.rules["sodium_low"]["threshold"]:
            recommendations.append(self.rules["sodium_low"]["message"])

        if "pH" in soil_data:
            if soil_data["pH"] < self.rules["soil_pH_low"]["threshold"]:
                recommendations.append(self.rules["soil_pH_low"]["message"])
            elif soil_data["pH"] > self.rules["soil_pH_high"]["threshold"]:
                recommendations.append(self.rules["soil_pH_high"]["message"])

        return recommendations

    def recommend_from_pest(self, pest_name: str):
        pest_name = pest_name.lower()
        if "brown blight" in pest_name:
            return [self.rules["brown_blight"]["message"]]
        return ["No specific recommendation found for this pest."]
