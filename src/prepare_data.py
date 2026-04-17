def generate_recommendation(input_data):
    bmi = input_data.get("bmi", 0)
    activity = input_data.get("physical_activity_minutes_per_week", 0)

    if bmi > 30:
        return "High risk! Increase physical activity and improve diet"
    elif activity < 2:
        return "Moderate risk! Improve activity levels"
    else:
        return "Low risk! Maintain current lifestyle"