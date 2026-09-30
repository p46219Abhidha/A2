 # --- Feature Engineering for App ---
    smoker_age = smoker * age
    smoker_children = smoker * children
    bmi_age = bmi * age
    is_obese = 1 if bmi >= 30 else 0
    
    if age <= 30: age_group = 0
    elif age <= 45: age_group = 1
    elif age <= 60: age_group = 2
    else: age_group = 3
    
    lifestyle_risk_score = (smoker * 3) + (is_obese * 2)

    # Create the input array with ALL columns
    input_data = {
        'age': [age],
        'sex': [sex],
        'bmi': [bmi],
        'children': [children],
        'smoker': [smoker],
        'region': [region],
        'smoker_age': [smoker_age],
        'smoker_children': [smoker_children],
        'bmi_age': [bmi_age],
        'is_obese': [is_obese],
        'age_group': [age_group],
        'lifestyle_risk_score': [lifestyle_risk_score]
    }
    input_df = pd.DataFrame(input_data)
    input_df = input_df[model_columns] # Ensure column order matches training
