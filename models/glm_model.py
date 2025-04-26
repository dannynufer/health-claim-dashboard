import os
import pandas as pd
import statsmodels.api as sm

def load_and_clean_data():
    # Dynamically find the right filepath based on where this script is
    script_dir = os.path.dirname(__file__)  
    file_path = os.path.join(script_dir, '../data/insurance_data.csv')
    file_path = os.path.abspath(file_path)

    df = pd.read_csv(file_path)
    
    # Drop rows with missing age
    df = df.dropna(subset=['age'])
    
    # One-hot encode categorical variables
    df_encoded = pd.get_dummies(df, columns=['gender', 'smoker', 'diabetic'], drop_first=True)
    
    # Convert bool columns to integers
    for col in df_encoded.columns:
        if df_encoded[col].dtype == 'bool':
            df_encoded[col] = df_encoded[col].astype(int)
    
    return df_encoded

def prepare_features(df_encoded):
    X = df_encoded[['age', 'bmi', 'bloodpressure', 'children', 'gender_male', 'smoker_Yes', 'diabetic_Yes']]
    y = df_encoded['claim']
    
    X = sm.add_constant(X)
    
    return X, y

def train_glm(X, y):
    model = sm.GLM(y, X, family=sm.families.Gamma(link=sm.families.links.log()))
    result = model.fit()
    print(result.summary())
    return result

if __name__ == "__main__":
    df_encoded = load_and_clean_data()
    X, y = prepare_features(df_encoded)
    result = train_glm(X, y)
