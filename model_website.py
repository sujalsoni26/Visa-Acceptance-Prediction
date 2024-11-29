import pickle
import sys
import json
import pandas as pd
import warnings
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import random
from sklearn.ensemble import GradientBoostingClassifier

# Specify the columns to scale
columns_to_scale = ['PREVAILING_WAGE_1', 'COLI']  # Replace with your column names

# Suppress all warnings
warnings.filterwarnings("ignore")

onehotencoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
scaler = MinMaxScaler()


# Load the trained model
def prediction(df):
    with open('ml_model.pkl', 'rb') as file:
        model = pickle.load(file)
   
    # Ensure numeric columns are treated as numeric
    df[columns_to_scale] = df[columns_to_scale].apply(pd.to_numeric, errors='coerce')

    # Select all columns except 'CASE_STATUS'
    X_normal = df.loc[:, df.columns != 'CASE_STATUS']
    
    # Identify categorical columns
    categorical_columns = X_normal.select_dtypes(include=['object']).columns
    onehotencoder.fit(X_normal[categorical_columns])

    # Apply the OneHotEncoder to the categorical columns
    X_categorical_encoded = onehotencoder.transform(X_normal[categorical_columns])

    # Create a DataFrame for the encoded categorical features
    encoded_feature_names = onehotencoder.get_feature_names_out(categorical_columns)
    X_categorical_encoded_df = pd.DataFrame(X_categorical_encoded, columns=encoded_feature_names)

    # Drop original categorical columns and merge encoded features
    X_normal = X_normal.drop(categorical_columns, axis=1).reset_index(drop=True)
    X_normal = pd.concat([X_normal, X_categorical_encoded_df.reset_index(drop=True)], axis=1)

    # Manually modify column names to match the training feature names
    original_feature_names = ['VISA_CLASS', 'NEW_EMPLOYMENT', 'CONTINUED_EMPLOYMENT', 'CHANGE_PREVIOUS_EMPLOYMENT', 'NEW_CONCURRENT_EMPLOYMENT', 'CHANGE_EMPLOYER', 'AMENDED_PETITION', 'EMPLOYER_NAME', 'SOC_CODE', 'JOB_TITLE', 'AGENT_REPRESENTING_EMPLOYER', 'WILLFUL_VIOLATOR', 'H-1B_DEPENDENT', 'TOTAL_WORKER_POSITIONS', 'FULL_TIME_POSITION', 'PREVAILING_WAGE_1', 'WORKSITE', 'lat', 'lng', 'COLI'] 

    X_normal.columns = original_feature_names

    # # Scale the numeric features using the saved scaler
    
    scaler.fit(X_normal[columns_to_scale])

    # Predict using the trained model
    res = model.predict(X_normal)
    return res

if __name__ == "__main__":
    # Redirect print to a log file
    # sys.stdout = open('python_output.log', 'w')

    input_data = json.loads(sys.stdin.read())  # Parse the input JSON string
    new_dict = {key: value for key, value in input_data.items()}  # Create a new dictionary from input_data
    
    # Convert the dictionary to a DataFrame
    df = pd.DataFrame([new_dict])  # Convert the dictionary into a DataFrame, wrapping it in a list for a single row
    
    # Call the prediction function
    result = prediction(df)
    # print(new_dict)
    # Map numeric predictions to string labels
    if result == 0:
        result = "Accepted"
    elif result == 1:
        result = "Accepted Withdrawn"
    elif result == 3:
        result = "Withdrawn"
    elif result == 2:
        result = "Denied"
    
    # Output the result as a JSON string
    # print("random: ", random.randint(1, 100))
    print(json.dumps(result))  # Send back a simple JSON response
    sys.stdout.flush()
