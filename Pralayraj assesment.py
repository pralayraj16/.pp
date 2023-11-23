import pandas as pd
import re

# Load your dataset into a Pandas DataFrame
# Assuming your dataset is in a CSV file, adjust the read_csv() method accordingly
df = pd.read_csv('your_dataset.csv')

# Display the first few rows of the dataset to understand its structure
print(df.head())

# Select a subset of 100 records for manual cleaning and standardization
subset = df.sample(n=100, random_state=42)

# Display the subset for manual inspection
print(subset)

# Manual cleaning and standardization
# Iterate through the subset and clean/standardize the company/supplier names
for index, row in subset.iterrows():
    # Access the company/supplier name in the current row
    company_name = row['company_name_column']  # Replace 'company_name_column' with the actual column name
    
    # Perform manual cleaning and standardization
    # Example: Convert to lowercase, remove extra whitespaces, handle variations
    cleaned_name = company_name.lower().strip()
    
    # Add more manual cleaning steps as needed, such as handling specific variations
    
    # Update the DataFrame with the cleaned name
    subset.at[index, 'cleaned_company_name'] = cleaned_name

# Display the cleaned and standardized subset for verification
print(subset[['company_name_column', 'cleaned_company_name']])

# Update the original dataset with the cleaned names
# This step depends on how you want to store the cleaned data (e.g., overwrite the original file)
# For demonstration purposes, let's save the cleaned subset to a new CSV file
subset.to_csv('cleaned_subset.csv', index=False)

2.
pip install openai

import pandas as pd
import openai

# Set your OpenAI GPT-3 API key
openai.api_key = 'your_api_key'

# Load your dataset into a Pandas DataFrame
# Assuming your dataset is in a CSV file, adjust the read_csv() method accordingly
df = pd.read_csv('your_dataset.csv')

# Select a subset of 100 records for automated standardization
subset = df.sample(n=100, random_state=42)

# Define a function to standardize company/supplier names using GPT-3
def standardize_name(name):
    # Customize the prompt based on your requirements
    prompt = f"Standardize the company name: '{name}'"
    
    # Use GPT-3 to generate the standardized name
    response = openai.Completion.create(
        engine="text-davinci-002",  # Choose the appropriate engine
        prompt=prompt,
        max_tokens=50,  # Adjust as needed
        n=1,  # Number of responses
        stop=None,  # Customize stopping criteria
    )
    
    # Extract the generated text from the response
    standardized_name = response['choices'][0]['text'].strip()
    
    return standardized_name

# Apply the standardization function to the subset
subset['standardized_company_name'] = subset['company_name_column'].apply(standardize_name)

# Display the original and standardized names for verification
print(subset[['company_name_column', 'standardized_company_name']])

# Update the original dataset with the standardized names
# This step depends on how you want to store the cleaned data (e.g., overwrite the original file)
# For demonstration purposes, let's save the standardized subset to a new CSV file
subset.to_csv('standardized_subset.csv', index=False)

3.

project_root/
|-- app/
|   |-- __init__.py
|   |-- standardization.py
|   |-- routes.py
|-- config/
|   |-- __init__.py
|   |-- production_config.py
|   |-- development_config.py
|-- requirements.txt
|-- run.py

import openai

openai.api_key = 'your_api_key'

def standardize_name(name):
    prompt = f"Standardize the company name: '{name}'"
    
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=50,
        n=1,
        stop=None,
    )
    
    standardized_name = response['choices'][0]['text'].strip()
    
    return standardized_name
from flask import Flask, request, jsonify
from app.standardization import standardize_name

app = Flask(__name__)

@app.route('/standardize', methods=['POST'])
def standardize():
    data = request.get_json()
    name = data.get('name', '')
    
    standardized_name = standardize_name(name)
    
    return jsonify({'standardized_name': standardized_name})

if __name__ == '__main__':
    app.run(debug=True)
class ProductionConfig:
    ENV = 'production'
    DEBUG = False
    # Add other production-specific configurations
class DevelopmentConfig:
    ENV = 'development'
    DEBUG = True
    # Add other development-specific configurations
# Empty __init__.py file in 'app' and 'config' folders
from flask import Flask
from config import ProductionConfig, DevelopmentConfig
from app.routes import app

if __name__ == '__main__':
    app.config.from_object(ProductionConfig)  # Switch to DevelopmentConfig for local development
    app.run()
Flask==2.0.2
openai==0.27.0
python -m venv venv
source venv/bin/activate  # On Windows, use 'venv\Scripts\activate'
pip install -r requirements.txt
python run.py


