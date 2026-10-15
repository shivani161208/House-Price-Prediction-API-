# House Price Prediction API

A Machine Learning project that predicts house prices based on input features using a trained ML model and a FastAPI backend.

## Project Overview

The House Price Prediction API provides a REST API for making house price predictions. It uses a trained machine learning model to process input features and return a predicted price.

## Features

* Machine Learning-based house price prediction
* FastAPI backend
* REST API endpoints
* Input validation using Pydantic
* Interactive API documentation using Swagger UI
* JSON request and response format

## Tech Stack

* **Language:** Python
* **API Framework:** FastAPI
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-learn
* **Model Storage:** Joblib
* **Server:** Uvicorn

## Project Structure

```text
House-Price-Prediction-API/
├── main.py
├── explore.py
├── requirements.txt
├── house_model.joblib
├── house_features.joblib
├── .gitignore
└── README.md
```

*Note: The trained model files are stored locally and are not included in the GitHub repository because of their large size.*

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/shivani161208/House-Price-Prediction-API-.git
cd House-Price-Prediction-API-
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```cmd
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not available, install the main dependencies:

```bash
pip install fastapi uvicorn pandas numpy scikit-learn joblib
```

### 4. Add the Trained Model Files

Place the required model files in the project directory:

* `house_model.joblib`
* `house_features.joblib`

These files must match the model-loading logic in `main.py`.

### 5. Run the API

```bash
uvicorn main:app --reload
```

The API will run locally at:

`http://127.0.0.1:8000`

## API Documentation

After starting the server, open Swagger UI in your browser:

`http://127.0.0.1:8000/docs`

Use the interactive documentation to inspect available endpoints, provide input data, and test predictions.

## How It Works

1. The user sends house-related input features to the API.
2. FastAPI validates the incoming request.
3. The trained machine learning model processes the features.
4. The API returns the predicted house price as a JSON response.

## Future Improvements

* Deploy the API to a cloud platform.
* Add more comprehensive input validation.
* Build a frontend for users to enter house details.
* Add model performance metrics and automated tests.
* Improve error handling and API monitoring.

## Author

**Shivani**

* GitHub: [shivani161208](https://github.com/shivani161208)

## Disclaimer

Predictions depend on the quality of the training data and the features provided. The output is an estimate and should not be treated as a guaranteed market price.
