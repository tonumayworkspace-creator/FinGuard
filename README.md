# FinGuard --- Credit Card Fraud Detection System

[![License:
MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Language](https://img.shields.io/badge/Python-3.13-brightgreen)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Flask-Web-yellowgreen)](https://flask.palletsprojects.com/)
[![Modeling](https://img.shields.io/badge/Modeling-scikit--learn-orange)](https://scikit-learn.org/)

FinGuard is an end-to-end fraud detection system built using Flask,
scikit-learn, SHAP explainability, and Plotly visual analytics. This
project demonstrates real-world AI engineering skills with clean UI, API
documentation, and reproducible ML workflows.

------------------------------------------------------------------------

## Quick Demo

Run the app locally and open:

    http://127.0.0.1:5000/

------------------------------------------------------------------------

## What This Project Contains

-   Flask application (`app/`)
-   SHAP explainability UI
-   Plotly ROC visualization
-   Swagger API documentation (`/docs`)
-   Screenshot assets (`docs/images/`)
-   Clean project structure suitable for recruiters

------------------------------------------------------------------------

## Tech Stack

-   Python 3.13\
-   Flask\
-   scikit-learn\
-   SHAP\
-   Plotly\
-   HTML/CSS/JS (modern UI)

------------------------------------------------------------------------

## How It Works

1.  Load dataset (`creditcard-database.csv`)
2.  Train Logistic Regression baseline model
3.  Save model + ROC metrics + SHAP explanations
4.  Provide prediction form + API + visualizations

------------------------------------------------------------------------

## Run Locally

``` bash
git clone https://github.com/tonumayworkspace-creator/FinGuard.git
cd FinGuard
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
pip install shap plotly matplotlib
python app\app.py
```

------------------------------------------------------------------------

## API Documentation (Swagger)

    http://127.0.0.1:5000/docs

Example Request:

``` json
{
  "features": {
    "Time": 34500,
    "Amount": 120.5,
    "V1": -1.23
  }
}
```

------------------------------------------------------------------------

## Screenshots

(Add matching images in docs/images/)

![Home](docs/images/screenshot_home.png)\
![Preview](docs/images/screenshot_preview.png)\
![Train](docs/images/screenshot_train.png)\
![Predict](docs/images/screenshot_predict.png)

------------------------------------------------------------------------

## Author

**Tonumay Bhattacharya**\
Data Science & AI Engineering Enthusiast

GitHub: https://github.com/tonumayworkspace-creator\
LinkedIn: *(add link here)*

------------------------------------------------------------------------

## License

MIT License
