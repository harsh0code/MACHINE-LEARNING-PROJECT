# 📱 Smartphone Recommendation System

A web-based application that recommends smartphones based on user preferences like **brand, features, and specifications**.  
Built using **Flask**, **Pandas**, and **scikit-learn**.

---

## 🚀 Features
- 🔎 Search smartphones by **brand** (full or partial name).
- ⚡ Machine learning model for **recommendations**.
- 🌐 Simple and clean **web interface**.
- 🛠 Easily customizable with new datasets and filters.

---

## 🛠 Tech Stack
- **Backend**: Python, Flask  
- **Machine Learning**: scikit-learn, pandas, numpy  
- **Frontend**: HTML, CSS  
- **Environment**: Virtualenv  

---

## 📂 Project Structure
SMARTPHONE_RECOMMENDATION/
│── app.py # Main Flask app
│── smartphoness.csv #data csv file
│── requirement.txt #for dependencies
│── .venv (environment for running the application)
│── requirements.txt # Dependencies
│── templates/ # HTML templates
│ └── index.html
│ └── result.html
│── static/ # CSS and assets
│ └── style.css

##Create a virtual environment

i have face many error like in version of pyhon incompatibile with some libraries so i created a virtual environment which helped in 
in execution of this programme 

1.python -m venv .venv                # command for creating virtual environment 
2..venv\Scripts\Activate.ps1          # command for activating virtual environment 
3.pip install -r requirements.txt     # command for installing all libraries at the same time (recommended )
4.python app.py                       # command for running the programme 

{after  running the programme it will show the warnning and give a local trl host code press ctrl + that link it will take you to the browser}

📊 Dataset

>The dataset contains smartphone specifications like brand, RAM, storage, price, battery, and camera.
>Format: CSV (data/smartphones.csv).


🙌 Acknowledgements

> [Flask](https://flask.palletsprojects.com/)
> [scikit-learn](https://scikit-learn.org/)
> [Pandas](https://pandas.pydata.org/)

