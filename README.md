# Disaster Response Pipeline Project
This project was completed as part of the course requirements of [Udacity's Data Scientist Nanodegree](https://www.udacity.com/course/data-scientist-nanodegree--nd025) certification.

<li><a href="#Installation">Installation</a></li>
<li><a href="#Project Motivation">Project Motivation</a></li>
<li><a href="#File Descriptions">File Descriptions</a></li>
<li><a href="#Instructions">Instructions</a></li>
<li><a href="#Screenshots">Screenshots</a></li>


# Installation:
Python version 3

# Project Motivation:
This project is created to analyze disaster data from [Figure Eight](https://www.figure-eight.com/) to build a model for an API that classifies disaster messages.

# File Descriptions:
 1. data
    - disaster_categories.csv: Dataset that includes all the categories.
    - disaster_messages.csv: Dataset that includes all the messages.
    - process_data.py: ETL pipeline scripts to read, clean, and save data into a database.
    - ETL Pipeline Preparation.ipynb: Jupyter file that includes all ETL pipeline scripts to read, clean, and save data into a database.
    - DisasterResponse.db: The output of the ETL pipeline, i.e. SQLite database containing messages and categories data.
2. models
    - train_classifier.py: machine learning pipeline scripts to train and export a classifier.
    - ML Pipeline Preparation.ipynb: Jupyter file that includes all scripts to train and export a classifier.
3. app
    - run.py: Flask file to run the web application.
    - templates: Contains html file for the web applicatin.

# Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


# Screenshots:
