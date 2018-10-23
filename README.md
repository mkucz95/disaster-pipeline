# Disaster Response Pipeline Project
## Included Files
data/
	- DisasterResponse.db: the sqlite3 databse file that is output by ETL pipeline
    - process_data.py: the python script that runs ETL file
    
models/
	- ada_classifier.pkl: the pickled classifier that was trained and optimized for classifying the dataset messages
    - train_classifier.py: the script that builds, trains, and saves the classifier.
    
app/
	- run.py: the script that runs the flask app
    - templates/
    	- go.html
        - master.html: the default page for the app
        
### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/