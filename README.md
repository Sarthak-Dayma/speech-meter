So i'll make it really easy...

data/raw/ted_talks_en.csv                         is the data set files which contains 4000+ transcripts
data/processed/ted_chunks_with_features.csv       is the file that contains feature extracted 500+ transcript .csv file

start by creating a new virtual environment and installing dependencies

  python3 -m venv env

  source env/bin/activate

  pip install -r requirements.txt

# tf-idf approach

then extract features (there's already extracted)
  python3 run-preprocessing.py

then to evaluate a model 
  python3 run-modeling.py

NOTE::::the project files are in src/ directory, to be careful, i created scripts outside the src folder, to run those codes......

# sentence-transformers
NOTE::::run-embeddings-model.py and src/modelingst.py have different approach using sentence-transformers, hence heavy on resources, if you have a good pc you can try to run these files

NOTE::::to run any of the above approach, please make sure your pc or laptop has capable hardware, i tried on mine and got my fingers burnt :-(, or else wait till eternity, it takes a lot of time, your code editor may chose not to respond.....
