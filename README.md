# Building an online learning classifier implemented using microservices 

The goal here is three-fold:
- To train a classifier able to provide reliable predictions when the dataset is highly imbalanced (13:1)
- To serve predictions of the model using microservices
- To adapt the classifier to learn incrementally as new data comes in

### File descriptions
- **EDA.ipynb**: Basic EDA performed in jupyter lab
- **model_building.ipynb**: Modelling the classifier and comparing it against a baseline model 
- **initial_train.py**: Script used for training the model with the initial training data
- **model_utils.py**: Script containing utility functions for evaluation and prediction
- **predict_service.py**: Microservice built using python flask for generating predictions and invoking the incremental learning process
- **request_predictions.py**: Script to test the online learning process


### Languages and tools used:
- Python v3
- Tensorflow-gpu
- other packages mentioned in requirements.txt

### Instructions on how to use:
- Copy datasets into project folder (not available in this repo)
- Run the **predict_service.py** script to start the service on localhost port 5000
- Make POST requests with each row converted to json to generate predictions. Alternatively, run the **model_utils.py** script for evaluation.
