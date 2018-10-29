from flask import Flask, request, jsonify
import pandas as pd
import model_utils


app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint for generating predictions. It receives a row of data as a
    JSON in the body of the POST request, parses into a DataFrame and sends
    over to the utility functions for preprocessing and generating the
    predictions.
    :return: Output JSON containing the class id, class label, and class
    probabilities, along with a success flag mentioning if the process was
    successful.
    """
    data = {"success": False}
    if request.method == 'POST':
        if request.headers['Content-Type'] == 'application/json':
            df = pd.read_json(request.json)
            data['predictions'] = model_utils.predict(df)
        else:
            data['error'] = 'No JSON received!'
        data["success"] = True
    return jsonify(data)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
