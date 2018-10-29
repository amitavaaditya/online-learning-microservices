from flask import Flask, request, jsonify
import pandas as pd
import model_utils


app = Flask(__name__)


@app.route('/retrain', methods=['POST'])
def retrain():
    """
    API endpoint for re-training the model on new data. It receives a row of
    new data as a JSON in the body of the POST request, parses into a
    DataFrame and sends over to the utility functions for preprocessing and
    invoking the incremental training process.
    :return: Output JSON containing a success flag mentioning if the process
    was successful.
    """
    data = {"success": False}
    if request.method == 'POST':
        if request.headers['Content-Type'] == 'application/json':
            df = pd.read_json(request.json)
            model_utils.incremental_train(df)
        else:
            data['error'] = 'No JSON received!'
        data["success"] = True
    return jsonify(data)


if __name__ == '__main__':
    app.run(debug=True, port=5001)
