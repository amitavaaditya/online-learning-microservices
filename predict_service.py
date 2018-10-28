from flask import Flask, request, jsonify
import pandas as pd
import model_utils


app = Flask(__name__)


def incremental_train(df):
    model_utils.incremental_train(df)
    print('Incremental training successful!')


@app.route('/predict', methods=['POST'])
def predict():
    data = {"success": False}
    if request.method == 'POST':
        if request.headers['Content-Type'] == 'application/json':
            df = pd.read_json(request.json, lines=True)
            data['predictions'] = model_utils.predict(df)
            incremental_train(df)
        else:
            data['status'] = 'No JSON received!'
        data["success"] = True
    return jsonify(data)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
