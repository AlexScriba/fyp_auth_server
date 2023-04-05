from flask import Flask, jsonify, request
import src.mlp as m


app = Flask(__name__)


@app.route("/", methods=['GET'])
def home():
    return "Hello world"


@app.route("/predict", methods=['POST'])
def get_prediction():
    data = request.get_json()['body']

    print(data.keys())

    (pred, proba_attacker, proba_user) = m.prediction(data)

    return {
        'result': pred,
        'attackerProbability': proba_attacker,
        'userProbability': proba_user
    }


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=3000)
