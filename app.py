from flask import Flask, jsonify, request
import src.mlp as m


# Create Flask App
app = Flask(__name__)

# Route handler for /predict route
# Receives data from request and runs authentication module


@app.route("/predict", methods=['POST'])
def get_prediction():
    data = request.get_json()

    # get prediction
    (pred, proba_attacker, proba_user) = m.prediction(data)

    return {
        'result': pred,
        'attackerProbability': proba_attacker,
        'userProbability': proba_user
    }


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=3000)
