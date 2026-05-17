
import pickle
from flask import Flask, request, jsonify

with open('models/lin_reg.bin', 'rb') as f:
    dv, model = pickle.load(f)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    trip = request.get_json()

    features = {
        'PULocationID': str(trip['PULocationID']),
        'DOLocationID': str(trip['DOLocationID']),
        'trip_distance': trip['trip_distance']
    }

    X = dv.transform([features])
    duration = model.predict(X)[0]

    return jsonify({
        'duration_minutes': round(float(duration), 2)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9696, debug=True)
