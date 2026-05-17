
import pickle
import pytest
import numpy as np

with open('models/lin_reg.bin', 'rb') as f:
    dv, model = pickle.load(f)

def predict_duration(trip):
    features = {
        'PULocationID': str(trip['PULocationID']),
        'DOLocationID': str(trip['DOLocationID']),
        'trip_distance': trip['trip_distance']
    }
    X = dv.transform([features])
    return float(model.predict(X)[0])

# Test 1: output is a float
def test_predict_returns_float():
    trip = {'PULocationID': 65, 'DOLocationID': 233, 'trip_distance': 6.2}
    result = predict_duration(trip)
    assert isinstance(result, float)

# Test 2: output is a positive number
def test_predict_positive_duration():
    trip = {'PULocationID': 65, 'DOLocationID': 233, 'trip_distance': 6.2}
    result = predict_duration(trip)
    assert result > 0

# Test 3: different locations give different durations
def test_different_locations_different_duration():
    trip_a = {'PULocationID': 65,  'DOLocationID': 233, 'trip_distance': 6.2}
    trip_b = {'PULocationID': 100, 'DOLocationID': 50,  'trip_distance': 6.2}
    assert predict_duration(trip_a) != predict_duration(trip_b)

# Test 4: duration is within realistic range
def test_predict_realistic_range():
    trip = {'PULocationID': 65, 'DOLocationID': 233, 'trip_distance': 6.2}
    result = predict_duration(trip)
    assert 1 <= result <= 120
