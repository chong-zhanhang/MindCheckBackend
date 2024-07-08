from flask import Flask, request, jsonify
import pandas as pd
from models import DepressionModelManager
app = Flask(__name__)

model_manager = DepressionModelManager('./model_ann_gscv_regularize.h5', './fine_tuned_models/')

@app.route('/predict', methods=['POST'])
def predict():
    user_id = request.json.get('user_id')
    data = request.json.get('data')
    best_model = request.json.get('best_model')
    preprocessed_data = model_manager.preprocess(data)
    model = model_manager.get_best_model(user_id, best_model)
    prediction = model.predict(preprocessed_data)
    y_pred = (prediction > 0.5).astype(int)
    y_pred = y_pred.flatten().tolist()

    num_ones = sum(y_pred)
    num_zeros = len(y_pred) - num_ones
    aggregated_result = num_ones > num_zeros

    return jsonify({'prediction': aggregated_result})

@app.route('/fine_tune', methods=['POST'])
def fine_tune():
    user_id = request.json.get('userId')
    data = request.json.get('data')
    labels = request.json.get('labels')
    preprocessed_data = model_manager.preprocess(data)
    fine_tuned_accuracy, fine_tuned_f1, base_accuracy, base_f1 = model_manager.update_model(user_id, preprocessed_data, labels)
    return jsonify({'status': 'model updated for user {}'.format(user_id), 'fine_tuned_accuracy':fine_tuned_accuracy, 'fine_tuned_f1':fine_tuned_f1, 'base_accuracy':base_accuracy, 'base_f1':base_f1})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)