from flask import Flask, request, jsonify
from inference import load_model, predict

app = Flask(__name__)

model_1 = load_model(1)
model_2 = load_model(2)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json()
        user_input = data.get('input')
        model_type = data.get('model_type')
        if not user_input:
            return jsonify({'error': 'Missing input'}), 400
        if model_type == 1:
            result = predict(model_1, user_input, 1)
        else: result = predict(model_2, user_input, 2)

        return jsonify({'result': result})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
