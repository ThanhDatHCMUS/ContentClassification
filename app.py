from flask import Flask, request, jsonify
from vit5_inference import load_model, predict

app = Flask(__name__)

model = load_model()

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json()
        user_input = data.get('input')
        if not user_input:
            return jsonify({'error': 'Missing input'}), 400

        result = predict(model, user_input)

        return jsonify({'result': result})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
