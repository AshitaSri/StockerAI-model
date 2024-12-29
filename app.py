from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
# In production, replace with your Vercel frontend URL
CORS(app, origins=['*'])  

# Your existing response patterns
responses = {
    "hello": "Hi there! How can I help you with trading today?",
    "how are you": "I'm doing well, thank you for asking!",
    "bye": "Goodbye! Have a great day!",
    "default": "I'm not sure how to respond to that. Could you rephrase?"
}

def get_response(message):
    message = message.lower()
    for key in responses:
        if key in message:
            return responses[key]
    return responses["default"]

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        message = data.get('message', '')
        response = get_response(message)
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)