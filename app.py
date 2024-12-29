from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
import os

app = Flask(__name__)

# Configure CORS for both development and Render deployment
CORS(app, resources={
    r"/chat": {
        # Allow both localhost and your deployed frontend URL
        "origins": [
            "http://localhost:3000",  # Development
            "https://stocker-ai.vercel.app/",  # Replace with your actual frontend URL
        ],
        "methods": ["POST", "OPTIONS"],
        "allow_headers": ["Content-Type"],
        "expose_headers": ["Access-Control-Allow-Origin"],
        "supports_credentials": True
    },
    r"/health": {
        "origins": ["*"],  # Health check can be more permissive
        "methods": ["GET"]
    }
})

# Your existing response patterns
responses = {
    "hello": "Hi there! How can I help you with trading today?",
    "how are you": "I'm doing well, thank you for asking!",
    "bye": "Goodbye! Have a great day!",
    "default": "I'm not sure how to respond to that. Could you rephrase?"
}

def get_response(message):
    if not isinstance(message, str):
        raise ValueError("Message must be a string")
    
    message = message.lower().strip()
    for key in responses:
        if key in message:
            return responses[key]
    return responses["default"]

@app.route('/chat', methods=['POST', 'OPTIONS'])
def chat():
    if request.method == 'OPTIONS':
        # Handle preflight requests
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response
        
    try:
        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 415
            
        data = request.json
        message = data.get('message')
        
        if not message:
            return jsonify({'error': 'Message is required'}), 400
            
        response = get_response(message)
        return jsonify({
            'response': response,
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat()
    }), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)