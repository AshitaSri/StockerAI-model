from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
import google.generativeai as genai
import yfinance as yf
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app, resources={
    r"/chat": {
        "origins": [
            "http://localhost:3000",
            "https://stocker-ai.vercel.app/",
        ],
        "methods": ["POST", "OPTIONS"],
        "allow_headers": ["Content-Type"],
    }
})

# Configure Gemini
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
model = genai.GenerativeModel('gemini-pro')

def extract_stock_symbol(message):
    """Extract stock symbol from user message using Gemini."""
    try:
        prompt = f"""Extract the stock symbol from this message. If multiple symbols are present, return the main one being asked about. If no valid stock symbol is found, return "NONE". Only return the symbol itself, no other text.

        Message: {message}

        Examples:
        "What do you think about Apple stock?" -> AAPL
        "Compare MSFT and GOOGL" -> MSFT
        "How's the market doing today?" -> NONE
        "Tell me about Tesla's performance" -> TSLA
        """

        response = model.generate_content(prompt)
        symbol = response.text.strip().upper()
        
        # Validate the response
        if symbol == "NONE" or not symbol.isalpha() or len(symbol) > 5:
            return None
            
        return symbol

    except Exception as e:
        print(f"Error in symbol extraction: {str(e)}")
        return fallback_extract_symbol(message)

def fallback_extract_symbol(message):
    """Fallback method for basic symbol extraction."""
    common_symbols = {
        'AAPL': ['AAPL', 'APPLE'],
        'GOOGL': ['GOOGL', 'GOOGLE'],
        'MSFT': ['MSFT', 'MICROSOFT'],
        'AMZN': ['AMZN', 'AMAZON'],
        'META': ['META', 'FACEBOOK'],
        'TSLA': ['TSLA', 'TESLA'],
        'NVDA': ['NVDA', 'NVIDIA']
    }
    
    message_upper = message.upper()
    
    for symbol, variants in common_symbols.items():
        for variant in variants:
            if variant in message_upper:
                return symbol
    
    words = message_upper.split()
    for word in words:
        if word.startswith('$'):
            word = word[1:]
        if word.isalpha() and 1 <= len(word) <= 5:
            return word
    
    return None

def get_stock_data(symbol):
    """Fetch stock data using yfinance."""
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period="1wk")
        info = stock.info
        
        return {
            'current_price': info.get('currentPrice', 'N/A'),
            'previous_close': info.get('previousClose', 'N/A'),
            'market_cap': info.get('marketCap', 'N/A'),
            'volume': info.get('volume', 'N/A'),
            'pe_ratio': info.get('trailingPE', 'N/A'),
            'week_high': hist['High'].max() if not hist.empty else 'N/A',
            'week_low': hist['Low'].min() if not hist.empty else 'N/A',
            'description': info.get('longBusinessSummary', 'N/A')
        }
    except Exception as e:
        return f"Error fetching stock data: {str(e)}"

def generate_analysis(message, stock_data=None):
    """Generate stock analysis using Gemini."""
    try:
        if stock_data:
            prompt = f"""As a stock market expert, analyze this data and answer the question: {message}
            
            Current Stock Data:
            - Current Price: ${stock_data['current_price']}
            - Previous Close: ${stock_data['previous_close']}
            - Market Cap: ${stock_data['market_cap']:,}
            - Volume: {stock_data['volume']:,}
            - P/E Ratio: {stock_data['pe_ratio']}
            - Week High: ${stock_data['week_high']}
            - Week Low: ${stock_data['week_low']}
            
            Company Description:
            {stock_data['description']}
            
            Please provide a concise and informative analysis."""
        else:
            prompt = f"As a stock market expert, please answer this question: {message}"

        response = model.generate_content(prompt)
        return response.text

    except Exception as e:
        return f"Error generating analysis: {str(e)}"

@app.route('/chat', methods=['POST', 'OPTIONS'])
def chat():
    if request.method == 'OPTIONS':
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
        
        # Extract stock symbol and get data
        symbol = extract_stock_symbol(message)
        stock_data = get_stock_data(symbol) if symbol else None
        
        # Generate analysis
        response = generate_analysis(message, stock_data)
        
        return jsonify({
            'response': response,
            'stock_data': stock_data if symbol else None,
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=True)