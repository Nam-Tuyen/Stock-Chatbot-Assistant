import os
import json
import openai
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')

# Function to get real-time stock news using Finnhub API
def get_stock_news(ticker):
    finnhub_api_key = os.getenv('FINNHUB_API_KEY')
    url = f'https://finnhub.io/api/v1/company-news?symbol={ticker}&from=2023-01-01&to=2024-01-01&token={finnhub_api_key}'
    response = requests.get(url)
    news_data = response.json()
    
    news_articles = []
    if response.status_code == 200:
        for article in news_data[:5]:  # Get the latest 5 news articles
            news_articles.append(f"{article['headline']} - {article['source']}\n{article['url']}")
    else:
        news_articles.append("Could not fetch news.")
    
    return "\n\n".join(news_articles)

# Function to fetch the stock price from Yahoo Finance
def get_stock_price(ticker):
    return str(yf.Ticker(ticker).history(period='1y').iloc[-1].Close)

# Function to calculate the Simple Moving Average (SMA)
def calculate_sma(ticker, window):
    data = yf.Ticker(ticker).history(period='1y').Close
    return str(data.rolling(window=window).mean().iloc[-1])

# Function to calculate the Exponential Moving Average (EMA)
def calculate_ema(ticker, window):
    data = yf.Ticker(ticker).history(period='1y').Close
    return str(data.ewm(span=window, adjust=False).mean().iloc[-1])

# Function to calculate the Relative Strength Index (RSI)
def calculate_rsi(ticker):
    data = yf.Ticker(ticker).history(period='1y').Close
    delta = data.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=14-1, adjust=False).mean()
    ema_down = down.ewm(com=14-1, adjust=False).mean()
    rs = ema_up / ema_down
    return str(100 - (100 / (1 + rs)).iloc[-1])

# Function to calculate the MACD (Moving Average Convergence Divergence)
def calculate_macd(ticker):
    data = yf.Ticker(ticker).history(period='1y').Close
    short_ema = data.ewm(span=12, adjust=False).mean()
    long_ema = data.ewm(span=26, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=9, adjust=False).mean()
    macd_histogram = macd - signal
    return f'{macd[-1]}, {signal[-1]}, {macd_histogram[-1]}'

# Function to plot the stock price
def plot_stock_price(ticker):
    data = yf.Ticker(ticker).history(period='1y')
    plt.figure(figsize=(10, 5))
    plt.plot(data.index, data.Close)
    plt.title(f'{ticker} Stock Price Over Last Year')
    plt.xlabel('Date')
    plt.ylabel('Stock Price ($)')
    plt.grid(True)
    plt.savefig('stock.png')
    plt.close()

# Define available functions and their descriptions
functions = [
    {
        'name': 'getStockPrice',
        'description': 'Gets the latest stock price given the ticker symbol of a company.',
        'parameters': {
            'type': 'object',
            'properties': {
                'ticker': {
                    'type': 'string',
                    'description': 'The stock ticker symbol for a company (e.g., AAPL for Apple).'
                }
            },
            'required': ['ticker']
        }
    },
    {
        'name': 'calculateSMA',
        'description': 'Calculate the simple moving average for a given stock ticker and a window.',
        'parameters': {
            'type': 'object',
            'properties': {
                'ticker': {
                    'type': 'string',
                    'description': 'The stock ticker symbol for a company (e.g., AAPL for Apple).'
                },
                'window': {
                    'type': 'integer',
                    'description': 'The timeframe to consider when calculating the SMA.'
                }
            },
            'required': ['ticker', 'window'],
        }
    },
    {
        'name': 'calculateEMA',
        'description': 'Calculate the exponential moving average for a given stock ticker and a window.',
        'parameters': {
            'type': 'object',
            'properties': {
                'ticker': {
                    'type': 'string',
                    'description': 'The stock ticker symbol for a company (e.g., AAPL for Apple).'
                },
                'window': {
                    'type': 'integer',
                    'description': 'The timeframe to consider when calculating the EMA.'
                }
            },
            'required': ['ticker', 'window'],
        }
    },
    {
        'name': 'calculateRSI',
        'description': 'Calculate the RSI for a given stock ticker.',
        'parameters': {
            'type': 'object',
            'properties': {
                'ticker': {
                    'type': 'string',
                    'description': 'The stock ticker symbol for a company (e.g., AAPL for Apple).'
                }
            },
            'required': ['ticker'],
        }
    },
    {
        'name': 'calculateMACD',
        'description': 'Calculate the MACD for a given stock ticker.',
        'parameters': {
            'type': 'object',
            'properties': {
                'ticker': {
                    'type': 'string',
                    'description': 'The stock ticker symbol for a company (e.g., AAPL for Apple).'
                },
            },
            'required': ['ticker'],
        }
    },
    {
        'name': 'plotStockPrice',
        'description': 'Plot the stock price for the last year given the ticker symbol of a company.',
        'parameters': {
            'type': 'object',
            'properties': {
                'ticker': {
                    'type': 'string',
                    'description': 'The stock ticker symbol for a company (e.g., AAPL for Apple).'
                }
            },
            'required': ['ticker'],
        }
    },
    {
        'name': 'getStockNews',
        'description': 'Gets the latest stock news for a given stock ticker.',
        'parameters': {
            'type': 'object',
            'properties': {
                'ticker': {
                    'type': 'string',
                    'description': 'The stock ticker symbol for a company (e.g., AAPL for Apple).'
                }
            },
            'required': ['ticker'],
        }
    }
]

# Define function mappings
available_functions = {
    'getStockPrice': get_stock_price,
    'calculateSMA': calculate_sma,
    'calculateEMA': calculate_ema,
    'calculateRSI': calculate_rsi,
    'calculateMACD': calculate_macd,
    'plotStockPrice': plot_stock_price,
    'getStockNews': get_stock_news
}

# Initialize Streamlit app
st.title('Stock AI Chatbot Assistant')

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

# Input field for user
user_input = st.text_input('Your input:')

if user_input:
    try:
        st.session_state['messages'].append({'role': 'user', 'content': user_input})

        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo-0613',
            messages=st.session_state['messages'],
            functions=functions,
            function_call='auto'
        )

        response_message = response['choices'][0]['message']
        if response_message.get('function_call'):
            function_name = response_message['function_call']['name']
            function_args = json.loads(response_message['function_call']['arguments'])

            if function_name in ['getStockPrice', 'calculateRSI', 'calculateMACD', 'plotStockPrice', 'getStockNews']:
                args_dict = {'ticker': function_args.get('ticker')}
            elif function_name in ['calculateSMA', 'calculateEMA']:
                args_dict = {'ticker': function_args.get('ticker'), 'window': function_args.get('window')}

            function_to_call = available_functions[function_name]
            function_response = function_to_call(**args_dict)

            if function_name == 'plotStockPrice':
                st.image('stock.png')
            else:
                st.session_state['messages'].append(response_message)
                st.session_state['messages'].append({'role': 'function', 'name': function_name, 'content': function_response})

                second_response = openai.ChatCompletion.create(
                    model='gpt-3.5-turbo-0613',
                    messages=st.session_state['messages']
                )
                st.text(second_response['choices'][0]['message']['content'])
                st.session_state['messages'].append({'role': 'assistant', 'content': second_response['choices'][0]['message']['content']})
        else:
            st.session_state['messages'].append({'role': 'assistant', 'content': response_message['content']})
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Show all messages
for message in st.session_state['messages']:
    if message['role'] == 'user':
        st.write(f"**You:** {message['content']}")
    else:
        st.write(f"**Assistant:** {message['content']}")
