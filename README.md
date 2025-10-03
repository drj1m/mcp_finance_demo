# MCP Stock Query System

An intelligent stock data retrieval system built on the Model Context Protocol (MCP) that combines AI-powered query understanding with reliable financial data access. The system uses Google's Gemini AI to interpret natural language queries and automatically selects the appropriate tools to fetch stock market information.

## Features

- 🤖 AI-Powered Query Understanding: Uses Google Gemini to interpret natural language stock queries

- 📊 Dual Data Sources: Primary Yahoo Finance API

- 🔄 Automatic Tool Selection: Intelligent mapping of user queries to appropriate stock tools

- 💬 Interactive Chat Interface: Simple command-line interface for natural conversations

- 🛡️ Robust Error Handling: Comprehensive fallback mechanisms and error recovery

- ⚡ Asynchronous Processing: High-performance async operations for better responsiveness

## Architecture

The system consists of two main components:

### MCP Client (mcp_client.py)

- Handles user input and natural language processing

- Connects to the MCP server via stdio communication

- Uses Gemini AI to identify appropriate tools and arguments

- Manages the interactive user session

### MCP Server (mcp_server.py)

- Provides stock data tools through the MCP protocol

- Implements Yahoo Finance API integration

- Exposes two main tools: get_stock_price and compare_stocks

- Handles data source failover automatically

## Installation

### Prerequisites

- Python 3.10 or higher

- Google AI API key (Gemini)

- Internet connection for Yahoo Finance data

### Setup Steps

1. Clone or download the project files
2. Install dependencies:
```
pip install -r requirements.txt
```
3. Configure environment variables (.env):
```
GEMINI_API_KEY=your_gemini_api_key_here
```

## Usage

### Starting the System

1. Run the client:
```
python stock_agent/mcp_client.py
```

2. Enter natural language queries:
```
What is your query? → What's the current price of Apple?
What is your query? → compare stock price of Apple and Microsoft
```

### Example Interactions

#### Single Stock Query:

```
Input: "What's the price of AAPL?"
Output: The current price of AAPL is $150.25 (from Yahoo Finance)
```

#### Stock Comparison:

```
Input: "Compare Apple and Microsoft stocks"
Output: AAPL ($150.25 YF) is lower than MSFT ($380.50 YF).
```

#### Other input examples:

```
"show TSLA history for 1 month"
"company info for NVDA"
"dividends for MSFT last 1y"
"next earnings for AMZN"
"volatility of NFLX for 3 months"
"news for AMD"
"sentiment of Tesla"
"benchmark AMZN vs sp500"
"top movers"
"top gainers"
```

## Configuration

### API Keys

Set your Gemini API key in the .env file:

```
GEMINI_API_KEY=your_actual_api_key_here
```

## Data Sources

### Primary: Yahoo Finance

- Real-time stock data via yfinance library
- Comprehensive market coverage
- Automatic retry mechanisms


## Troubleshooting

### Common Issues

#### "TLS connect error" or "OpenSSL invalid library" when accessing Yahoo Finance:

ERROR Failed to get ticker 'AAPL' reason: Failed to perform, curl: (35) TLS connect
error: error:00000000:invalid library (0):OPENSSL_internal:invalid library (0).


**Cause**: This error occurs when your environment has network restrictions, firewall policies, or SSL/TLS configuration issues that prevent secure connections to Yahoo Finance servers.

**Common scenarios**:
- Corporate networks with strict SSL/TLS policies
- Outdated OpenSSL libraries or certificates
- VPN or proxy configurations blocking financial APIs
- Restricted network environments (institutional, educational)

**Manual resolution**:
- Update your system's OpenSSL libraries
- Configure proxy settings if behind corporate firewall
- Contact your network administrator for API access permissions

#### "Connection error":
- Verify mcp_server.py is in the correct directory
- Check the cwd parameter in mcp_client.py
- Ensure Python is in your system PATH

#### "Could not retrieve price":
- Verify stock symbol is correct
- Check internet connection for Yahoo Finance
- Ensure stocks_data.csv exists and has correct format

#### "API key error":
- Verify GEMINI_API_KEY is set in .env
- Check API key validity and quotas
- Ensure .env file is in the project root

#### Debug Mode

For detailed debugging, check console output which shows:
- Connection status
- Tool identification process
- Data source selection
- Error details
