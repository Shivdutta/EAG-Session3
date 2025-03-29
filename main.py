import os
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Union
from fastapi.middleware.cors import CORSMiddleware

import google.generativeai as genai
from dotenv import load_dotenv

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field, validator

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
GNEWS_API_KEY = os.getenv("GNEWS_API_KEY")
client = genai.GenerativeModel("gemini-2.0-flash")
def get_news_gnews(query,from_date,to_date):
    # params = param.split(",")
    # query = params[0]
    # from_date = params[1]
    # to_date = params[2]

    """Fetch news from GNews API between given dates."""
    url = f"https://gnews.io/api/v4/search?q={query}&from={from_date}&to={to_date}&token={GNEWS_API_KEY}&max=100"
    headers = {"User-Agent": "Mozilla/5.0"}

    #print(f"Requesting URL: {url}")
    #print(f"Using API Key: {GNEWS_API_KEY}")

    response = requests.get(url, headers=headers)
    #print(f"Response Status Code: {response.status_code}")

    if response.status_code != 200:
        print(f"Error Response: {response.text}")
        return pd.DataFrame()  # Return empty DataFrame on error

    data = response.json()
    articles = data.get("articles", [])

    if not articles:
        print("No articles found in API response.")
        return pd.DataFrame()

    news_df = pd.DataFrame(
        [(article.get('publishedAt', '')[:10], article.get('title', ''), article.get('url', '')) for article in articles],
        columns=['Date', 'Title', 'URL']
    )

    #print(f"Fetched {len(news_df)} news articles.")
    return news_df



# Stock Data Fetching Function
def get_stock_data(ticker,from_date,to_date):
    # params=param.split(",")
    # ticker =params[0]
    # from_date=params[1]
    # to_date=params[2]
    """Fetch historical stock data from Yahoo Finance."""
    stock_df = yf.download(ticker, start=from_date, end=to_date)
    stock_df = stock_df[['Close']].reset_index()
    stock_df['Date'] = stock_df['Date'].dt.strftime('%Y-%m-%d')  # Format date
    return stock_df

# Data Analysis Function
def analyze_data(news_df, stock_df):
    """Analyze correlation between news sentiment and stock price movement using Gemini AI."""
    
    # Debugging: Print types to verify the correct data structure
    #print("Type of news_df:", type(news_df))
    #print("Type of stock_df:", type(stock_df))

    # Ensure inputs are DataFrames
    if not isinstance(news_df, pd.DataFrame) or not isinstance(stock_df, pd.DataFrame):
        return "Error: Inputs should be Pandas DataFrames."
    
    analysis_prompt = f"""
    Given the following financial news headlines and stock price movements, analyze the potential correlation.
    News: {news_df}
    Stock Prices: {stock_df}
    Provide a summary of how the news might have influenced stock prices.
    """

    response = client.generate_content(analysis_prompt)

    insight = response.text.strip() if response.text else "Analysis could not be generated."
    return insight


# Function Caller for Routing
import pandas as pd
import json
from io import StringIO

# Store actual DataFrames
stored_data = {
    "news_df": None,
    "stock_df": None
}

def function_caller(func_name, *params):
    function_map = {
        "fetch_news": get_news_gnews,
        "fetch_stock": get_stock_data,
        "analyze_data": analyze_data
    }

    if func_name in function_map:
        if func_name == "analyze_data":
            try:
                # Use stored DataFrames instead of incorrect string values
                news_df = stored_data.get("news_df", pd.DataFrame())
                stock_df = stored_data.get("stock_df", pd.DataFrame())

                return function_map[func_name](news_df, stock_df)
            
            except (ValueError, json.JSONDecodeError) as e:
                return f"Error: Failed to parse JSON data - {str(e)}"
        
        else:
            result = function_map[func_name](*params)
            if func_name == "fetch_news":
                stored_data["news_df"] = result  # Store news DataFrame
            elif func_name == "fetch_stock":
                stored_data["stock_df"] = result  # Store stock DataFrame
            return result

    return f"Function {func_name} not found"


# Financial Agent Function
def financial_agent(stock_name):
    max_iterations = 4
    last_response = None
    iteration = 0
    iteration_response = []

    system_prompt = """You are a financial analysis agent. Respond with EXACTLY ONE of these formats:
    1. FUNCTION_CALL: python_function_name|input
    2. FINAL_ANSWER: [sring]

    where python_function_name is one of the followin:
    1. fetch_news(query,start_date,end_date) and fetch_news returns pandas dataframe by the name news_df
    2. fetch_stock(ticker,start_date,end_date) and fetch_stock returns pandas dataframe by the name stock_df
    3. analyze_data|news_df,stock_df

    Date Rules:
    - `end_date` MUST be today's date in YYYY-MM-DD format.
    - `start_date` MUST be exactly 30 days before `end_date` (YYYY-MM-DD format).
    - Example: If today is 2025-03-28, then:
    - `start_date = 2025-03-28`
    - `end_date = 2025-02-28`


    DO NOT include multiple responses. Give ONE response at a time.


    Execute only one step per iteration and do not skip steps."""


    query = "Analyze the impact of " + str(stock_name) + "'s financial news on its stock price."

    while iteration < max_iterations:
        #print(f"\n--- Iteration {iteration + 1} ---")
        
        if last_response is None:
            current_query = query
        else:
            current_query += "\n\n" + " ".join(iteration_response)
            current_query += " What should I do next?"

        # Generate agent's response
        prompt = f"{system_prompt}\n\nQuery: {current_query}"
        response = client.generate_content(contents=prompt)
        response_text = response.text.strip()
        #print(f"LLM Response: {response_text}")

        # Execute function call
        if response_text.startswith("FUNCTION_CALL:"):
            _, function_info = response_text.split(":", 1)
            func_name, params = [x.strip() for x in function_info.split("|", 1)]
            param_list = params.split(",")

            if response_text.startswith("FUNCTION_CALL:"):
                response_text = response.text.strip()
                #print("response_text",response_text)
                _, function_info = response_text.split(":", 1)
                #print("_",_)
                #print("function_info",function_info)

                func_name, params = [x.strip() for x in function_info.split("|", 1)]
                #print("func_name",func_name)
                #print("params",params)

                # Correctly unpack the parameters before calling the function
                iteration_result = function_caller(func_name, *param_list)
                #print("iteration_result",iteration_result)

            # Check if it's the final answer
            elif response_text.startswith("FINAL_ANSWER:"):
                #print("\n=== Agent Execution Complete ===")
                break

            #print(f" *** Result: {iteration_result}")
            last_response = iteration_result
            iteration_response.append(f"Iteration {iteration + 1}: Called {func_name} with {params}, returned {iteration_result}.")

        iteration += 1
    return iteration_result    


class StockTRequest(BaseModel):
    stock_name: str

# Pydantic Models
class NewsRequest(BaseModel):
    query: str = Field(..., description="Search query for financial news", min_length=1, max_length=100)
    start_date: Optional[str] = Field(None, description="Start date for news search (YYYY-MM-DD)")
    end_date: Optional[str] = Field(None, description="End date for news search (YYYY-MM-DD)")

    @validator('start_date', 'end_date', pre=True, always=True)
    def validate_dates(cls, v, values):
        if v is None:
            today = datetime.now()
            if 'start_date' in values and values['start_date'] is None:
                return (today - timedelta(days=30)).strftime('%Y-%m-%d')
            if 'end_date' in values and values['end_date'] is None:
                return today.strftime('%Y-%m-%d')
        return v

class StockRequest(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol", min_length=1, max_length=10)
    start_date: Optional[str] = Field(None, description="Start date for stock data (YYYY-MM-DD)")
    end_date: Optional[str] = Field(None, description="End date for stock data (YYYY-MM-DD)")

    @validator('start_date', 'end_date', pre=True, always=True)
    def validate_dates(cls, v, values):
        if v is None:
            today = datetime.now()
            if 'start_date' in values and values['start_date'] is None:
                return (today - timedelta(days=30)).strftime('%Y-%m-%d')
            if 'end_date' in values and values['end_date'] is None:
                return today.strftime('%Y-%m-%d')
        return v

class NewsResponse(BaseModel):
    articles: List[dict]
    total_articles: int

class StockPriceResponse(BaseModel):
    prices: List[dict]
    ticker: str

class FinancialAnalysisResponse(BaseModel):
    analysis: str

# Create FastAPI app
app = FastAPI(
    title="Agentic Financial Analysis API",
    description="API for conducting intelligent financial analysis using news and stock data",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Endpoints
@app.post("/news/fetch", response_model=NewsResponse)
async def fetch_financial_news(request: NewsRequest):
    """
    Fetch financial news based on query and date range
    """
    try:
        news_df = function_caller(
            "fetch_news", 
            request.query, 
            request.start_date, 
            request.end_date
        )
        
        return {
            "articles": news_df.to_dict('records'),
            "total_articles": len(news_df)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/stocks/prices", response_model=StockPriceResponse)
async def fetch_stock_prices(request: StockRequest):
    """
    Retrieve historical stock prices for a given ticker
    """
    try:
        stock_df = function_caller(
            "fetch_stock", 
            request.ticker, 
            request.start_date, 
            request.end_date
        )
        
        return {
            "ticker": request.ticker,
            "prices": stock_df.to_dict('records')
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analysis/financial", response_model=FinancialAnalysisResponse)
async def perform_financial_analysis(request: StockTRequest):
    """
    Perform comprehensive financial analysis for a given stock
    """
    try:
        analysis_result = financial_agent(request.stock_name)
        
        return {
            "analysis": analysis_result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Main entry point
if __name__ == "__main__":
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True
    )