import streamlit as st
import pandas as pd
import numpy as np
import base64
import time
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from openai import OpenAI
from dotenv import load_dotenv
import os
import requests
import json

# For RAG capabilities
try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import OpenAIEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain.chains.question_answering import load_qa_chain
    from langchain_community.chat_models import ChatOpenAI
    langchain_available = True
except ImportError:
    st.warning("LangChain not available. PDF question answering will not work. Install with: pip install langchain langchain-openai faiss-cpu")
    langchain_available = False

# --------------------------
# CSS and HTML styles
# --------------------------
USER_BUBBLE_STYLE = "background-color:#DCF8C6;"
ASSISTANT_BUBBLE_STYLE = "background-color:#F1F0F0; margin-left:30px; margin-top:10px;"
STREAMING_BUBBLE_STYLE = "background-color:#F1F0F0; margin-left:30px; margin-top:10px;"
COMMON_BUBBLE_STYLE = """
    padding:10px 15px;
    border-radius:15px;
    width:fit-content;
    max-width:80%;
    margin-bottom:10px;
    box-shadow:1px 1px 5px rgba(0,0,0,0.1);
    text-align:left;
"""

# --------------------------
# Initialize API and App
# --------------------------
load_dotenv()
OPENAI_API_KEY = st.secrets["openai"]["api_key"]
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

st.set_page_config(page_title="HDB Companion", layout="wide")
st.title("🏡 HDB Buying & Selling Companion")
st.header("💬 Ask the Chatbot about HDB trends")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False

# --------------------------
# Data loading (do this at startup)
# --------------------------
@st.cache_data
def load_data():
    try:
        data = pd.read_csv("data_concat.csv", header=0, parse_dates=["month"], low_memory=False)
        st.session_state.data_loaded = True
        return data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Load data at startup
with st.spinner("Loading HDB data..."):
    df = load_data()
    if df is not None:
        st.success("✅ HDB data loaded successfully")

# --------------------------
# Load vector store at startup
# --------------------------
@st.cache_resource
def initialize_vector_db():
    """Load PDF and create vector database for RAG at startup"""
    if not langchain_available:
        return None

    try:
        with st.spinner("Loading knowledge base..."):
            loader = PyPDFLoader("buy_sell_eligibility.pdf")
            data = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400)
            texts = text_splitter.split_documents(data)

            embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
            vectorstore = FAISS.from_documents(texts, embedding=embeddings)

            st.success("✅ Knowledge base loaded successfully")
            return vectorstore
    except Exception as e:
        st.error(f"Error loading vector database: {str(e)}")
        return None

# Initialize vector DB at startup if langchain is available
if langchain_available:
    st.session_state.vector_db = initialize_vector_db()

def plotly_to_base64(fig):
    """Convert a Plotly figure to base64 encoded image"""
    img_bytes = fig.to_image(format="png", width=1000, height=600, scale=2)
    encoded = base64.b64encode(img_bytes).decode("utf-8")
    return encoded

# --------------------------
# Functions for function calling
# --------------------------
def categorize_storey(storey_range):
    try:
        # Extract the first number from strings like "07 TO 09"
        if "TO" in storey_range:
            start = int(storey_range.split("TO")[0].strip())
        else:
            start = int(storey_range)

        if start <= 6:
            return "Low"
        elif 7 <= start <= 12:
            return "Mid"
        else:
            return "High"
    except (ValueError, TypeError):
        # Default to "Mid" if conversion fails
        return "Mid"

def predict_resale_price_via_api(args):
    try:
        # Ensure all required fields are present
        required_fields = ["block", "street_name", "town", "flat_type", "storey_range",
                         "floor_area_sqm", "remaining_lease_year", "month_year"]
        for field in required_fields:
            if field not in args:
                return f"Missing required field: {field}"

        # Convert numeric strings to actual numbers
        if isinstance(args["floor_area_sqm"], str):
            args["floor_area_sqm"] = float(args["floor_area_sqm"])
        if isinstance(args["remaining_lease_year"], str):
            args["remaining_lease_year"] = float(args["remaining_lease_year"])

        # Adjust the storey_range to be 'Low', 'Mid', or 'High'
        args["storey_range"] = categorize_storey(args["storey_range"])

        # Ensure town is uppercase
        if "town" in args:
            args["town"] = args["town"].upper()

        base_url = "https://demodayprediction-28523621686.asia-southeast1.run.app/predict"
        response = requests.get(base_url, params=args)

        if response.status_code == 200:
            return f"S${response.json()['HDB Price']:,.2f}"
        else:
            return f"API Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Error predicting price: {str(e)}"

def plot_priceTrend_single(args):
    """Plot resale price trend for a specific town and flat type using Plotly"""
    try:
        if df is None or not st.session_state.data_loaded:
            return "Error: Data not loaded properly"

        # Extract parameters
        town = args.get("town", "").upper()
        flat_type = args.get("flat_type", "")

        # Validate parameters
        if not town or not flat_type:
            return "Missing required parameters: town and flat_type"

        # Filter data
        df_query = df[
            (df["town"].str.upper() == town) &
            (df["flat_type"].str.upper() == flat_type.upper())
        ]

        if len(df_query) == 0:
            return f"No data found for town: {town} and flat type: {flat_type}"

        # Group by month and calculate average
        df_agg = df_query.groupby("month")["resale_price"].agg(['mean', 'count', 'min', 'max']).reset_index()
        df_agg = df_agg[df_agg['count'] >= 3]  # Only include months with at least 3 transactions

        # Format for better display
        df_agg['year_month'] = df_agg['month'].dt.strftime('%b %Y')
        df_agg['mean_formatted'] = df_agg['mean'].apply(lambda x: f"S${x:,.0f}")

        # Create a nicer Plotly figure
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Add price line
        fig.add_trace(
            go.Scatter(
                x=df_agg['month'],
                y=df_agg['mean'],
                mode='lines+markers',
                name='Average Price',
                line=dict(color='#1f77b4', width=3),
                marker=dict(size=8),
                hovertemplate='%{x|%b %Y}<br>Average: S$%{y:,.0f}<extra></extra>'
            )
        )

        # Add volume bars
        fig.add_trace(
            go.Bar(
                x=df_agg['month'],
                y=df_agg['count'],
                name='Transaction Volume',
                marker_color='rgba(158, 202, 225, 0.6)',
                opacity=0.7,
                hovertemplate='%{x|%b %Y}<br>Transactions: %{y}<extra></extra>'
            ),
            secondary_y=True
        )

        # Add min/max price range
        fig.add_trace(
            go.Scatter(
                x=df_agg['month'],
                y=df_agg['min'],
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            )
        )

        fig.add_trace(
            go.Scatter(
                x=df_agg['month'],
                y=df_agg['max'],
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(31, 119, 180, 0.2)',
                name='Price Range',
                hovertemplate='%{x|%b %Y}<br>Min: S$%{y:,.0f}<extra></extra>'
            )
        )

        # Calculate trendline
        if len(df_agg) > 3:
            x = np.array((df_agg['month'] - df_agg['month'].min()).dt.days)
            y = df_agg['mean'].values
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)

            trendline_y = p(x)
            trendline_color = 'green' if z[0] > 0 else 'red'

            # Add trendline
            fig.add_trace(
                go.Scatter(
                    x=df_agg['month'],
                    y=trendline_y,
                    mode='lines',
                    name='Trend',
                    line=dict(color=trendline_color, width=2, dash='dash')
                )
            )

        # Update layout for better appearance
        fig.update_layout(
            title={
                'text': f"HDB Resale Price Trend: {town} - {flat_type}",
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(size=22)
            },
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            ),
            hovermode="x unified",
            plot_bgcolor='white',
            xaxis=dict(
                title="Date",
                gridcolor='lightgray',
                showgrid=True,
                tickangle=45,
                tickformat='%b %Y',
                tickmode='auto',
                nticks=10
            ),
            yaxis=dict(
                title="Average Resale Price (SGD)",
                gridcolor='lightgray',
                showgrid=True,
                tickformat='S$,.0f'
            ),
            yaxis2=dict(
                title="Number of Transactions",
                overlaying="y",
                side="right",
                showgrid=False
            ),
            height=600,
            width=1000,
            margin=dict(l=60, r=60, b=80, t=100)
        )

        # Get the image in base64 format
        img_base64 = plotly_to_base64(fig)

        # Calculate stats for the report
        latest_price = df_agg.iloc[-1]['mean']
        earliest_price = df_agg.iloc[0]['mean']
        pct_change = ((latest_price - earliest_price) / earliest_price) * 100

        # Create result payload immediately and display the chart
        result = {
            "image": img_base64,
            "analysis": "Generating analysis...",  # Placeholder, will be updated later
            "data_points": len(df_query),
            "town": town,
            "flat_type": flat_type,
            "price_range": f"S${int(df_query['resale_price'].min()):,} - S${int(df_query['resale_price'].max()):,}",
            "pct_change": f"{pct_change:.2f}%",
            "transactions": df_query['month'].dt.year.value_counts().to_dict(),
            "ready_for_analysis": True
        }

        return result
    except Exception as e:
        return f"Error generating plot: {str(e)}"

def analyze_chart(chart_data):
    """Analyze the chart data with GPT-4o (called separately after displaying the chart)"""
    try:
        if not chart_data.get("ready_for_analysis", False):
            return "Chart data not ready for analysis"

        # Create a detailed prompt for better analysis
        prompt = f"""Analyze this HDB resale price trend chart for {chart_data['town']} - {chart_data['flat_type']}.

        Key details:
        - Total transactions: {chart_data['data_points']}
        - Price range: {chart_data['price_range']}
        - Price change: {chart_data['pct_change']}

        Please provide:
        1. Overall trend assessment
        2. Key factors likely influencing these prices
        3. Specific insights for this town and flat type
        4. Brief advice for potential buyers or sellers
        """

        analysis_response = client.chat.completions.create(
            model="gpt-4o",
            temperature=0.2,
            max_tokens=500,
            messages=[
                {"role": "user", "content": prompt},
                {"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{chart_data['image']}"}}]}
            ]
        )

        analysis = analysis_response.choices[0].message.content
        return analysis
    except Exception as e:
        return f"Error analyzing chart: {str(e)}"

def answer_pdf_question(args):
    """Answer questions using the PDF knowledge base"""
    try:
        if not langchain_available:
            return "PDF question answering is not available. Please install langchain with: pip install langchain langchain-openai faiss-cpu"

        query = args.get("query", "")
        if not query:
            return "No question provided"

        if st.session_state.vector_db is None:
            return "Error: Vector database not loaded properly. Please restart the application."

        llm = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
        chain = load_qa_chain(llm, chain_type="map_reduce")
        docs = st.session_state.vector_db.similarity_search(query, k=2)

        answer = chain.run(input_documents=docs, question=query)
        return answer
    except Exception as e:
        return f"Error answering question: {str(e)}"

# --------------------------
# Render chat history
# --------------------------
for msg in st.session_state.messages:
    if msg["role"] == "user":
        cols = st.columns([4, 8])
        with cols[0]:
            st.markdown(
                f"""
                <div style='{USER_BUBBLE_STYLE}{COMMON_BUBBLE_STYLE}'>
                    🧍‍♂️ {msg['content']}
                </div>
                """,
                unsafe_allow_html=True,
            )
    elif msg["role"] == "assistant":
        cols = st.columns([4, 8])
        with cols[0]:
            st.empty()
        with cols[1]:
            st.markdown(
                f"""
                <div style='{ASSISTANT_BUBBLE_STYLE}{COMMON_BUBBLE_STYLE}'>
                    🤖 {msg['content']}
                </div>
                """,
                unsafe_allow_html=True,
            )
    elif msg["role"] == "function":
        cols = st.columns([4, 8])
        with cols[0]:
            st.empty()
        with cols[1]:
            # Check if it is a regular function result or a chart with analysis
            if "image" not in msg:
                st.markdown(
                    f"""
                    <div style='{ASSISTANT_BUBBLE_STYLE}{COMMON_BUBBLE_STYLE}'>
                        🧮 {msg['content']}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div style='{ASSISTANT_BUBBLE_STYLE}{COMMON_BUBBLE_STYLE}'>
                        📊 Chart for {msg['town']} - {msg['flat_type']}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.image(f"data:image/png;base64,{msg['image']}")
                if msg.get('analysis') and msg['analysis'] != "Generating analysis...":
                    st.markdown(
                        f"""
                        <div style='{ASSISTANT_BUBBLE_STYLE}{COMMON_BUBBLE_STYLE}'>
                            📈 Analysis: {msg['analysis']}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )


user_input = st.chat_input("Ask about resale prices, trends, towns, or flat types!")

# --------------------------
# When the user submits a query
# --------------------------
if user_input:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Display user message
    user_cols = st.columns([4, 8])
    with user_cols[0]:
        st.markdown(
            f"""
            <div style='{USER_BUBBLE_STYLE}{COMMON_BUBBLE_STYLE}'>
                🧍‍♂️ {user_input}
            </div>
            """,
            unsafe_allow_html=True
        )

    # Create a placeholder for the assistant's response
    response_cols = st.columns([4, 8])
    with response_cols[0]:
        st.empty()
    with response_cols[1]:
        response_placeholder = st.empty()

    # Prepare messages for the API call
    api_messages = [{"role": m["role"], "content": m["content"]}
                   for m in st.session_state.messages
                   if m["role"] != "function"]

    # Add function results as system messages if they exist
    for m in st.session_state.messages:
        if m["role"] == "function":
            content = m['content']
            if 'analysis' in m and m['analysis'] != "Generating analysis...":
                content += f"\nAnalysis: {m['analysis']}"
            api_messages.append({
                "role": "system",
                "content": f"Function {m.get('name', 'result')}: {content}"
            })

    # Get streaming response from OpenAI
    collected_text = ""
    fc_details = None

    response_stream = client.chat.completions.create(
        model="gpt-4o",
        messages=api_messages,
        functions=[
            {
                "name": "predict_resale_price",
                "description": "Predict HDB resale price using a trained ML model",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "block": {"type": "string"},
                        "street_name": {"type": "string"},
                        "town": {"type": "string"},
                        "flat_type": {"type": "string"},
                        "storey_range": {"type": "string"},
                        "floor_area_sqm": {"type": "number"},
                        "remaining_lease_year": {"type": "number"},
                        "month_year": {"type": "string", "description": "Format YYYY-MM"}
                    },
                    "required": ["block", "street_name", "town", "flat_type", "storey_range",
                                "floor_area_sqm", "remaining_lease_year", "month_year"]
                }
            },
            {
                "name": "plot_priceTrend_single",
                "description": "Plot resale price trend across a single town and flat type",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "town": {"type": "string", "description": "Town in Singapore (e.g., ANG MO KIO, BEDOK)"},
                        "flat_type": {"type": "string", "description": "Flat type (e.g., 3 ROOM, 4 ROOM, 5 ROOM)"},
                    },
                    "required": ["town", "flat_type"],
                }
            },
            {
                "name": "answer_pdf_question",
                "description": "Answer questions using the 'buy_sell_eligibility.pdf' document about HDB buying/selling eligibility",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Question about HDB buying/selling eligibility"}
                    },
                    "required": ["query"]
                }
            }
        ],
        function_call="auto",
        stream=True,
    )

    # Stream the response
    for chunk in response_stream:
        delta = chunk.choices[0].delta

        # Handle regular content
        if hasattr(delta, "content") and delta.content is not None:
            collected_text += delta.content
            response_placeholder.markdown(
                f"""<div style='{STREAMING_BUBBLE_STYLE}{COMMON_BUBBLE_STYLE}'>
                    🤖 {collected_text}
                </div>""",
                unsafe_allow_html=True
            )
            time.sleep(0.02)

        # Handle function call
        if hasattr(delta, "function_call") and delta.function_call is not None:
            if fc_details is None:
                fc_details = {"name": "", "arguments": ""}

            if hasattr(delta.function_call, "name") and delta.function_call.name is not None:
                fc_details["name"] += delta.function_call.name

            if hasattr(delta.function_call, "arguments") and delta.function_call.arguments is not None:
                fc_details["arguments"] += delta.function_call.arguments

    # Save assistant's response to chat history if any plain text was received.
    if collected_text:
        st.session_state.messages.append({
            "role": "assistant",
            "content": collected_text
        })

    # Process function call if one was made
    if fc_details:
        try:
            args = json.loads(fc_details["arguments"])
            function_name = fc_details["name"]

            if function_name == "predict_resale_price":
                result = predict_resale_price_via_api(args)

                # Add function result to chat history
                st.session_state.messages.append({
                    "role": "function",
                    "name": function_name,
                    "content": f"The predicted resale price is {result}"
                })

                # Display function result immediately
                func_result_cols = st.columns([4, 8])
                with func_result_cols[0]:
                    st.empty()
                with func_result_cols[1]:
                    st.markdown(
                        f"""<div style='{ASSISTANT_BUBBLE_STYLE}{COMMON_BUBBLE_STYLE}'>
                            🧮 The predicted resale price is {result}
                        </div>""",
                        unsafe_allow_html=True
                    )

            elif function_name == "plot_priceTrend_single":
                with st.spinner("Generating price trend chart..."):
                    plot_result = plot_priceTrend_single(args)

                if isinstance(plot_result, str):
                    # An error occurred. Save and display it.
                    st.session_state.messages.append({
                        "role": "function",
                        "name": function_name,
                        "content": plot_result
                    })

                    func_result_cols = st.columns([4, 8])
                    with func_result_cols[0]:
                        st.empty()
                    with func_result_cols[1]:
                        st.markdown(
                            f"""<div style='{ASSISTANT_BUBBLE_STYLE}{COMMON_BUBBLE_STYLE}'>
                                ❌ {plot_result}
                            </div>""",
                            unsafe_allow_html=True
                        )
                else:
                    # Successfully generated a chart.
                    function_result = {
                        "role": "function",
                        "name": function_name,
                        "content": f"Price trend chart for {plot_result['town']} - {plot_result['flat_type']}",
                        "image": plot_result["image"],
                        "analysis": "Generating analysis...",
                        "town": plot_result["town"],
                        "flat_type": plot_result["flat_type"],
                        "price_range": plot_result["price_range"],
                        "data_points": plot_result["data_points"],
                        "pct_change": plot_result["pct_change"]
                    }
                    # Record the chart result in chat history.
                    chart_idx = len(st.session_state.messages)
                    st.session_state.messages.append(function_result)

                    func_result_cols = st.columns([4, 8])
                    with func_result_cols[0]:
                        st.empty()
                    with func_result_cols[1]:
                        st.markdown(
                            f"""<div style='{ASSISTANT_BUBBLE_STYLE}{COMMON_BUBBLE_STYLE}'>
                                📊 Chart for {plot_result['town']} - {plot_result['flat_type']}
                            </div>""",
                            unsafe_allow_html=True
                        )
                        st.image(f"data:image/png;base64,{plot_result['image']}")
                        analysis_placeholder = st.empty()
                        analysis_placeholder.markdown(
                            f"""<div style='{ASSISTANT_BUBBLE_STYLE}{COMMON_BUBBLE_STYLE}'>
                                📈 Analysis: Generating analysis...
                            </div>""",
                            unsafe_allow_html=True
                        )

                    # Now process the analysis using your separate function.
                    with st.spinner("Analyzing the chart..."):
                        analysis = analyze_chart(plot_result)

                        # Update the chart entry with the computed analysis.
                        st.session_state.messages[chart_idx]["analysis"] = analysis

                        # Update displayed analysis.
                        analysis_placeholder.markdown(
                            f"""<div style='{ASSISTANT_BUBBLE_STYLE}{COMMON_BUBBLE_STYLE}'>
                                📈 Analysis: {analysis}
                            </div>""",
                            unsafe_allow_html=True
                        )

            elif function_name == "answer_pdf_question":
                with st.spinner("Searching knowledge base..."):
                    answer = answer_pdf_question(args)

                # Save PDF-based answer in chat history and display it.
                st.session_state.messages.append({
                    "role": "function",
                    "name": function_name,
                    "content": answer
                })

                func_result_cols = st.columns([4, 8])
                with func_result_cols[0]:
                    st.empty()
                with func_result_cols[1]:
                    st.markdown(
                        f"""<div style='{ASSISTANT_BUBBLE_STYLE}{COMMON_BUBBLE_STYLE}'>
                            📚 {answer}
                        </div>""",
                        unsafe_allow_html=True
                    )

        except Exception as e:
            error_message = f"Error processing function call: {str(e)}"
            st.error(error_message)
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"I encountered an error: {error_message}"
            })

    # If no function call was made, then proceed with the follow-up response from OpenAI
    else:
        follow_up = client.chat.completions.create(
            model="gpt-4o",
            messages=api_messages,
            stream=False
        )
        follow_up_content = follow_up.choices[0].message.content
        st.session_state.messages.append({
            "role": "assistant",
            "content": follow_up_content
        })

        follow_up_cols = st.columns([4, 8])
        with follow_up_cols[0]:
            st.empty()
        with follow_up_cols[1]:
            st.markdown(
                f"""<div style='{ASSISTANT_BUBBLE_STYLE}{COMMON_BUBBLE_STYLE}'>
                    🤖 {follow_up_content}
                </div>""",
                unsafe_allow_html=True
            )
