import dash
from dash import dcc,html,Input,Output,State
import plotly.express as px
import pandas as pd

app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "Stonk AI"

sample_dates = pd.date_range(start="2024-11-01", periods=30)
sample_df = pd.DataFrame({
    "date": sample_dates,
    "sentiment": [0.1, 0.15, -0.05, 0.2, 0.1, -0.1, 0.05, 0.1, -0.05, 0.15]*3,
    "close_price": [200 + i for i in range(30)],
})

sample_fig = px.line(sample_df, x="date", y="sentiment", title="Sample Sentiment Trend")

app.layout = html.Div([
    html.H1("Stonk AI: Stock Prediction Dashboard", style= {'textAlign': 'center'}),
    dcc.Tabs([
        dcc.Tab(label='Overview', children=[
            html.H3("Raw Reddit Data"),
            html.P("This section will show the aggregated posts, comments, and stock data"),
            dcc.Dropdown(
                id='ticker-dropdown',
                options=[
                    {'label': 'AAPL', 'value': 'AAPL'},
                    {'label': 'TSLA', 'value': 'TSLA'},
                    {'label': 'NVDA', 'value': 'NVDA'}
                ],
                placeholder='Select ticker',
            ),
            dcc.Graph(figure = sample_fig)
        ]),

        dcc.Tab(label='Sentiment Trends', children=[
            html.H3("Reddit Sentiment Over Time"),
            html.P("This will visualize daily mean sentiment, positive/negative ratios, and engagement levels."),
            dcc.Graph(id='sentiment-graph', figure=sample_fig)
        ]),

        dcc.Tab(label= 'Model Training', children=[
            html.H3("Model Input Pipeline"),
            html.P("We will put more info about the LSTM model")
        ]),

        dcc.Tab(label='Predictions & Evaluation', children=[
            html.H3("Model Outputs"),
            html.P("Click the button below to run the prediction model."),

            html.Button("Run Model", id="run-model-btn", n_clicks=0, style={"margin": "10px"}),

            html.Div(id="model-output", children=[
                html.P("Next week direction: ---", id="direction-text"),
                html.P("Magnitude: ---", id="magnitude-text"),
                html.P("Confidence: ---", id="confidence-text"),
            ]),

            dcc.Graph(
                id='predictions-graph',
                figure=px.line(sample_df, x="date", y="close_price", title="Predicted vs Actual Stock Price")
            )
        ])
    ])
])

@app.callback(
    [
        Output("direction-text", "children"),
        Output("magnitude-text", "children"),
        Output("confidence-text", "children")
    ],
    Input("run-model-btn", "n_clicks")
)
def run_model(n_clicks):
    if n_clicks == 0:
        return (
            "Next week direction: ---",
            "Magnitude: ---",
            "Confidence: ---"
        )

    # I'll plug in actual model here when we have it
    # Mock example for now:
    predicted_direction = "Up"
    predicted_magnitude = "+2.4%"
    predicted_confidence = "88%"

    return (
        f"Next week direction: {predicted_direction}",
        f"Magnitude: {predicted_magnitude}",
        f"Confidence: {predicted_confidence}"
    )

if __name__ == '__main__':
    app.run(debug=True)