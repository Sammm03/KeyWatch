# ----------------------------
# ANALYTICS DASHBOARD
# ----------------------------
"""
DASHBOARD FEATURES:
1. Real-time data visualization updates (15s intervals)
2. Interactive timeline and anomaly scatter plots
3. Machine learning-powered anomaly detection
4. Adjustable sensitivity levels (Low/Medium/High)
5. Date range filtering for historical analysis
6. Tabbed interface with raw data table view
7. Automatic model retraining every 10 minutes
8. Local data processing - no cloud dependencies
9. Responsive design with fixed header
10. Privacy-focused - no external data sharing
"""
# Dashboard framework
import dash  # Core dash functionality
from dash import html, dcc, dash_table, callback_context  # UI components
from dash.dependencies import Input, Output  # Interactive callbacks

# Visualization
import plotly.express as px  # Interactive graph creation

# Data processing
import pandas as pd  # DataFrame operations on CSV data

# System utilities
import os        # File path operations
import time      # Timestamp comparisons
import logging   # Error logging

# Machine Learning
from sklearn.ensemble import IsolationForest  # Anomaly detection model
from sklearn.preprocessing import MinMaxScaler  # Feature normalization
import plotly.graph_objects as go

# ----------------------------
# DASH APP SETUP
# ----------------------------
app = dash.Dash(__name__)
app.title = "Typing Behavior Analytics Dashboard"

# Configure logging for debugging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/debug.log'),  # Dashboard operation log
        logging.StreamHandler()                 # Console output
    ]
)

# ----------------------------
# GLOBAL SETTINGS
# ----------------------------
DATA_FILE = 'logs/keystrokes.csv'  # Data source
TRAINING_THRESHOLD = 100           # Minimum records to train ML model

global_state = {
    'last_data_hash': None,        # For detecting data changes
    'last_trained': 0,             # Last model training time
    'last_contamination': 0.05,    # Current sensitivity level
    'model': IsolationForest(contamination=0.05),  # Anomaly detection model
    'scaler': MinMaxScaler(),      # Feature normalizer
    'current_data': pd.DataFrame() # Active dataset
}

# ========== LOAD DATA ==========
def load_data():
    """Load and clean typing data from CSV"""
    try:
        if not os.path.exists(DATA_FILE):
            os.makedirs('logs', exist_ok=True)
            with open(DATA_FILE, 'w') as f:
                f.write("timestamp,key_pressed,dwell_time,flight_time\n")
            return pd.DataFrame()

        # Read and parse data
        df = pd.read_csv(DATA_FILE)
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        # Clean numerical fields
        df['dwell_time'] = pd.to_numeric(df['dwell_time'], errors='coerce').fillna(0)
        df['flight_time'] = pd.to_numeric(df['flight_time'], errors='coerce').fillna(0)

        return df.dropna(subset=['timestamp'])  # Remove invalid entries

    except Exception as e:
        logging.error(f"Data load error: {str(e)}")
        return pd.DataFrame()

# ========== TRAIN MODEL ==========
def train_model(data, contamination=0.05):
    """Train/re-train anomaly detection model when needed"""
    try:
        # Check if retraining is required
        data_hash = pd.util.hash_pandas_object(data).sum()
        retrain_needed = (
            data_hash != global_state['last_data_hash'] or
            contamination != global_state['last_contamination'] or
            (time.time() - global_state['last_trained']) > 600  # 10m cooldown
        )
        
        # Skipping redundant training
        if not retrain_needed:
            # logging.info("Skipping redundant training")
            return global_state['model'], global_state['scaler']

        # Only train if we have enough data
        if len(data) < TRAINING_THRESHOLD:
            logging.info("Not enough data for training")
            return global_state['model'], global_state['scaler']

        # Prepare features
        features = data[['dwell_time', 'flight_time']]
        global_state['scaler'] = MinMaxScaler().fit(features)
        scaled_features = global_state['scaler'].transform(features)

        # Train Isolation Forest model
        global_state['model'] = IsolationForest(
            n_estimators=150,       # Number of trees
            contamination=contamination,  # Expected outlier ratio
            random_state=42         # Reproducibility
        ).fit(scaled_features)

        # Update state tracking
        global_state['last_data_hash'] = data_hash
        global_state['last_contamination'] = contamination
        global_state['last_trained'] = time.time()

        logging.info(f"Model retrained on {len(data)} records with contamination {contamination}")
        return global_state['model'], global_state['scaler']
    except Exception as e:
        logging.error(f"Training error: {str(e)}")
        return global_state['model'], global_state['scaler']

# ----------------------------
# USER INTERFACE
# ----------------------------
app.layout = html.Div([
    # Data management
    dcc.Store(id='data-cache'),                     # Shared data storage
    dcc.Interval(id='data-refresh', interval=15_000),  # 15s refresh
    dcc.Interval(id='auto-retrain', interval=600_000), # 10m retrain

    # UI components
    html.H1("Typing Behavior Analytics Dashboard", style={'textAlign': 'center'}),
    html.Div("Keystrokes are stored locally, NO network access.", 
             style={'position': 'fixed',
            'bottom': '10px',
            'left': '15px',
            'fontSize': '1em',
            'color': '#666',
            'backgroundColor': '#ffffff',  # Solid white
            'boxShadow': '0 2px 5px rgba(0,0,0,0.1)',  # Subtle shadow
            'padding': '5px 10px',
            'borderRadius': '4px',
            'zIndex': '1000'}),

    dcc.Tabs([
        # Analysis Tab
        dcc.Tab(label='Behavior Analysis', children=[
            html.Div([
                html.Div([

                    # Date range selector
                    dcc.DatePickerRange(
                        id='date-picker',
                        min_date_allowed=pd.to_datetime('2023-01-01'),
                        max_date_allowed=pd.to_datetime('2030-12-31'),
                        start_date=pd.to_datetime(pd.Timestamp.now() - pd.Timedelta(days=7)),
                        end_date=pd.to_datetime(pd.Timestamp.now()),
                        display_format='MM/DD/YYYY'
                    ),
                    # Sensitivity control
                    dcc.Dropdown(
                        id='sensitivity-preset',
                        options=[
                            {'label': 'Low Sensitivity', 'value': 0.01},    # 1% anomalies
                            {'label': 'Medium Sensitivity', 'value': 0.05}, # 5% anomalies
                            {'label': 'High Sensitivity', 'value': 0.1}     # 10% anomalies
                        ],
                        value=0.05,
                        clearable=False,
                        style={
                               'fontSize': '16px',
                               'width': '200px',
                               'height': '50px',
                               'text-align': 'auto',
                               'marginLeft': '10px'}
                    ),
                    # Reset Button
                    html.Button(
                        '↺ Reset Filters',
                        id='reset-button',
                        style={
                            'fontSize': '16px',
                            'marginLeft': '30px',
                            'padding': '5px 15px',
                            'backgroundColor': '#f8f9fa',
                            'border': '1px solid #ddd'
                        }
                    )
                ], style={'display': 'flex', 'justifyContent': 'left', 'margin': '20px'}),

                # tooltip explanations
                html.Div([
                    html.P(
                        "ℹ️ Dwell Time: Duration a key is held down.  "
                        "Flight Time: Time between releasing one key and pressing the next.",
                        style={
                            'fontSize': '16px',
                            'fontWeight': 'bold',
                            'color': '#666',
                            'margin': '0 20px 10px',
                            'padding': '10px',
                            'backgroundColor': '#f8f9fa',
                            'borderRadius': '5px'
                        }
                    )
                ]),

                # Visualization container
                html.Div([
                    dcc.Graph(id='timeline-graph'),   # Key press timeline
                    dcc.Graph(id='anomaly-graph')     # Anomaly visualization
                ])
            ])
        ]),

        # Data Table Tab
        dcc.Tab(label='Event Logs', children=[
            dash_table.DataTable(
                id='log-table',
                columns=[
                    {'name': 'Timestamp', 'id': 'timestamp'},
                    {'name': 'Key', 'id': 'key_pressed'},
                    {'name': 'Dwell Time(s)', 'id': 'dwell_time'},
                    {'name': 'Flight Time(s)', 'id': 'flight_time'},
                    {'name': 'Status', 'id': 'anomaly'}
                ],
                style_table={'height': '70vh', 'overflowY': 'auto'},
                style_cell_conditional=[
                    {'if': {'column_id': 'timestamp'}, 'width': '240px'},
                    {'if': {'column_id': 'key_pressed'}, 'width': '140px'},
                    {'if': {'column_id': 'dwell_time'}, 'width': '120px'},
                    {'if': {'column_id': 'flight_time'}, 'width': '120px'},
                    {'if': {'column_id': 'anomaly'}, 'width': '100px'}
                ],
                style_cell={'fontSize': '15px', 'textAlign': 'left'},
                style_header={'backgroundColor': '#f1f1f1', 'fontWeight': 'bold', 'position': 'sticky', 'top': 0, 'zIndex': 999},
                filter_action='native', # Built-in filtering
                page_size=50            # Rows per page
            )
        ]) 
    ])
])

# ----------------------------
# INTERACTIVE BEHAVIORS
# ----------------------------
@app.callback(
    Output('data-cache', 'data'),
    [Input('data-refresh', 'n_intervals'), Input('auto-retrain', 'n_intervals')]
)
def update_cache(*_):
    df = load_data()
    global_state['current_data'] = df.copy()
    return df.to_dict('records')

@app.callback(
    [Output('timeline-graph', 'figure'),
     Output('anomaly-graph', 'figure'),
     Output('log-table', 'data')],
    [Input('data-cache', 'data'),
     Input('sensitivity-preset', 'value'),
     Input('date-picker', 'start_date'),
     Input('date-picker', 'end_date')]
)
def update_dashboard(data, sensitivity, start_date, end_date):
    """Update all visual components when data changes"""
    try:
        # Safely convert input data to DataFrame
        df = pd.DataFrame(data) if data else pd.DataFrame()
        
        if df.empty or 'timestamp' not in df.columns:
            return go.Figure(), go.Figure(), []

        # Convert timestamp and filter
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp'])
        
        if start_date and end_date:
            start = pd.to_datetime(start_date).normalize()
            end = pd.to_datetime(end_date).normalize() + pd.Timedelta(days=1)
            df = df[(df['timestamp'] >= start) & (df['timestamp'] < end)]

        # Model training
        model, scaler = train_model(df, contamination=sensitivity)

        # Anomaly detection
        if not df.empty and 'dwell_time' in df.columns and 'flight_time' in df.columns:
            try:
                features = df[['dwell_time', 'flight_time']].fillna(0)
                scaled = scaler.transform(features)
                df['anomaly'] = model.predict(scaled)
                df['anomaly'] = df['anomaly'].map({-1: '🔴 Anomaly', 1: '🟢 Normal'})
            except Exception as e:
                df['anomaly'] = '❓ Unknown'

        # Create visualizations with fallbacks
        try:
            timeline_fig = px.scatter(df, x='timestamp', y='key_pressed', 
                                    title='Keystroke Timeline') if not df.empty else go.Figure()
            
            timeline_fig.update_layout(
                xaxis=dict(
                    title='Timestamp',
                ),
                yaxis=dict(
                    title='Key Pressed',
                )
            )

        except Exception:
            timeline_fig = go.Figure()
            
        try:
            # Create the anomaly figure with custom styling
            anomaly_fig = px.scatter(
                df,
                x='timestamp',
                y='dwell_time',
                color='anomaly',
                title='Typing Behavior Analysis',
                color_discrete_map={
                    '🔴 Anomaly': '#dc3545',  # Bright red for anomalies
                    '🟢 Normal': '#28a745',   # Nice green for normal
                    '❓ Unknown': '#95a5a6'   # Gray for unknown
                }
            )
            
            # Enhanced marker styling
            anomaly_fig.update_traces(
                marker=dict(
                    size=7,
                    opacity=1,
                    line=dict(width=0.1, color='DarkSlateGrey')
                ),
                selector=dict(mode='markers')
            )
            
            # Layout improvements
            anomaly_fig.update_layout(
                xaxis=dict(
                    title='Timestamp',
                ),
                yaxis=dict(
                    title='Dwell Time (seconds)',
                ),
                legend=dict(
                    title='Anomaly Status',
                    orientation='h',
                    yanchor='bottom',
                    y=1.02,
                    xanchor='right',
                    x=1
                )
            )

        except Exception:
            anomaly_fig = go.Figure()

        return timeline_fig, anomaly_fig, df.to_dict('records')

    except Exception as e:
        logging.error(f"Critical callback failure: {str(e)}", exc_info=True)
        return go.Figure(), go.Figure(), []

@app.callback(
    [Output('date-picker', 'start_date'),
     Output('date-picker', 'end_date'),
     Output('sensitivity-preset', 'value')],
    [Input('reset-button', 'n_clicks')]
)
def reset_filters(n_clicks):
    if n_clicks and n_clicks > 0:
        default_start = pd.to_datetime(pd.Timestamp.now() - pd.Timedelta(days=7))
        default_end = pd.to_datetime(pd.Timestamp.now())
        return default_start, default_end, 0.05
    return dash.no_update

# ========== RUN APP ==========
if __name__ == '__main__':
    os.makedirs('logs', exist_ok=True)
    app.run_server(debug=True, host='127.0.0.1', port=8050)
