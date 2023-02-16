import dash
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import pandas as pd

colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

# -*- coding: utf-8 -*-
from dash import Dash, dcc, html, Input, Output

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
topics = ["Bahrain", "Cyprus", "Israel", "Jordan", "Kuwait", "Lebanon", "Oman", "Palestine",
          "Qatar", "Saudi Arabia", "Syria", "United Arab Emirates"]
# "Egypt", "Iran", "Iraq","Turkey", "Yemen"
import pandas as pd
import plotly as py
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

data = dict(type='choropleth', locations=topics, locationmode='country names', z=list(range(len(topics))), colorbar={'title': 'Country Colours'})
layout = dict(geo={'scope': 'world'})
import plotly.graph_objs as go

chmap = go.Figure(data=[data], layout=layout)



import glob
import os
csv_files = glob.glob("../../data/complete/*.csv")
dfs = {}
# loop through the list and read the CSV files, storing their names and data
for csv_file in csv_files:
    nation = csv_file.split(os.sep)[-1].split('.')[0]
    df = pd.read_csv(csv_file)

    df['pub_date'] = pd.to_datetime(df['pub_date'])
    df['month'] = df['pub_date'].dt.to_period('M').apply(lambda x: x.to_timestamp())

    df['pub_date'] = pd.to_datetime(df['pub_date'],unit='ms')
    df['month'] = df['month']
    bin_labels = ['Negative', 'Neutral', 'Positive']
    df['bins'] = pd.cut(df['sentiment_score'], bins=[-1, -0.5, 0.5, 1], labels=bin_labels)
    dfs[nation] = df

from flask_caching import Cache
app = Dash(__name__, external_stylesheets=external_stylesheets)
cache = Cache()
CACHE_CONFIG= {'CACHE_TYPE':'filesystem','CACHE_DIR':'cache', 'CACHE_DEFAULT_TIMEOUT':3600}
cache.init_app(app.server, config=CACHE_CONFIG)


app.layout = html.Div([
    html.H1(children='Hackathon 4 - Crisis Evolution by Sentiment News Analysis'),  # Create a title with H1 tag
    dcc.Graph(
        id='world-graph',
        figure=chmap
    )  # Display the Plotly figure
    , html.Div(id='click-value'),
    dcc.Dropdown(topics, 'No Nation', id='nations-dropdown'),
    html.Div(id='dd-output-container'),
    dcc.Graph(id='line-mean-graph'),
    dcc.Graph(id='line-count-graph'),
    dcc.Graph(id='freq-graph'),
    dcc.Graph(id='word-graph'),
    dcc.Store(id='signal')
])
@cache.memoize()
def global_store(value):
    if value is None:
        nation = 'Israel'
        value = f'CL_{nation}nyt_complete'

    print('global '+value)
    return dfs[value]

"""@app.callback(
    Output('dd-output-container', 'children'))
def update_output(value):
    print(value+'update')
    return f'You have selected {value}'"""

from src.visualizations.visualizations_2 import *
@app.callback(
    Output('click-value', 'children'),Output('signal', 'data'),
    Input('world-graph', 'clickData'))
def update_click_val(clickData):
    if clickData is not None:
        nation = clickData["points"][0]["location"]
        file_name = f'CL_{nation}nyt_complete'
        nation_df = global_store(file_name)
        print('dim '+str(nation_df.shape))
        return f'You have selected {nation}', \
            nation_df.to_json(orient='records',
                              date_format='epoch')
    return 'You have no nation selected', pd.DataFrame()

@app.callback(Output('line-count-graph', 'figure'),
              Output('line-mean-graph', 'figure'),
              Output('freq-graph', 'figure'),
              Input('signal', 'data'))
def plot_graph(data):
    print(data)
    df = pd.read_json(data)
    print('load '+str(df.shape))
    count_fig = plot_count_linechart(df, '')
    mean_fig = plot_linechart(df, '')
    freq_fig = plot_frequency_linechart(df, '')
    print("figure")
    plt.show()
    return count_fig, mean_fig, freq_fig

if __name__ == '__main__':
    app.run_server(debug=True)

#    iplot( chmap )
