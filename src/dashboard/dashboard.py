import dash
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import pandas as pd

colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

def plot_graph():
    df=pd.DataFrame( {
        "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
        "Amount": [4, 1, 2, 2, 4, 5],
        "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
    } )

    fig=px.bar( df, x="Fruit", y="Amount", color="City", barmode="group" )
    fig.update_layout(
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font_color=colors['text']
    )
    return fig

# -*- coding: utf-8 -*-
from dash import Dash, dcc, html, Input, Output

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
topics= ["Bahrain", "Cyprus", "Egypt", "Iran", "Iraq", "Israel", "Jordan", "Kuwait", "Lebanon", "Oman", "Palestine", "Qatar", "Saudi Arabia", "Syria", "Turkey", "United Arab Emirates", "Yemen"]

app = Dash(__name__, external_stylesheets=external_stylesheets)

import pandas as pd
import plotly as py
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

data=dict( type='choropleth', locations=['Iraq', 'Iran', 'Israel'], locationmode='country names',
           text=['IRQ', 'IRN', 'ISR'], z=[1.0, 2.0, 3.0], colorbar={'title': 'Country Colours'} )
layout=dict( geo={'scope': 'world'} )
import plotly.graph_objs as go

chmap=go.Figure( data=[data], layout=layout )


app.layout = html.Div([
    html.H1(children='Hackathon 4 - Crisis Evolution by Sentiment News Analysis'),  # Create a title with H1 tag
    dcc.Graph(
       id='world-graph',
       figure=chmap
   )  # Display the Plotly figure
    ,    html.Div(id='click-value'),
    dcc.Dropdown(topics, 'No Nation', id='nations-dropdown'),
    html.Div(id='dd-output-container'),
    dcc.Graph(id='1-graph'),
    dcc.Graph(id='2-graph')
])

@app.callback(
    Output('dd-output-container', 'children'),
    Input('nations-dropdown', 'value'))
def update_output(value):
    return f'You have selected {value}'

@app.callback(
    Output('click-value', 'children'),
    Input('world-graph', 'clickData'))
def update_click_val(clickData):
    if clickData is not None:
        return f'You have selected {clickData["points"][0]["location"]}'

if __name__ == '__main__':
    app.run_server(debug=True)

#    iplot( chmap )
