# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 20:10:29 2021

@author: Preshita
"""

import pandas as pd
import numpy as np
import pickle as pkl
import webbrowser

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

import plotly.graph_objects as go

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

data = pd.read_csv(r"C:\Users\Preshita\Documents\Internship2020\balanced_reviews.csv")

data.isnull().sum() 
data[data.isnull().any(axis=1)] 
data.dropna(inplace=True) 
data = data[data['overall']!=3] 
data['Positivity'] = np.where(data['overall'] > 3, 1, 0) 

labels = ["Positive","Negative"]
values=[len(data[data.Positivity == 1]), len(data[data.Positivity == 0])]
scrape = pd.read_csv(r"C:\Users\Preshita\Documents\Internship2020\reviews.csv")
scrape.head()
 
graph = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])  
app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

def load_model():
    global df
    df = pd.read_csv(r"C:\Users\Preshita\Documents\Internship2020\week2\balanced_reviews.csv")
    
    global  pickle_model
    file = open("pickle_model.pkl","rb")
    pickle_model = pkl.load(file)
    
    global vocab
    file = open("feature.pkl", 'rb') 
    vocab = pkl.load(file)
    
def open_browser():
    webbrowser.open_new('http://127.0.0.1:8050/')
    
def create_app_ui():
    main_layout = html.Div([
            html.Div([html.H1(children='Sentiments Analysis with Insights', id='Main_title',style={'padding':5})],style={'backgroundColor':'#bdb9b9','height': 70}),
            html.Div([html.H1(children='Pie Chart', id='pieh1',style={'padding':5})]),
            dcc.Graph(figure = graph, style={'width': '100%','height':700,},className = 'd-flex justify-content-center'),
            html.Br(),html.Hr(),html.Br(),
            html.Div([html.H1(children='Choose Review', id='dropdown_h1',style={'padding':5})],style={'backgroundColor':'#bdb9b9','height': 60}),
            html.Br(),html.Br(),
            dbc.Container([dbc.FormGroup([dcc.Dropdown(id='dropdown', 
                         options=[{'label': i, 'value': i} for i in scrape.Reviews.unique()], 
                         placeholder='Enter your review...',
                         style={'width': '100%', 'height': 50})])]),
            html.Br(),
            html.H2(children=None, id='result1'),
            html.Br(),html.Hr(),html.Br(),html.Br(),
            html.Div([html.H1(children='Enter Review', id='textarea_h1',style={'padding':5})],style={'backgroundColor':'#bdb9b9','height': 60}),
            html.Br(),html.Br(),
            dbc.Container([dbc.Textarea(id='textarea_review',
                         placeholder='Enter the review here...',
                         value='',
                         style={'width': '100%', 'height': 100})]),
            html.Br(),html.Br(),
            dbc.Button("Find Review", color="info", className="mr-1",id='button_review',n_clicks=0, style={}),
            html.Br(),html.Br(),
            html.H2(children=None, id='result2'),
            html.Br(),
            ],style={'textAlign':'center','backgroundColor':'#e8f6fa','margin-left':20,'margin-right':20})
    return main_layout

@app.callback(
    Output('result1', 'children'),
    [
    Input('dropdown', 'value')
    ]
    )

def update_dropdown_ui(textarea_value):

    result_list = check_review(textarea_value)
    
    if (result_list[0] == 0 ):
        result = dbc.Alert("The review is Negative", color="danger")
    elif (result_list[0] == 1 ):
        result = dbc.Alert("The review is Positive", color="success")
    else:
        result = 'Unknown'
    
    return result

@app.callback(
    Output('result2', 'children'),
    [
    Input('button_review', 'n_clicks'),
    State('textarea_review','value')
    ]
    )
def update_app_ui(n_click,textarea_value):
    result_list = check_review(textarea_value)
    if n_click > 0:
        if (result_list[0] == 0 ):
            result1 = dbc.Alert("The review is Negative", color="danger"),
        elif (result_list[0] == 1 ):
            result1 = dbc.Alert("The review is Positive", color="success"),
        else:
            result1 = 'Unknown'
        
    return result1
    
def check_review(reviewText):
    transformer = TfidfTransformer()
    loaded_vec = CountVectorizer(decode_error="replace",vocabulary=vocab)
    reviewText = transformer.fit_transform(loaded_vec.fit_transform([reviewText]))
    return pickle_model.predict(reviewText)

def main():  
    
    load_model()
    open_browser()
    
    global project_name
    project_name = 'Sentiments Analysis with Insights'
    
    global app
    app.layout = create_app_ui()
    app.title = project_name
    app.run_server()
    
    app = None
    project_name = None
    

if __name__ == '__main__':
    main()