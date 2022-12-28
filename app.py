
from dash import html, Dash, dcc, Input, Output, State

import plotly.graph_objects as go
import pandas as pd
import numpy as np
import plotly.express as px
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

# data prepareation
df=pd.read_csv("sample_data.csv")
df["date"]=df["date"].apply(lambda x: pd.to_datetime(x, format='%m/%d/%Y', errors='ignore'))
now =  dt.datetime(2022,12,27)
df["date"]=df["date"].apply(lambda x: pd.to_datetime(x, format='%m/%d/%Y', errors='ignore'))
rfm = df.groupby('user_id').agg({'date': lambda date: (now - date.max()).days,
                                 'order_id': lambda num: len(num),
                                 'total_purchase': lambda total_purchase: total_purchase.sum()

                                 })
col_list = ['Recency', 'Frequency', 'Monetary']
rfm.columns = col_list

rfm.to_csv("rfm.csv", index=False)
#############


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)
def get_day_name(x):
    d={5:'شنبه',
        6:"یکشنبه",
        0:"دوشنبه",
        1:"سه شنبه",
        2:"چهارشنبه",
        3:"پنجشنبه",
        4:"جمعه"}
    return d.get(x,None)

def day_sort(x):
    d = {'شنبه': 0,
         "یکشنبه": 1,
         "دوشنبه": 2,
         "سه شنبه": 3,
         "چهارشنبه": 4,
         "پنجشنبه": 5,
         "جمعه": 6}
    return d.get(x,None)


def stat():
    df = pd.read_csv("sample_data.csv")
    f = df.groupby("date").size().reset_index(name='Size')
    f["date"] = f["date"].apply(lambda x: pd.to_datetime(x, format='%m/%d/%Y', errors='ignore'))
    f["weekday"]=f["date"].apply(lambda x: get_day_name(x.weekday()))
    g = f[["Size", "weekday"]].groupby("weekday").agg([np.mean, np.std]).reset_index()
    g.columns = [
        "روز هفته",
        "میانگین تقاضای روزانه",
        "انحراف معیار تقاضای روزانه",
    ]
    g.index = [day_sort(i) for i in g["روز هفته"]]
    g=g.sort_index()
    fig = go.Figure(data=[go.Table(
        header=dict(values=list(g.columns),
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[g[i] for i in g],
                   fill_color='lavender',
                   align='left'))
    ])
    return fig

def day_type(date):
    w=date.weekday()
    if w in [3,4]:
        return "Weekend"
    return "WorkingDay"

def hist():
    df = pd.read_csv("sample_data.csv")
    f = df.groupby("date").size().reset_index(name='Size')
    f["date"] = f["date"].apply(lambda x: pd.to_datetime(x, format='%m/%d/%Y', errors='ignore'))
    f["day_type"] = f.date.apply(lambda x: day_type(x))
    fig = px.histogram(f, x="Size", color="day_type", barmode="overlay", )
    fig.update_layout(
        title="Histogram of Demand",
        legend_title="",
        xaxis_title="Demand",
        yaxis_title=""
    )
    return fig


app.layout = html.Div(children=[
    # All elements from the top of the page
    html.Div([
        html.Div([



            dcc.Graph(
                id='graph1',
                figure=hist()
            ),
        ], className='six columns'),
        html.Div([

            dcc.Graph(
                id='graph2',
                figure=stat()
            ),
        ], className='six columns'),
    ], className='row'),
    # New Div for all elements in the new 'row' of the page
    html.Div([
        html.Div([

            html.Div(children='''
    Enter K
'''),
        html.Div(dcc.Input(placeholder='Enter number of clusters ...', id='clust_value', type='number', value=2)),
        html.Button(id='Set-val', n_clicks=0, children='Set'),
        dcc.Graph(
            id='dash_graph'
        ),

        ], className='six columns'),
        html.Div([


            dcc.Graph(
                id='dash_table'
            ),
        ], className='six columns'),
    ], className='row'),

])



@app.callback(
    Output('dash_graph', 'figure'),
    Output('dash_table', 'figure'),
    State('clust_value', 'value'),
    Input('Set-val', 'n_clicks'),

)
def update_value(clust_value, *_):
    # check if a button was triggered, if not then just render both plots with 0


    named_colorscales = px.colors.named_colorscales()
    rfm = pd.read_csv("rfm.csv")
    scaler = MinMaxScaler()
    normal = pd.DataFrame(scaler.fit_transform(rfm))
    normal.columns = rfm.columns

    X = normal.to_numpy()
    kmeans = KMeans(n_clusters=clust_value, random_state=0, ).fit(X)
    rfm["color_labels"] = [named_colorscales[i] for i in kmeans.labels_]
    rfm["labels"] = [i+1 for i in kmeans.labels_]
    df = rfm.sample(frac=0.01, replace=True, random_state=1)  # iris is a pandas DataFrame
    fig = px.scatter(df, y="Recency", x="Frequency", color="color_labels")
    # size='petal_length', hover_data=['petal_width'])
    fig.update_layout(showlegend=False)

    f = rfm.groupby("labels").agg([np.mean])
    f = f.reset_index()
    f["labels"] = ["خوشه " + str(i) for i in f.labels]
    f.columns = ['گروه مشتریان', 'Ave.R', 'Ave.F', 'Ave.M']
    fig1 = go.Figure(data=[go.Table(
        header=dict(values=list(f.columns),
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[f[i] for i in f],
                   fill_color='lavender',
                   align='left'))
    ])


    return [fig, fig1]


#app.layout = dcc.Graph(figure=hist()) #dash_table.DataTable(g.to_dict('records'), [{"name": i, "id": i} for i in g.columns])

if __name__ == '__main__':
    app.run_server(debug=True)