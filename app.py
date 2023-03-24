import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd
import seaborn as sns
app = dash.Dash()
df = sns.load_dataset('titanic')

data_nvidia = pd.DataFrame({
    'Category': ['Good', 'Bad', 'Neutral'],
    'Value': [20, 30, 50]
})

data_adobe = pd.DataFrame({
    'Category': ['Good', 'Bad', 'Neutral'],
    'Value': [20, 30, 50]
})

data_iFixIt = pd.DataFrame({
    'Category': ['Good', 'Bad', 'Neutral'],
    'Value': [20, 30, 50]
})

data_microsoft = pd.DataFrame({
    'Category': ['Good', 'Bad', 'Neutral'],
    'Value': [20, 30, 50]
})

# Create a pie chart using Plotly Express
fig_nvidia = px.pie(data_nvidia, values='Value', names='Category')
fig_adobe = px.pie(data_adobe, values='Value', names='Category')
fig_iFixIt = px.pie(data_iFixIt, values='Value', names='Category')
fig_microsoft = px.pie(data_microsoft, values='Value', names='Category')

app.layout = html.Div(children = [
html.H1(children='Nvidia Twitter Sentiment'),
dcc.Graph(id="nvidia", figure=fig_nvidia),
html.H1(children='Adobe Twitter Sentiment'),
dcc.Graph(id="adobe", figure=fig_adobe),
html.H1(children='iFixIt Twitter Sentiment'),
dcc.Graph(id="iFixIt", figure=fig_adobe),
html.H1(children='Microsoft Twitter Sentiment'),
dcc.Graph(id="Microsoft", figure=fig_adobe)])

if __name__ == "__main__":
    app.run_server(debug=True)