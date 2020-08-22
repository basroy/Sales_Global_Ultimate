import pandas as pd
import numpy as np 
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import matplotlib.pyplot as plt
import seaborn as sns
#from IPython.display import display, HTML
#import os
import plotly.offline as py
import plotly.graph_objs as go 
from plotly import tools
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import cufflinks as cf

"""
  We will display stats for Fifa2019 on Jupyter NB and dataframes
  
"""
init_notebook_mode(connected=True)
cf.go_offline()    # connecting jupyter
fifa = pd.read_csv('srcdata/fifa2019data.csv')
print('Categorical columns: ', len(fifa.select_dtypes(include=object).columns))
print('Numerical columns: ', len(fifa.select_dtypes(exclude=object).columns))
# Drop columns
fifa.drop(columns=['Unnamed: 0', 'ID', 'Photo', 'Flag', 'Club Logo', 'Special', 'Real Face', 'Release Clause', 
                   'Joined', 'Contract Valid Until'], inplace=True)
fifa.isnull().sum()[fifa.isnull().sum()>-9000]
#Drop players not belonging to any club
fifa['Club'].fillna(value='No Club', inplace=True)
# Position column values is also available as features, 20+
# For position = GK, the position features are NaN, so fill them with zero
# Can do this conditionally, after transpose of the df, we will skip this exercise for now
#Drop players with no position, and those with position but positiob feature as Null to be filled in as 0
fifa.drop(index=fifa[fifa['Position'].isna()].index, inplace=True)
fifa.fillna(value=0, inplace=True)
#print(fifa.isnull().sum().sum())
#print(fifa)

def currency_convert(vali):
    if vali[-1] == 'M':       #last  character indicates million of 100 of K
        vali = vali[1:-1]    # pick all characters from 2 position exclusing last char
        vali = float(vali)*1000000*1.17
        return vali

    elif vali[-1] == 'K':       #last  character indicates million of 100 of K
        vali = vali[1:-1]    # pick all characters from 2 position exclusing last char
        vali = float(vali)*1000*1.17
        return vali
    else:
        return 0    
        
   
# Columns Value and Wage are in alphanumeric, Convert them to numerical currency , dollars
fifa['Wage USD'] = fifa['Wage'].apply(currency_convert)
fifa['Value USD'] = fifa['Value'].apply(currency_convert)
#fifa['Release Clause USD'] = fifa['Release Clause'].apply(currency_convert)
fifa.drop(columns=['Value', 'Wage'], inplace=True)
# Position Skill columsn such as LW, CW, about 26 such positions, have a numerical operator embedded, let us add it up to a prime number.

def skill_add(val):
    if type(val) == str:
        
        if val[-2] == '+':
            #print(int(val[0:-2]))
            val = int(val[-1]) + int(val[0:-2])
        else:
            val = int(val[0:-3])
        return val
    else:
        return 0            

skill_columns = ['LS', 'ST', 'RS', 'LW', 'LF', 'CF', 'RF', 'RW', 'LAM', 'CAM',
       'RAM', 'LM', 'LCM', 'CM', 'RCM', 'RM', 'LWB', 'LDM', 'CDM', 'RDM',
       'RWB', 'LB', 'LCB', 'CB', 'RCB', 'RB']
for col in skill_columns:
    fifa[col] = fifa[col].apply(skill_add)      
    #print(col)
    #print(fifa[col].head())

def height_in_meter(val):
    if type(val) == str:
       return (int(val[0]) *30.48) + (int(val[2]) * 2.54)
    else:
        return 0

def weight_numerals(val):
    return int(val.split('lbs')[0])

fifa['Height_cms'] = fifa['Height'].apply(height_in_meter)
fifa['Weight_lbs'] = fifa['Weight'].apply(weight_numerals)
fifa.drop(columns=['Height','Weight'], inplace=True)
#print(fifa['Body Type'].unique())
# Incorrect Body Type values, replace them with meaningful ones, instead of player na,es
fifa['Body Type'][fifa['Body Type'] == 'Messi'] = 'Lean'
fifa['Body Type'][fifa['Body Type'] == 'C.Ronaldo'] = 'Normal'
fifa['Body Type'][fifa['Body Type'] == 'Coutois'] = 'Lean'
fifa['Body Type'][fifa['Body Type'] == 'Neymar'] =  'Lean'
fifa['Body Type'][fifa['Body Type'] == 'Shaqiri'] = 'Stocky'
fifa['Body Type'][fifa['Body Type'] == 'Akinfenwa'] = 'Stocky' 
fifa['Body Type'][fifa['Body Type'] == 'PLAYER_BODY_TYPE'] = 'Normal'

# Condense the 26 positions to fit into 4 categories - Forward, Defence, Midfield, Goal
def position_condense(val):
    if val == 'RF' or val == 'ST' or val == 'LF' or val == 'RS' or val == 'LS' or val == 'CF':
        return 'F'
    elif val == 'LW' or val == 'RCM' or val == 'LCM' or val == 'LDM' or val == 'CAM' or val == 'CDM' or val == 'RM' \
         or val == 'LAM' or val == 'LM' or val == 'RDM' or val == 'RW' or val == 'CM' or val == 'RAM':
        return 'M'
    elif val == 'RCB' or val == 'CB' or val == 'LCB' or val == 'LB' or val == 'RB' or val == 'RWB' or val == 'LWB':
        return 'D'
    else:
        return val    
fifa['Position'] = fifa['Position'].apply(position_condense)

# Players in nations
fifa_nations = fifa.groupby(by='Nationality').size().reset_index()
fifa_nations.columns = ['Nation','Count']
#print(fifa_nations)


"""
trace2 = dict(type='chloroplethmapbox', locations=fifa_nations['Nation'],
             z=fifa_nations['Count'],
             locationmode='country names',
             colorscale='Portland'
             )
app.layout = go.Layout(title='<b>Number of players in each country</b>',
                  geo=dict(showocean=True,
                           oceancolor='#AEDFDF',
                           projection=dict(type='natural earth'),
                          )
                  )             
fig = go.Figure(data=[trace2], layout=app.layout)
py.iplot(fig)

"""
bapp = dash.Dash(__name__)
server = bapp.server
Club_Names = fifa.Club.unique()
Club_Names.sort()
#print(Club)
bapp.layout = html.Div([
    html.Div([dcc.Dropdown(id='ClubName', options=[{'label': i, 'value': i} for i in Club_Names],
                           value='No Club'), 
                  ],
                  style={'width': '45%',
                         'display': 'inline-block'}),
    
    dcc.Graph(id='club-player-graph', config={'displayModeBar': False}),
])

@bapp.callback(
    Output('club-player-graph', 'figure'),
    [Input('ClubName', 'value')]
)
def update_graph(iClub):
    if iClub == "No Club":
        fifa_plot = fifa.copy()
    else:
        fifa_plot = fifa[fifa['Club'] == iClub]

    import plotly.express as px
    return px.scatter(fifa_plot, x='Dribbling', y='Name', size='Skill Moves', color='Penalties')

pl_skill = fifa.groupby(by='Position')['Dribbling', 'Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 
                                       'Acceleration', 'SprintSpeed'].mean().reset_index()

trace_a = go.Scatterpolar(theta=['Dribbling', 'Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 
                                       'Acceleration', 'SprintSpeed'
                                ],
                          r=pl_skill[pl_skill['Position'] == 'GK'][['Dribbling', 'Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 
                                       'Acceleration', 'SprintSpeed'
                                                                    ]].values[0],
                          fill='toself',
                          name='Goal Keepers'
                         )
trace_b = go.Scatterpolar(theta=['Dribbling', 'Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 
                                       'Acceleration', 'SprintSpeed'
                                ],
                          r=pl_skill[pl_skill['Position'] == 'F'][['Dribbling', 'Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 
                                       'Acceleration', 'SprintSpeed'
                                                                    ]].values[0],
                          fill='toself',
                          name='Forwards'
                         )
trace_c = go.Scatterpolar(theta=['Dribbling', 'Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 
                                       'Acceleration', 'SprintSpeed'
                                ],
                          r=pl_skill[pl_skill['Position'] == 'D'][['Dribbling', 'Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 
                                       'Acceleration', 'SprintSpeed'
                                                                    ]].values[0],
                          fill='toself',
                          name='Defenders'
                         )
trace_d = go.Scatterpolar(theta=['Dribbling', 'Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 
                                       'Acceleration', 'SprintSpeed'
                                ],
                          r=pl_skill[pl_skill['Position'] == 'M'][['Dribbling', 'Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 
                                       'Acceleration', 'SprintSpeed'
                                                                    ]].values[0],
                          fill='toself',
                          name='Midfielders'
                         )

layout = go.Layout(polar=dict(radialaxis=dict(visible=True, 
                                              range=[0, 100]
                                              )
                             ),
                    showlegend=True,
                    title='bashobi'
                  )
fig = go.Figure(data=[trace_a, trace_b, trace_c, trace_d], layout=layout)
r_app = dash.Dash()
r_app.layout = html.Div([dcc.Graph(figure=fig)])

fifa['Position_n'] = fifa['Position']
fifa['Position_n'][fifa['Position_n'] == 'GK'] = 0
fifa['Position_n'][fifa['Position_n'] == 'D'] = 1
fifa['Position_n'][fifa['Position_n'] == 'M'] = 2
fifa['Position_n'][fifa['Position_n'] == 'F'] = 3

#New dataframe to perform regression for Position based data
fifa_pos = fifa.copy()
fifa_pos.drop(columns=['Name', 'Nationality', 'Club'], inplace=True)

#fifa_pos.drop(columns=['Name', 'Nationality', 'Club', 'Flag', 'Club Logo'], inplace=True)

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_squared_error, r2_score


X = fifa_pos.drop(columns=['Position'])
X = pd.get_dummies(X)
y = fifa_pos['Position']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)
LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True, 
          intercept_scaling=1, max_iter=100, multi_class='warn',
          n_jobs=None, penalty='12', random_state=None, solver='warn',
          tol=0.001, verbose=0, warm_start=False)
prediction = logmodel.predict(X_test)
print(classification_report(y_test, prediction))
print('\n')
print(confusion_matrix(y_test, prediction))
print('\n')
print('Accuracy Score: ', accuracy_score(y_test,prediction))

#Correlation of Position to other features
#fifa_pos.corr().abs()['Position'].sort_values(ascending=False)
fifa_pos.drop(columns=['StandingTackle', 'Potential', 'Age', 'Value USD', 
                     'Jumping', 'Jersey Number', 'Wage USD', 'Overall', 'Marking',
                     'International Reputation', 'Strength', 'Preferred Foot'], inplace=True)

X = fifa_pos.drop(columns=['Position'])
X = pd.get_dummies(X)
y = fifa_pos['Position']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)
LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True, 
          intercept_scaling=1, max_iter=100, multi_class='warn',
          n_jobs=None, penalty='12', random_state=None, solver='warn',
          tol=0.001, verbose=0, warm_start=False)
prediction = logmodel.predict(X_test)
print(classification_report(y_test, prediction))
print('\n')
print(confusion_matrix(y_test, prediction))
print('\n')
print('Accuracy Score: ', accuracy_score(y_test,prediction))

#Detecting outliers via a scatterplot
sns.set_style(style='darkgrid')
plt.rcParams['figure.figsize'] = 12, 8
sns.scatterplot(data=fifa_pos, x='Finishing', y='Positioning', hue='Position', palette='viridis')
#plt.show()

#Remove outliers for each Position
fifa_pos = fifa_pos[~((fifa_pos['Position'] == 'GK') & (fifa_pos['Finishing'] > 20 ) & ( fifa_pos['Positioning'] < 30)) ]
fifa_pos = fifa_pos[~((fifa_pos['Position'] == 'F') & (fifa_pos['Finishing'] < 60 ) & ( fifa_pos['Positioning'] < 40)) ]
fifa_pos = fifa_pos[~((fifa_pos['Position'] == 'M') & (fifa_pos['Finishing'] < 30 ) & ( fifa_pos['Positioning'] < 30)) ]
fifa_pos = fifa_pos[~((fifa_pos['Position'] == 'D') & (fifa_pos['Finishing'] < 20 ) & ( fifa_pos['Positioning'] > 60)) ]
sns.scatterplot(data=fifa_pos, x='Finishing', y='Positioning', hue='Position', palette='viridis')
#plt.show()
#Demonstrated the model using another means. By removing outliers, The Accuracy gets improved. In this set of steps, the accuracy has been
#achieved to be 1 ( Not much clean up can be done :) ).

"""
Getting an error for Gradient Classification
fifa_pos.drop(columns=['Position'], inplace=True)
X = fifa_pos.drop(columns=['Position_n'])
X = pd.get_dummies(X)
y = fifa_pos['Position_n']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
gbclassifier = GradientBoostingRegressor()
gbclassifier.fit(X_train, y_train)
GradientBoostingClassifier(criterion='friedman_mse', init=None,
              learning_rate=0.1, loss='deviance', max_depth=3,
              max_features=None, max_leaf_nodes=None,
              min_impurity_decrease=0.0, min_impurity_split=None,
              min_samples_leaf=1, min_samples_split=2,
              min_weight_fraction_leaf=0.0, n_estimators=100,
              n_iter_no_change=None, presort='auto', random_state=None,
              subsample=1.0, tol=0.0001, validation_fraction=0.1,
              verbose=0, warm_start=False)
prediction =  gbclassifier.predict(X_test)
print(classification_report(y_test, prediction))
print('\n')
print(confusion_matrix(y_test, prediction))
print('\n')
print('Accuracy Score: ', accuracy_score(y_test,prediction))
"""

"""
Section to perform Linear Regression with another metric 'Overall'

"""

fifa_ovr = fifa.copy()
fifa_ovr.drop(columns=['Name', 'Nationality', 'Club'], inplace=True)
X = fifa_ovr.drop(columns=['Overall'])
X = pd.get_dummies(X)
y = fifa_ovr['Overall']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
linmodel = LinearRegression()
linmodel.fit(X_train, y_train)
LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=None)
prediction = linmodel.predict(X_test)
print('RMSE:', np.sqrt(mean_squared_error(y_test, prediction)))
print('r square:', r2_score(y_test, prediction))

# Identify Correlation with 'Overall' and hence drop outliers
print(fifa_ovr.corr().abs()['Overall'].sort_values(ascending=False).head(40))
# Can drop some of these categorical columns, which are less than 0.1 in correlation. 
# We will skip the exercise for now

sns.set_style(style='darkgrid')
plt.rcParams['figure.figsize'] = 12, 8
sns.scatterplot(data=fifa_ovr, x='Reactions', y='Overall', hue='Position', palette='viridis')
#plt.show()

#Remove outliers if needed
fifa_ovr = fifa_ovr[~((fifa_ovr['Reactions'] < 25))]
fifa_ovr = fifa_ovr[~((fifa_ovr['Reactions'] < 35) & (fifa_ovr['Overall'] > 55))]
fifa_ovr = fifa_ovr[~((fifa_ovr['Reactions'] < 35) & (fifa_ovr['Overall'] > 55))]
fifa_ovr = fifa_ovr[~((fifa_ovr['Reactions'] > 62) & (fifa_ovr['Overall'] < 55) & (fifa_ovr['Reactions'] < 70))]
fifa_ovr.drop(fifa_ovr[(fifa_ovr['Reactions'] == 73) & (fifa_ovr['Overall'] == 55)].index, inplace=True)
fifa_ovr.drop(fifa_ovr[(fifa_ovr['Reactions'] == 74) & (fifa_ovr['Overall'] == 59)].index, inplace=True)
fifa_ovr.drop(fifa_ovr[(fifa_ovr['Reactions'] == 79) & (fifa_ovr['Overall'] == 64)].index, inplace=True)
fifa_ovr.drop(fifa_ovr[(fifa_ovr['Reactions'] == 82) & (fifa_ovr['Overall'] == 68)].index, inplace=True)
fifa_ovr.drop(fifa_ovr[(fifa_ovr['Reactions'] == 83) & (fifa_ovr['Overall'] == 70)].index, inplace=True)
fifa_ovr.drop(fifa_ovr[(fifa_ovr['Reactions'] == 84) & (fifa_ovr['Overall'] == 69)].index, inplace=True)
sns.scatterplot(data=fifa_ovr, x='Reactions', y='Overall', hue='Position', palette='viridis')
plt.show()

X = fifa_ovr.drop(columns=['Overall'])
X = pd.get_dummies(X)
y = fifa_ovr['Overall']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
gbregressor = GradientBoostingRegressor()
gbregressor.fit(X_train, y_train)
GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
             learning_rate=0.1, loss='ls', max_depth=3, max_features=None,
             max_leaf_nodes=None, min_impurity_decrease=0.0,
             min_impurity_split=None, min_samples_leaf=1,
             min_samples_split=2, min_weight_fraction_leaf=0.0,
             n_estimators=100, n_iter_no_change=None, presort='auto',
             random_state=None, subsample=1.0, tol=0.0001,
             validation_fraction=0.1, verbose=0, warm_start=False)
prediction = gbregressor.predict(X_test)
print('RMSE:', np.sqrt(mean_squared_error(y_test, prediction)))
print('r square:', r2_score(y_test, prediction))


if __name__ == '__main__':
    r_app.run_server(debug=False)
