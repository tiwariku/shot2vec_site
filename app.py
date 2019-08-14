'''
shot2vec webapp
@tiwariku
'''
#import base64
#from ast import literal_eval
#import uuid
#from datetime import datetime as dt
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
#import plotly.graph_objs as go
#from numpy.random import randint
#from flask_caching import Cache
from gensim.models import KeyedVectors
import functions as fn
import model_fns as mf
import in_out as io
import data_processing as dp
import corpse_maker as cm
from baseline_models import MarkovEstimator
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras import layers
#from keras.models import load_model
global graph
graph = tf.get_default_graph()


EXTERNAL_STYLESHEETS = ['https://codepen.io/chriddyp/pen/bWLwgP.css',
                        'https://stackpath.bootstrapcdn.com/bootstrap'
                        +'/4.3.1/css/bootstrap.min.css']
CORPUS_FILENAME = '../assets/corpi/full_coords_bin_10'
ASSET_DIR = './assets/coords_bin10/'
MODEL_WEIGHTS = ASSET_DIR+'model_weights.hdf5'
VOCABULARY = dp.unpickle_it(ASSET_DIR+'vocabulary')
PLAY_TO_ID = dp.unpickle_it(ASSET_DIR+'play_to_id')
ID_TO_PLAY = dp.unpickle_it(ASSET_DIR+'id_to_play')
EMBEDDING_DIM = 20
CORPUS_NAME = f'./assets/zones_{EMBEDDING_DIM}'

WV = KeyedVectors.load(f'{CORPUS_NAME}.wv')
EMBEDDING_MATRIX = WV.vectors
CAT_DIM, EMB_DIM = EMBEDDING_MATRIX.shape
HDF5_FILE = f"./assets/RNN_MODEL_SOFT_FIT.h5"
#MODEL_PRED = make_rnn_predicting_model(HDF5_FILE, CAT_DIM, EMB_DIM)

from tensorflow.keras.models import model_from_json
def load_model_json(filename):
    with open(filename, "r") as json_file:
        loaded_model_json = json_file.read()
    return model_from_json(loaded_model_json)

#print(model.predict([0,0,0,0]))

STRIPPER = cm.strip_name_zone
BASE_MODEL_DIR = './assets/markov_zone'
BASE_MODEL = MarkovEstimator()
BASE_MODEL.keys_dict = dp.unpickle_it('./assets/keys_dict')
BASE_MODEL.probs_dict = dp.unpickle_it('./assets/probs_dict')
#BASE_MODEL = dp.unpickle_it(BASE_MODEL_DIR)

TITLE = html.Div(html.Div(children=html.H1(children='shot2vec'),
                          className='col-3 offset-1'
                         ),
                 className='row',
                )

HOCKEY_RINK = html.Div(html.Div(dcc.Graph(id='rink_plot',
                                          figure=fn.make_rink_fig(None),
                                          #style={'width':800,
                                          #       'height':450,
                                          #      },
                                         ),
                                className='col-8 offset-1'
                               ),
                       className='row'
                      )

GET_DATE = dcc.DatePickerSingle(id='date-picker',
                                date=None,
                                #style={'width':200},
                               )
GAME_DROPDOWN = dcc.Dropdown(id='game-dropdown',
                             options=[],
                             #style={'width':200}
                            )

STEP_FORWARD_BUTTOM = html.Button(id='step forward',
                                  children='Next play',
                                  n_clicks=3,
                                  #style={'width':200},
                                 )

BUTTONS = html.Div(children=[html.Div(children=GET_DATE,
                                      className='col-1 offset-1'),
                             html.Div(children=GAME_DROPDOWN,
                                      className='col-4 offset-1'),
                             html.Div(STEP_FORWARD_BUTTOM,
                                      className='col-1 offset-1')
                            ],
                   className='row'
                  )

RECENT_PLAYS = html.Div(
    html.Div(children=dcc.Graph(id='recent-table',
                                figure=fn.serve_recent_plays_table()
                               ),
             className='col-8 offset-1',
            ),
    className='row',
    )

STORE = dcc.Store(id='my-store')
GAME_PLAYS = dcc.Store(id='game-plays', data=None)

DEBUG_OUTPUT = html.Div(id='debug',
                        children='Hi, World')

LAYOUT_KIDS = [TITLE,
               BUTTONS,
               HOCKEY_RINK,
               RECENT_PLAYS,
               DEBUG_OUTPUT, STORE, GAME_PLAYS]
LAYOUT = html.Div(LAYOUT_KIDS, className='contianer')

APP = dash.Dash(__name__, external_stylesheets=EXTERNAL_STYLESHEETS)
SERVER = APP.server
APP.layout = LAYOUT

# callbacks
@APP.callback(Output(component_id='recent-table', component_property='figure'),
              [Input(component_id='game-plays', component_property='data')]
             )
def update_recent_table(plays):
    """
    in: plays, the recent plays list from game_plays stored
    out: new table, wrapped in a figure, for display
    """
    return fn.serve_recent_plays_table(plays)

@APP.callback(Output(component_id='rink_plot', component_property='figure'),
              [Input(component_id='game-plays',
                     component_property='data')]
             )
def update_rink_fig(plays):
    """
    This callback updates the 'rink fig' with the game json, which is stored in
    my-store under property 'data'
    """
    MODEL_PRED = load_model_json('./assets/RNN_PRED.json')
    MODEL_PRED.load_weights("./assets/RNN_PRED.h5")
    goal_probs = None
    if plays:
        #play_str = str(STRIPPER(plays[-1]))
        plays_serial = [str(STRIPPER(play)) for play in plays]
        plays_ind = [WV.vocab[play].index for play in plays_serial]
        preds = MODEL_PRED.predict(plays_ind)[-1, 0, :]
        goal_probs = [preds[27], preds[39], preds[26]]
        #goal_probs = BASE_MODEL.goal_probs(play_str)
    return fn.make_rink_fig(plays, goal_probs)

@APP.callback(Output(component_id='my-store', component_property='data'),
              [Input(component_id='game-dropdown',
                     component_property='value')]
             )
def store_game_json(game_id):
    """
    callback function to store the selected game's response from the NHL API in
    my-store as property 'data' in JSON format
    """
    game_json = io.get_game_response(game_id=game_id).json()
    return game_json

@APP.callback(Output(component_id='game-plays', component_property='data'),
              [Input(component_id='step forward',
                     component_property='n_clicks')],
              state=[State(component_id='my-store',
                           component_property='data')]
             )
def get_current_plays(n_clicks, game_json):
    """
    makes a truncated list of game plays (so far) into store game-plays
    """
    if game_json:
        plays = dp.game_to_plays(game_json, cast_fn=lambda x: x)[:n_clicks]
        return plays
    return None

@APP.callback(Output(component_id='step forward',
                     component_property='n_clicks'),
              [Input(component_id='game-dropdown', component_property='value')])
def reset_stepper(game_id):
    """
    resets to the step forward start of the game when 'get game' is clicked
    """
    start_at = 3
    if game_id:
        return 0*game_id +start_at
    return start_at

@APP.callback(Output(component_id='game-dropdown', component_property='options'),
              [Input(component_id='date-picker', component_property='date')]
             )
def update_dropdown(date):
    """
    Callback for debuging app.. accepts some input and displays it in a Div at
    the bottom of the page
    """
    if date:
        schedule_json = io.get_schedule_json(date)
        return dp.schedule_to_game_list(schedule_json)
    return []

@APP.callback(Output(component_id='debug', component_property='children'),
              [Input(component_id='date-picker', component_property='date')]
             )
def debug_display(date):
    """
    allback for debuging app.. accepts some input and displays it in a Div at
    the bottom of the page
    """
    #    seed_list = [PLAY_TO_ID[str(STRIPPER(play))] for play in data]
    #    return mf.next_probs(seed_list, MODEL_PREDICTING)
    if date:
        schedule_json = io.get_schedule_json(date)
        return str(dp.schedule_to_game_list(schedule_json))
    return 'No date yet'

if __name__ == '__main__':
    APP.run_server(debug=True)
