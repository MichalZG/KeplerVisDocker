import dash_html_components as html
import dash_core_components as dcc
import dash
import logging

from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import redis
import json
import io

import pandas as pd
from itertools import chain
import urllib.parse as urllib
from datetime import datetime
import time
import os
from flask_caching import Cache
from utils import (timeit, fit_function, create_shadow_shape,
                   create_function_plot, create_line,
                   StateRecorder, load_button_times, config, open_upload_file,
                   logger, FitRecorder)

pd.options.mode.chained_assignment = None

app = dash.Dash()

try:
    cache_redis_url = os.environ['CACHE_REDIS_URL']
except KeyError:
    cache_redis_url = config.get('DB', 'CACHE_REDIS_URL')

CACHE_CONFIG = {
    'CACHE_TYPE': config.get('DB', 'CACHE_TYPE'),
    'CACHE_REDIS_URL': cache_redis_url}

log = logging.getLogger('werkzeug')
log.setLevel(logging.WARNING)

cache = Cache()
cache.init_app(app.server, config=CACHE_CONFIG)
cache.clear()

stateRecorder = StateRecorder()
FitRecorder = FitRecorder()

app.scripts.config.serve_locally = True
app.config['suppress_callback_exceptions'] = True

scattergl_limit = config.getint('ZOOM_GRAPH', 'POINTS_LIMIT')

sf = []
start_date_int = None
fit_func = None
confirmed_fit_func = None
shadow_shape = None
zoomStartPoint = None
refPoint_x = None
fit_start_value_x = None
fit_end_value_x = None
refPoint_y = None
fit_start_value_y = None
fit_end_value_y = None
fileName = None
sf_trigger = 1

# df = pd.DataFrame(
#     data=dict(
#         [(name, []) for name in config.get(
#             'FILES', 'COLUMNS_NAMES').split(',')])
# )


# LAYOUT
###############################################################################


app.layout = html.Div([
    html.Br(),
    html.Div([
        html.Div([
            html.Div([
                dcc.Upload(id='upload-file',
                           children=html.Button('Upload File')),
                html.Div(id='upload-file-name', children=None)
                ],
                     className='upload-button'),
            dcc.Input(
                id='input-binning-full',
                placeholder='Enter a value...',
                type='number',
                value=0,
                min=0,
                max=config.getint('INPUTS', 'BINNING_MAX')),
            html.Button('Set binning', id='set-binning-full-button',
                        n_clicks=0, n_clicks_timestamp=0)]),

    ]),

    dcc.Location(id='location2', refresh=True),
    html.Div(id='all-data-mean-text', children=0),
    html.P('Points between start-end'),
    html.Div(id='n-points-day', children=0),
    dcc.Graph(id='all-data-graph', style={
        'height': config.getint('MAIN_GRAPH', 'HEIGHT')}),



    html.Div([
        html.Div([
            html.Button('Zoom point', id='zoom-point-button', n_clicks=0,
                        n_clicks_timestamp=0),
            html.P(id='zoom-point-x-text', children='---'),
            # html.P(id='zoom-point-y-text', children='---'),
            html.Div([
                dcc.Dropdown(
                    id='saved-states-list',
                    options=[])]),
            html.Div([
                html.Button('Load', id='load-state-button', n_clicks=0,
                            n_clicks_timestamp=0),
                html.Button('Save', id='save-output-button', n_clicks=0,
                            n_clicks_timestamp=0),
            ], className='fit-states-box'),
            html.Div([
                dcc.Dropdown(
                    id='save-format',
                    options=[{'label': 'csv', 'value': 'csv'},
                             {'label': 'txt', 'value': 'txt'}],
                    value='csv',
                    placeholder="Select format"
                    ),
                dcc.Checklist(
                    id='ppt',
                    options=[
                        {'label': 'ppt', 'value': 'ppt'}
                    ],
                    values=[],
                    labelStyle={'display': 'inline-block'}),
            ], className='fit-states-box')

        ]),

        html.Div([
            html.Button('Ref point', id='fit-ref-point-button', n_clicks=0,
                        n_clicks_timestamp=0),
            html.Button('Start', id='fit-start-value-button', n_clicks=0,
                        n_clicks_timestamp=0),
            html.Button('End', id='fit-end-value-button', n_clicks=0,
                        n_clicks_timestamp=0),
            html.Button('Refresh', id='refresh-graph-button', n_clicks=0,
                        n_clicks_timestamp=0),
            html.Br(),
            dcc.Input(
                id='fit-ref-point-x',
                placeholder='Ref point x',
                type='str',
                value=None
            ),
            dcc.Input(
                id='fit-start-value-x',
                placeholder='Start point x',
                type='str',
                value=None
            ),
            dcc.Input(
                id='fit-end-value-x',
                placeholder='End point x',
                type='str',
                value=None
            ),
            html.Br(),
            dcc.Input(
                id='fit-ref-point-y',
                placeholder='Ref point y',
                type='str',
                value=None
            ),
            dcc.Input(
                id='fit-start-value-y',
                placeholder='Start point y',
                type='str',
                value=None
            ),
            dcc.Input(
                id='fit-end-value-y',
                placeholder='End point y',
                type='str',
                value=None
            ),
        ], className='fit-states-box'),
        html.Div([
            html.P('Fit function'),
            dcc.Dropdown(
                id='fit-function-type',
                options=[
                    {'label': 'Moving Average P', 'value': 'movingaverage_p'},
                    {'label': 'Moving Average T', 'value': 'movingaverage_t'},
                    {'label': 'Spline', 'value': 'spline'},
                    {'label': 'Parabola', 'value': 'parabola'},
                    {'label': 'Line', 'value': 'line'},
                    {'label': 'Shift', 'value': 'shift'}
                ],
                value='line',
                clearable=False
            ),
            html.Br(),
            html.Div([
                html.Div([
                    html.P(id='fit-function-parameter-text',
                           children='Parameter:'),
                    dcc.Input(
                        id='input-function-parameter',
                        placeholder='Enter a value...',
                        type='text',
                        value=config.getint('INPUTS', 'PARAM_1_DEFAULT'),
                        min=config.getint('INPUTS', 'PARAM_1_MIN'),
                        max=config.getint('INPUTS', 'PARAM_1_MAX'),
                        step=config.getfloat('INPUTS', 'PARAM_1_STEP'),
                        disabled=True)], className='parameter-box'),
                html.Div([
                    html.P(id='fit-function-parameter-text2',
                           children='Parameter:'),
                    dcc.Input(
                        id='input-function-parameter2',
                        placeholder='Enter a value...',
                        type='number',
                        value=config.getint('INPUTS', 'PARAM_2_DEFAULT'),
                        min=config.getint('INPUTS', 'PARAM_2_MIN'),
                        max=config.getint('INPUTS', 'PARAM_2_MAX'),
                        step=config.getfloat('INPUTS', 'PARAM_2_STEP'),
                        disabled=True)], className='parameter-box')],
                     className='parameters-box'),
            html.Div([
              html.Button('Fit', id='fit-button', n_clicks=0,
                          n_clicks_timestamp=0),
              html.Button('Confirm', id='fit-confirm-button',
                          n_clicks=0, n_clicks_timestamp=0),
              html.Button('Save', id='fit-save-button',
                          n_clicks=0, n_clicks_timestamp=0),
              html.Button('Clear', id='fit-clear-button', n_clicks_timestamp=0)],
              className='fit-box')
        ])
    ],
        style={}, className='button-box'),

    html.Hr(),
    html.Div([
    html.Div([
        dcc.Input(
            id='input-binning-zoom',
            placeholder='Enter a value...',
            type='number',
            value=0,
            min=0,
            max=config.getint('INPUTS', 'BINNING_MAX')),
        html.Button('Set binning', id='set-binning-zoom-button', n_clicks=0,
                    n_clicks_timestamp=0),
    html.Button('Delete', id='delete-button', n_clicks=0,
                n_clicks_timestamp=0),
    ])], className='zoom-data-control-box'),

    dcc.Location(id='location3', refresh=True),

    # ZOOM Data Grap
    dcc.Graph(id='zoom-data-graph', style={
        'height': config.getint('ZOOM_GRAPH', 'HEIGHT')}),



    html.Div(id='all-data', children='value', style={'display': 'none'}),
    html.Div(id='delete-data', children=0, style={'display': 'none'}),
    html.Div(id='reload-data', children=0, style={'display': 'none'}),
    html.Div(id='fit-confirmed', children=0, style={'display': 'none'}),
    html.Div(id='data-modification-trigger', children=0,
             style={'display': 'none'}),
    html.Div(id='upload-temp', children=None,
             style={'display': 'none'}),
    html.Div(id='fitted-function-temp',
             style={'display': 'none'}),
    html.Div(id='fitted-function-temp2',
             style={'display': 'none'}),
    html.Div(id='fitted-function-temp3',
             style={'display': 'none'}),
    html.Div(id='fitted-function-temp4',
             style={'display': 'none'}),
    html.Div(id='save-state-temp',
             style={'display': 'none'}),
    html.Div(id='load-state-temp',
             style={'display': 'none'}),
    html.Div(id='save-output-temp',
             style={'display': 'none'}),
    html.Div(id='zoom-graph-relayout-temp', children=None,
             style={'display': 'none'}),
    html.Div(id='refresh-graph-button-temp',
             style={'display': 'none'}),
    html.Div(id='clear-data', children='value', style={'display': 'none'}),
    html.Div(id='buttons-times', children=None,
             style={'display': 'none'}),
])


###############################################################################
# CALLBACKS
###############################################################################

@app.callback(
    Output('buttons-times', 'children'),
    [Input('set-binning-full-button', 'n_clicks_timestamp'),
     Input('zoom-point-button', 'n_clicks_timestamp'),
     Input('load-state-button', 'n_clicks_timestamp'),
     #Input('fit-ref-point-button', 'n_clicks_timestamp'),
     #Input('fit-start-value-button', 'n_clicks_timestamp'),
     #Input('fit-end-value-button', 'n_clicks_timestamp'),
     Input('fit-button', 'n_clicks_timestamp'),
     Input('fit-confirm-button', 'n_clicks_timestamp'),
     Input('fit-clear-button', 'n_clicks_timestamp'),
     Input('set-binning-zoom-button', 'n_clicks_timestamp'),
     Input('delete-button', 'n_clicks_timestamp'),
     Input('refresh-graph-button', 'n_clicks_timestamp')
     ],
    [State('buttons-times', 'children')])
@timeit
def update_last_clicked_button(*args):
    buttonNames = ('set-binning-full-button', 'zoom-point-button',
                   'load-state-button', 
                   #'fit-ref-point-button',
                   #'fit-start-value-button',
                   #'fit-end-value-button',
                   'fit-button', 'fit-confirm-button',
                   'clear-fit-button', 'set-binning-zoom-button',
                   'delete-button', 'refresh-graph-button',
                   # 'reload-button', 'download-button'
                   )

    buttonsTimesDict = dict(zip(buttonNames, args))
    previousState = args[-1]

    if previousState is not None:
        previousState = json.loads(previousState)
        for k, v in buttonsTimesDict.items():
            buttonsTimesDict[k] = [v, previousState[k][0]]

    else:
        for k, v in buttonsTimesDict.items():
            buttonsTimesDict[k] = [v, 0]
    return json.dumps(buttonsTimesDict)


@app.callback(Output('all-data-mean-text', 'children'),
              [Input('data-modification-trigger', 'children')])
@timeit
def update_global_mean_text(_):
    global_mean = get_global_mean(df, sf_trigger)

    return 'Global mean value: {:.2f}'.format(global_mean)


@app.callback(Output('n-points-day', 'children'),
              [Input('data-modification-trigger', 'children'),
               Input('buttons-times', 'children')])
@timeit
def update_n_points_day_between(_, _2):
    global fit_start_value_x, fit_end_value_x

    if (fit_start_value_x is not None) and (fit_end_value_x is not None):
        if fit_start_value_x > fit_end_value_x:
            fit_start_value_x, fit_end_value_x = fit_end_value_x, fit_start_value_x

        dff = get_activ(sf_trigger)
        dff = dff[(dff.time >= fit_start_value_x) &
                  (dff.time <= fit_end_value_x)]
        if dff.time.__len__() > 0:
            days = dff.time.values[-1] - dff.time.values[0]
        else:
            days = 1

        return 'n: {:d};    n/day: {:.1f}'.format(dff.time.__len__(),
            dff.time.__len__() / days)
    return 'n: ---;    n/day: ---'

@app.callback(Output('zoom-point-x-text', 'children'),
              [Input('zoom-point-button', 'n_clicks_timestamp')],
              [State('all-data-graph', 'clickData')])
@timeit
def update_zoom_start_point_x(_, clickData):
    global zoomStartPoint
    if clickData is not None:
        zoomStartPoint = [clickData['points'][0]['x'],
                          clickData['points'][0]['y']]
        return 'x: {:.8f}'.format(zoomStartPoint[0])
    if len(df.time) > 0:
        return 'x: {:.8f}'.format(df.time.values[0])
    return 'x: ---'


@app.callback(Output('saved-states-list', 'options'),
              [Input('save-state-temp', 'children'),
               Input('upload-temp', 'children')])
@timeit
def update_saved_states_list(buttonsTimes, _):
    states = stateRecorder.entries
    statesDict = []
    for state in states:
        state_str = datetime.strptime(
            state[1], 'D%d%m%yT%H%M%S').isoformat()
        state_str = state_str.split('T')[1]
        statesDict.append(dict(label=state_str, value=state[1]))
    return statesDict

@app.callback(Output('fit-ref-point-x', 'value'),
              [Input('fit-ref-point-button', 'n_clicks_timestamp')],
              [State('all-data-graph', 'clickData')])
@timeit
def update_ref_point_x(_, clickData):
    global refPoint_x
    if clickData is not None:
        refPoint_x = clickData['points'][0]['x']
        return '{:.8f}'.format(refPoint_x)
    return None


@app.callback(Output('fit-ref-point-y', 'value'),
              [Input('fit-ref-point-button', 'n_clicks_timestamp')],
              [State('all-data-graph', 'clickData')])
@timeit
def update_ref_point_y(_, clickData):
    global refPoint_y
    if clickData is not None:
        new_refPoint_y = clickData['points'][0]['y']
        if new_refPoint_y == refPoint_y:
            refPoint_y = None
            return None
        else:
            refPoint_y = new_refPoint_y
        return '{:.8f}'.format(refPoint_y)
    return None


@app.callback(Output('fit-start-value-x', 'value'),
              [Input('fit-start-value-button', 'n_clicks_timestamp')],
              [State('all-data-graph', 'clickData')])
@timeit
def update_start_point_value_x(_, clickData):
    global fit_start_value_x
    if clickData is not None:
        fit_start_value_x = float(clickData['points'][0]['x'])
        return '{:.8f}'.format(fit_start_value_x)
    return None


@app.callback(Output('fit-end-value-x', 'value'),
              [Input('fit-end-value-button', 'n_clicks_timestamp')],
              [State('all-data-graph', 'clickData')])
@timeit
def update_end_point_value_x(_, clickData):
    global fit_end_value_x
    if clickData is not None:
        fit_end_value_x = float(clickData['points'][0]['x'])
        return '{:.8f}'.format(fit_end_value_x)
    return None

@app.callback(Output('fit-start-value-y', 'value'),
              [Input('fit-start-value-button', 'n_clicks_timestamp')],
              [State('all-data-graph', 'clickData')])
@timeit
def update_start_point_value_y(_, clickData):
    global fit_start_value_y
    if clickData is not None:
        fit_start_value_y = float(clickData['points'][0]['y'])
        return '{:.8f}'.format(fit_start_value_y)
    return None


@app.callback(Output('fit-end-value-y', 'value'),
              [Input('fit-end-value-button', 'n_clicks_timestamp')],
              [State('all-data-graph', 'clickData')])
@timeit
def update_end_point_value_y(_, clickData):
    global fit_end_value_y
    if clickData is not None:
        fit_end_value_y = float(clickData['points'][0]['y'])
        return '{:.8f}'.format(fit_end_value_y)
    return None


@app.callback(Output('refresh-graph-button-temp', 'value'),
              [Input('refresh-graph-button', 'n_clicks_timestamp')],
              [State('fit-ref-point-y', 'value'),
               State('fit-start-value-x', 'value'),
               State('fit-end-value-x', 'value')])
@timeit
def refresh_graph(_, _fit_ref_point_y, _fit_start_value_x, _fit_end_value_x):
    global fit_ref_point_y, fit_start_value_x, fit_end_value_x
    if _fit_ref_point_y is not None:
        fit_ref_point_y = float(_fit_ref_point_y)
    if fit_start_value_x is not None:
        fit_start_value_x = float(_fit_start_value_x)
    if fit_end_value_x is not None:
        fit_end_value_x = float(_fit_end_value_x)

    return time.time()


@app.callback(Output('input-function-parameter', 'disabled'),
              [Input('fit-function-type', 'value')])
@timeit
def lock_fit_function_parameter_value(fitFunction):
    if fitFunction in ['spline', 'movingaverage_p',
                       'movingaverage_t', 'shift']:
        return False
    return True


@app.callback(Output('input-function-parameter2', 'disabled'),
              [Input('fit-function-type', 'value')])
@timeit
def lock_fit_function_parameter_value2(fitFunction):
    if fitFunction in ['movingaverage_p']:
        return False
    return True


@app.callback(Output('fit-function-parameter-text', 'children'),
              [Input('fit-function-type', 'value')])
def update_parameter_text(fitFunction):
    if fitFunction in ['movingaverage_p', 'movingaverage_t']:
        return 'Window'
    elif fitFunction == 'spline':
        return 'Binning'
    elif fitFunction == 'shift':
        return 'Shift'
    else:
        return '---'


@app.callback(Output('fit-function-parameter-text2', 'children'),
              [Input('fit-function-type', 'value')])
def update_parameter_text2(fitFunction):
    if fitFunction in ['movingaverage_p']:
        return 'nSigma'
    else:
        return '---'


@app.callback(Output('fitted-function-temp', 'children'),
              [Input('fit-button', 'n_clicks_timestamp')],
              [State('fit-function-type', 'value'),
               State('input-function-parameter', 'value'),
               State('input-function-parameter2', 'value'),
               State('fit-ref-point-x', 'value'),
               State('fit-ref-point-y', 'value')])
@timeit
def update_fit_function(_, fitFunction,
                        parameterValue, parameterValue2,
                        refPointValueX, refPointValueY):

    logger.warning(fitFunction)
    
    parameterValue = float(parameterValue)
    global fit_func, fit_start_value_x, fit_end_value_x
    if fit_start_value_x is not None and fit_end_value_x is not None:
        if fit_start_value_x > fit_end_value_x:
            fit_start_value_x, fit_end_value_x = fit_end_value_x, fit_start_value_x
    else:
        fit_start_value_x = df.time.min()
        fit_end_value_x = df.time.max()
    dff = get_activ(sf_trigger)
    dff = dff[(dff.time >= fit_start_value_x) &
              (dff.time <= fit_end_value_x)]

    if fitFunction == 'spline':
        edge_distance = int(parameterValue // 2)
        parameters = {
          'fit_start_value_x': fit_start_value_x,
          'fit_end_value_x': fit_end_value_x,
          'left_bin': dff.head(edge_distance).mean(),
          'right_bin': dff.tail(edge_distance).mean()
        }
        dff = get_binned_xy(dff, parameterValue, sf_trigger)

        fit_func = fit_function(dff, fitFunction, parameters)
        return []
    if (fitFunction == 'movingaverage_p' or fitFunction == 'movingaverage_t'):
        fit_func = fit_function(dff, fitFunction,
            [parameterValue, parameterValue2])
        return []
    if fitFunction == 'shift':
        fit_func = fit_function(dff, fitFunction,
            [parameterValue, None,
            refPointValueX, refPointValueY])
        return []

    fit_func = fit_function(dff, fitFunction)
    return []


@app.callback(Output('fit-confirmed', 'children'),
              [Input('fit-confirm-button', 'n_clicks_timestamp')],
              [State('fit-ref-point-y', 'value'),
               State('all-data-mean-text', 'children')])
@timeit
def confirm_fit_function(_, refPointValue, all_data_mean):
    global confirmed_fit_func, fit_func, sf_trigger, df
    global fit_start_value_x, fit_end_value_x

    if fit_func is not None:
        z, xnew, ynew, _, func_name, *_ = fit_func

        if func_name == 'movingaverage_p':
            deactivate_points([z[1].time])
        elif func_name == 'movingaverage_t':
            df.loc[
                list(z[0].time.mean().index), 'counts'] -= z[0].counts.mean()
            if refPointValue is not None:
                df.loc[
                    list(z[0].time.mean().index), 'counts'] += float(
                        refPointValue)
            else:
                df.loc[
                    list(z[0].time.mean().index), 'counts'] += get_global_mean(
                        df, sf_trigger)
        elif func_name == 'shift':
            df.counts[(df.time >= float(xnew[0])) & (
                df.time <= float(xnew[-1])) & df.activ == 1] = ynew
        else:
            x = df.time[
                (df.time >= float(xnew[0])) & (df.time <= float(xnew[-1]))].values
            y = df.counts[
                (df.time >= float(xnew[0])) & (df.time <= float(xnew[-1]))].values

            if refPointValue is not None:
                ynew = y - z(x) + float(refPointValue)
            else:
                ynew = y - z(x) + get_global_mean(df, sf_trigger)

            df.counts[(df.time >= float(xnew[0])) & (
                df.time <= float(xnew[-1]))] = ynew
    sf_trigger += 1

    confirmed_fit_func = fit_func
    fit_func = None

    return time.time()


@app.callback(Output('fitted-function-temp4', 'children'),
              [Input('fit-save-button', 'n_clicks_timestamp')])
@timeit
def save_fit_function(_):
    global confirmed_fit_func, file_name
    if confirmed_fit_func is not None:
        FitRecorder.save_output(confirmed_fit_func, 0, 1, 5, fileName)
    confirmed_fit_func = None
    return time.time()


@app.callback(Output('fitted-function-temp3', 'children'),
              [Input('fit-clear-button', 'n_clicks_timestamp')])
@timeit
def clear_fit_function(_):
    global fit_func, confirmed_fit_func
    fit_func = None
    confirmed_fit_func = None

    return time.time()


@app.callback(Output('all-data-graph', 'figure'),
              [Input('data-modification-trigger', 'children'),
               Input('buttons-times', 'children'),
               Input('all-data-graph', 'relayoutData')],
              [State('input-binning-full', 'value'),
               State('all-data-graph', 'clickData'),
               State('fit-ref-point-y', 'value'),
               State('fit-start-value-x', 'value'),
               State('fit-end-value-x', 'value')])
@timeit
def update_all_data_graph(_, buttonsTimes, relayoutData,
                          binningValue, clickData, refPointValue,
                          startFitValue, endFitValue):
    global shadow_shape
    dff = get_activ(sf_trigger)
    layout = dict(shapes=[])

    buttonsTimes, lastClickedButton = load_button_times(buttonsTimes)

    zoomPointButtonTimes = buttonsTimes['zoom-point-button']
    if zoomStartPoint is not None:
        if zoomPointButtonTimes[0] > zoomPointButtonTimes[1]:
            endPoint = get_from_clicked(
                dff, buttonsTimes[
                    lastClickedButton][0], sf_trigger).time.values[
                :scattergl_limit][-1]
            startPoint = zoomStartPoint[0]

            shadow_shape = create_shadow_shape(startPoint, endPoint)
            layout['shapes'].append(shadow_shape)

        elif zoomPointButtonTimes[0] == zoomPointButtonTimes[1]:
            layout['shapes'].append(shadow_shape)

    if startFitValue is not None:
        layout['shapes'].append(create_line(startFitValue, 'vertical'))
    if endFitValue is not None:
        layout['shapes'].append(create_line(endFitValue, 'vertical'))
    if (refPointValue is not None) and (refPointValue != ''):
        try:
            set_point = float(refPointValue)
        except ValueError:
            logger.warning('Wrong ref point format')
        layout['shapes'].append(create_line(set_point, 'horizontal'))

    dff = get_binned_xy(dff, binningValue, sf_trigger)

    relayout_xrange = []
    relayout_yrange = []
    if relayoutData:
        if 'xaxis' in relayoutData:
            relayout_xrange = relayoutData['xaxis']

        if 'yaxis' in relayoutData:
            relayout_yrange = relayoutData['yaxis']

    layout['xaxis'] = dict(range=relayout_xrange,
                           title='Time [JD - {}]'.format(start_date_int))
    layout['yaxis'] = dict(range=relayout_yrange, title='Counts')

    layout['showlegend'] = False

    fig = get_full_graph(dff.time, dff.counts, sf_trigger)
    fig['layout'] = layout

    if fit_func is not None:
        functionPlot = create_function_plot(df, fit_func)
        for plot in functionPlot:
            fig['data'].append(plot)

    return fig


@app.callback(Output('zoom-data-graph', 'figure'),
              [Input('data-modification-trigger', 'children'),
               Input('buttons-times', 'children')],
              [State('zoom-graph-relayout-temp', 'children'),
               # State('zoom-data-graph', 'relayoutData'),
               State('input-binning-zoom', 'value'),
               State('all-data-graph', 'clickData')])
@timeit
def update_zoom_data_graph(_, buttonsTimes,
                           relayoutData, binningValue, clickData):

    dff = get_activ(sf_trigger)
    buttonsTimes, lastClickedButton = load_button_times(buttonsTimes)

    relayout_xrange = []
    relayout_yrange = []
    layout = {}
    if relayoutData:
        relayoutData = json.loads(relayoutData)
        if lastClickedButton == 'zoom-point-button':
            pass
        elif ('xaxis.range[0]' in relayoutData or
              'yaxis.range[0]' in relayoutData):
            relayout_xrange = [relayoutData['xaxis.range[0]'],
                               relayoutData['xaxis.range[1]']]

            relayout_yrange = [relayoutData['yaxis.range[0]'],
                               relayoutData['yaxis.range[1]']]

    layout['xaxis'] = dict(range=relayout_xrange,
                           title='Time [JD - {}]'.format(start_date_int))
    layout['yaxis'] = dict(range=relayout_yrange, title='Counts')

    layout['dragmode'] = config.get('ZOOM_GRAPH', 'DEFAULT_TOOL')
    layout['showlegend'] = False

    if zoomStartPoint is not None:
        dff = get_from_clicked(
            dff, buttonsTimes[
                lastClickedButton][0], sf_trigger).head(scattergl_limit)
    else:
        dff = dff.head(scattergl_limit)

    dff = get_binned_xy(dff, binningValue, sf_trigger)

    if dff.columns.__len__() > 0:
        fig = get_zoom_graph(dff.time, dff.counts, sf_trigger)
    else:
        fig = get_zoom_graph([], [], sf_trigger)

    fig['layout'] = layout

    return fig



@app.callback(Output('zoom-graph-relayout-temp', 'children'),
              [Input('zoom-data-graph', 'relayoutData'),
               Input('buttons-times', 'children')],
              [State('zoom-graph-relayout-temp', 'children')])
@timeit
def zoom_graph_relayout_data_update(relayoutData, buttonsTimes, prevData):

    if (relayoutData is not None) and ('xaxis.range[0]' in relayoutData or
                                       'yaxis.range[0]' in relayoutData):
        return json.dumps(relayoutData)

    elif (relayoutData is not None) and ('xaxis.autorange' in relayoutData or
                                         'yaxis.autorange' in relayoutData):
        return []
    return prevData


@app.callback(Output('delete-data', 'children'),
              [Input('delete-button', 'n_clicks_timestamp')],
              [State('zoom-data-graph', 'selectedData'),
               State('zoom-data-graph', 'clickData'),
               State('data-modification-trigger', 'children')])
@timeit
def delete_data(deleteButton, selectedData,
                clickData, _):

    global sf_trigger

    sf = []
    selectedDataTable = []
    if selectedData is not None and 'points' in selectedData:
        for point in selectedData['points']:
            selectedDataTable.append(point['x'])
        sf.append(selectedDataTable)

    if clickData is not None and 'points' in clickData:
        for point in clickData['points']:
            selectedDataTable.append(point['x'])
        sf.append(selectedDataTable)

    deactivate_points(sf)
    sf_trigger += 1

    return time.time()


@app.callback(Output('data-modification-trigger', 'children'),
              [Input('delete-data', 'children'),
               Input('fit-confirmed', 'children'),
               Input('upload-temp', 'children')])
@timeit
def data_modification_trigger(*args):
    return time.time()


@app.callback(Output('upload-temp', 'children'),
              [Input('upload-file', 'contents'),
               Input('upload-file', 'filename')])
@timeit
def upload_file(contents, file_name):
    global df, start_date_int, sf_trigger, fileName, stateRecorder
    if reload_all():
        if contents is not None:
            content_type, content_string = contents.split(',')
            df, start_date_int = open_upload_file(content_string)
            stateRecorder = StateRecorder()
            sf_trigger += 1
            fileName = file_name

    return time.time()


@app.callback(Output('upload-file-name', 'children'),
              [Input('upload-temp', 'children')])
@timeit
def show_upload_file(_):
    if fileName is not None:
        return fileName

    return '---'

@app.callback(Output('save-state-temp', 'children'),
              [Input('buttons-times', 'children')])
@timeit
def save_state(buttonsTimes):
    buttonsTimes, lastClickedButton = load_button_times(buttonsTimes)
    if lastClickedButton in ['delete-button', 'fit-confirm-button']:
        stateRecorder.save_state(df, lastClickedButton)
    return time.time()


@app.callback(Output('load-state-temp', 'children'),
              [Input('load-state-button', 'n_clicks_timestamp')],
              [State('saved-states-list', 'value')])
@timeit
def load_state(_, state):
    global df, sf, fit_func, shadow_shape
    global shadow_shape, zoomStartPoint, sf_trigger

    if state is not None:
        if reload_all():
            df = stateRecorder.load_state(state)

    return time.time()


@app.callback(Output('save-output-temp', 'children'),
              [Input('save-output-button', 'n_clicks_timestamp')],
              [State('save-format', 'value'),
               State('ppt', 'values')])
@timeit
def save_output(_, save_format, ppt):
    logger.info('ppt calculate - {}'.format(ppt))
    if len(ppt) > 0:
      ppt = True
    else:
      ppt = False
    if fileName is not None:
        stateRecorder.save_output(df, fileName, save_format, ppt)


@timeit
def deactivate_points(sf):
    points_for_dactiv = list(chain(*sf))
    try:
        df.loc[points_for_dactiv, 'activ'] = 0
    except ValueError:
        pass


@timeit
def reload_all():
    global df, sf, fit_func, shadow_shape, stateRecorder
    global shadow_shape, zoomStartPoint, sf_trigger
    global refPoint_x, refPoint_y, fit_start_value_x, fit_start_value_y
    global fit_end_value_x, fit_end_value_y

    df = pd.DataFrame(
        data={'time': [], 'counts': [], 'err': [],
              'flag': [], 'activ': []})
    sf = []
    start_date_int = None
    fit_func = None
    shadow_shape = None
    zoomStartPoint = None
    refPoint_x = None
    refPoint_y = None
    fit_start_value_x = None
    fit_start_value_y = None
    fit_end_value_x = None
    fit_end_value_y = None
    fileName = None
    sf_trigger += 1

    return True

########################################################
# CACHE UTILS FUNCTIONS
########################################################


@timeit
@cache.memoize()
def get_binned_xy(dff, binningValue, sf_trigger):
    if binningValue != 0:
        groups = dff.groupby(
            pd.cut(
                dff.index, dff.index.__len__() // binningValue)
        ).mean().dropna()
        # groups[.time'] = [i.left for i in groups.index.values]

        return groups
    return dff


@timeit
@cache.memoize()
def get_activ(sf_trigger):
    return df[df['activ'] == 1]


@timeit
@cache.memoize()
def get_from_clicked(dff, timeStamp, sf_trigger):
    dff = dff[dff.time >= zoomStartPoint[0]]
    return dff


@timeit
@cache.memoize()
def get_full_graph(x, y, sf_trigger):
    fig = {
        'data': [go.Pointcloud(
            x=x,
            y=y,
            marker=dict(color=config.get('MAIN_GRAPH',
                                         'POINTS_COLOR'),
                        sizemin=config.getfloat('MAIN_GRAPH',
                                                'POINTS_SIZEMIN'),
                        sizemax=config.getfloat('MAIN_GRAPH',
                                                'POINTS_SIZEMAX')))],
        'layout': []}

    return fig


@timeit
@cache.memoize()
def get_zoom_graph(x, y, sf_trigger):
    fig = {
        'data': [go.Scattergl(
            x=x,
            y=y,
            mode='markers',
            marker=dict(color=config.get('ZOOM_GRAPH',
                                         'POINTS_COLOR'),
                        size=config.getint('ZOOM_GRAPH',
                                           'POINTS_SIZE')))],
        'layout': []}

    return fig


@timeit
@cache.memoize()
def get_global_mean(dff, sf_trigger):
    dff = get_activ(sf_trigger)
    global_mean = dff.counts.mean()

    return global_mean


########################################################
# CSS
########################################################

app.css.append_css({
    "external_url": "/static/main.css"})
app.css.append_css({
    "external_url": "/static/loading.css"})

if __name__ == '__main__':

    app.run_server(host="0.0.0.0", port=8050, debug=True, threaded=False)
