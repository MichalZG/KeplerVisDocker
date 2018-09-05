from scipy.interpolate import CubicSpline

import numpy as np
import time
import pandas as pd
import plotly.graph_objs as go
from collections import deque
import os
import json
import configparser
import logging
import io
import base64


config = configparser.ConfigParser()
config.read('./config/config.cfg')


def create_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    log_file_handler = logging.FileHandler(
        config.get('LOG', 'LOG_FILE_PATH'))
    fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    formatter = logging.Formatter(fmt)
    log_file_handler.setFormatter(formatter)

    logger.addHandler(log_file_handler)

    return logger


logger = create_logger()


def timeit(method):
    def timed(*args, **kw):

        result = None
        try:
            ts = time.time()
            result = method(*args, **kw)
            te = time.time()
        except Exception:
            logger.exception('')
        if result is not None:
            logger.info('{}: {:.2f} s'.format(method.__name__, te - ts))
            return result
    return timed


@timeit
def open_upload_file(content_string):
    df = pd.read_fwf(io.StringIO(
        base64.b64decode(content_string).decode('utf-8')),
        usecols=[int(x) - 1 for x in config.get(
            'FILES', 'USE_COLUMNS').split(',')],
        names=config.get('FILES', 'COLUMNS_NAMES').split(','))
    start_date_int = config.getfloat('FILES', 'START_JD')
    df.jd -= start_date_int
    df = df.set_index('jd')
    df = df.assign(jd=df.index)
    df = df.assign(activ=1).copy()

    return df, start_date_int


@timeit
def fit_function(dff, fitFuntion, parameters=[]):

    xnew = np.linspace(dff.jd.min(), dff.jd.max(),
                       num=dff.jd.__len__() * config.getint(
        'MAIN_GRAPH', 'FIT_DENSITY'),
        endpoint=True)

    if fitFuntion == 'line':
        z = np.poly1d(np.polyfit(dff.jd, dff.counts, 1))
        ynew = z(xnew)
        yrescale = 1
    elif fitFuntion == 'parabola':
        z = np.poly1d(np.polyfit(dff.jd, dff.counts, 2))
        ynew = z(xnew)
        yrescale = 1
    elif fitFuntion == 'spline':
        z = CubicSpline(dff.jd, dff.counts)
        ynew = z(xnew)
        yrescale = 1
    elif fitFuntion == 'movingaverage':
        roll_dff = dff.rolling(window=parameters[0], min_periods=1)
        roll_df_mean = roll_dff.mean()[roll_dff.mean().jd > 0]
        xnew = roll_df_mean.jd
        ynew = roll_df_mean.counts

        dff.loc[:, 'median'] = roll_dff.mean().counts
        dff.loc[:, 'std'] = roll_dff.std().counts

        dff_out = dff
        dff_out = dff[(
            dff.counts >= dff['median'] + parameters[1] * dff['std']) | (
            dff.counts <= dff['median'] - parameters[1] * dff['std'])]

        z = [roll_dff, dff_out]
        yrescale = 1

    return [z, xnew, ynew, yrescale, fitFuntion, parameters]


@timeit
def create_shadow_shape(startPoint, endPoint):

    shadowShape = dict(
        type='rect',
        xref='x',
        yref='paper',
        x0=startPoint,
        y0=0,
        x1=endPoint,
        y1=1,
        fillcolor=config.get('MAIN_GRAPH', 'SHADOW_COLOR'),
        opacity=config.getfloat('MAIN_GRAPH', 'SHADOW_OPACITY'),
        line=dict(width=0))

    return shadowShape


@timeit
def create_line(point):

    line = dict(
        type='line',
        xref='x',
        yref='paper',
        x0=point,
        y0=0,
        x1=point,
        y1=1,
        opacity=config.getfloat('MAIN_GRAPH', 'LINE_OPACITY'),
        line=dict(color=config.get('MAIN_GRAPH', 'LINE_COLOR'),
                  width=config.getfloat('MAIN_GRAPH', 'LINE_WIDTH')))

    return line


@timeit
def create_function_plot(dff, fit_func):
    func, xnew, ynew, yrescale, func_name, parameters = fit_func
    functionPlot = [go.Pointcloud(
        x=xnew,
        y=ynew * yrescale,
        marker=dict(color=config.get('MAIN_GRAPH', 'FIT_COLOR'),
                    sizemin=config.getfloat(
            'MAIN_GRAPH', 'FIT_POINTS_SIZEMIN'),
            sizemax=config.getfloat('MAIN_GRAPH',
                                    'FIT_POINTS_SIZEMAX')))]
    if func_name == 'movingaverage':
        dff_out = func[1]

        functionPlot.append(go.Pointcloud(
            x=dff_out.jd,
            y=dff_out.counts,
            marker=dict(color=config.get('MAIN_GRAPH', 'FIT_COLOR'),
                        sizemin=config.getfloat('MAIN_GRAPH',
                                                'FIT_POINTS_SIZEMIN'),
                        sizemax=config.getfloat('MAIN_GRAPH',
                                                'FIT_POINTS_SIZEMAX'))))
    return functionPlot


@timeit
def load_button_times(buttonsTimes):
    buttonsTimes = json.loads(buttonsTimes)
    lastClickedButton = max(buttonsTimes,
                            key=buttonsTimes.get)

    return buttonsTimes, lastClickedButton


class StateRecorder:
    def __init__(self):
        self.limit = config.getint('STATE', 'MAX_STATES')
        self.entries = deque(maxlen=self.limit)
        self.states_save_path = self.path_create(config.get('STATE',
                                                            'STATE_FILE_PATH'))
        self.output_save_path = self.path_create(config.get('STATE',
                                                            'OUTPUT_PATH'))
        self.clear_old()

    @timeit
    def save_state(self, dff):
        time_now = time.strftime("D%d%m%yT%H%M%S", time.gmtime())
        dff.to_hdf(self.states_save_path, key=time_now)
        self.entries.append(time_now)

    @timeit
    def load_state(self, key):
        dff = pd.read_hdf(self.states_save_path, key=key)
        return dff

    def clear_old(self):
        try:
            os.remove(self.states_save_path)
        except FileNotFoundError:
            pass

    def path_create(self, save_path):
        dirname = os.path.dirname(save_path)
        os.makedirs(dirname, exist_ok=True)
        return save_path

    def save_output(self, dff, file_name, save_format):
        time_now = time.strftime("D%d%m%yT%H%M%S", time.gmtime())
        file_name = file_name.replace('.', '_{}.'.format(time_now))
        if save_format == 'csv':
            dff.to_csv(os.path.join(
                config.get('STATE', 'OUTPUT_PATH'), file_name))
        elif save_format == 'txt':
            dff.to_csv(os.path.join(
                config.get('STATE', 'OUTPUT_PATH'), file_name),
                sep=' ')

        return True
