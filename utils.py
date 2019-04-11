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
    df = pd.read_csv(io.StringIO(
        base64.b64decode(content_string).decode('utf-8')),
        usecols=[int(x) - 1 for x in config.get(
            'FILES', 'USE_COLUMNS').split(',')],
        names=config.get('FILES', 'COLUMNS_NAMES').split(','),
        delim_whitespace=True, comment='#', skip_blank_lines=True,
        dtype=np.float64)
    start_date_int = config.getfloat('FILES', 'START_JD')
    if df.time[0] > 2450000.0:
        start_date_int += 2450000.0
    df.time -= start_date_int
    df = df.set_index('time')
    df = df.assign(time=df.index)
    df = df.assign(activ=1).copy()

    return df, start_date_int


@timeit
def fit_function(dff, fitFunction, parameters=[]):

    xnew = np.linspace(dff.time.min(), dff.time.max(),
                       num=dff.time.__len__() * config.getint(
        'MAIN_GRAPH', 'FIT_DENSITY'),
        endpoint=True)

    if fitFunction == 'line':
        z = np.poly1d(np.polyfit(dff.time, dff.counts, 1))
        ynew = z(xnew)
        yrescale = 1
    elif fitFunction == 'parabola':
        z = np.poly1d(np.polyfit(dff.time, dff.counts, 2))
        ynew = z(xnew)
        yrescale = 1
    elif fitFunction == 'spline':
        z = CubicSpline(dff.time, dff.counts)
        ynew = z(xnew)
        yrescale = 1
    elif (fitFunction == 'movingaverage_p' or
          fitFunction == 'movingaverage_t'):

        parameters[0] = int(parameters[0])
        roll_dff = dff.copy().rolling(window=parameters[0],
            min_periods=1, center=True) 
        roll_df_mean = roll_dff.mean()[roll_dff.mean().time > 0]
        xnew = roll_df_mean.index
        ynew = roll_df_mean.counts

        dff.loc[:, 'median'] = roll_dff.counts.mean()
        dff.loc[:, 'std'] = roll_dff.counts.std()

        dff_out = dff[(
            dff.counts >= dff['median'] + (parameters[1] * dff['std'])) | (
            dff.counts <= dff['median'] - (parameters[1] * dff['std']))]

        z = [roll_dff, dff_out]
        yrescale = 1
    elif fitFunction == 'shift':
        z = None
        yrescale = 1
        xnew = dff.time.values
        ynew = dff.counts 

        if parameters[2] is None:
            parameters[2] = xnew[0]
        else:
            parameters[2] = float(parameters[2])
            
        if parameters[3] is None:
            parameters[3] = ynew.values[0]
        else:
            parameters[3] = float(parameters[3])

        if abs(xnew[0] - parameters[2]) < abs(xnew[-1] - parameters[2]):
            y0_point = ynew.values[0]
        else:
            y0_point = ynew.values[-1]

        if parameters[0] != 0:
            ynew = (
                dff.counts / dff.counts.mean()) * (
                dff.counts.mean() + parameters[0])
        elif parameters[3] is not None:
            ynew = (dff.counts / y0_point) * parameters[3]

        ynew = ynew.values

    return [z, xnew, ynew, yrescale, fitFunction, parameters]


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
def create_line(point, line_type):

    if line_type == 'vertical':
        line = dict(
            type='line',
            xref='x',
            yref='paper',
            x0=point,
            y0=0,
            x1=point,
            y1=1,
            opacity=config.getfloat(
                'MAIN_GRAPH', 'LINE_VERTICAL_OPACITY'),
            line=dict(
                color=config.get(
                    'MAIN_GRAPH', 'LINE_VERTICAL_COLOR'),
                width=config.getfloat(
                    'MAIN_GRAPH', 'LINE_VERTICAL_WIDTH'),
                dash=config.get(
                    'MAIN_GRAPH', 'LINE_VERTICAL_DASH')))

    elif line_type == 'horizontal':
        line = dict(
            type='line',
            xref='paper',
            yref='y',
            x0=0,
            y0=point,
            x1=1,
            y1=point,
            opacity=config.getfloat(
                'MAIN_GRAPH', 'LINE_HORIZONTAL_OPACITY'),
            line=dict(
                color=config.get(
                    'MAIN_GRAPH', 'LINE_HORIZONTAL_COLOR'),
                width=config.getfloat(
                    'MAIN_GRAPH', 'LINE_HORIZONTAL_WIDTH'),
                dash=config.get(
                    'MAIN_GRAPH', 'LINE_HORIZONTAL_DASH')))
    else:
        line = dict()

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
    if (func_name == 'movingaverage_p'):
        dff_out = func[1]

        functionPlot.append(go.Pointcloud(
            x=dff_out.time,
            y=dff_out.counts,
            marker=dict(color=config.get(
                'MAIN_GRAPH', 'FIT_MOVING_AVERAGE_POINTS_COLOR'),
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
    def save_state(self, dff, lastClickedButton):
        time_now = time.strftime("D%d%m%yT%H%M%S", time.gmtime())
        dff.to_hdf(self.states_save_path, key=time_now)
        self.entries.append([lastClickedButton, time_now])

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

    def save_output(self, dff, file_name, save_format, ppt):
        time_now = time.strftime("D%d%m%yT%H%M%S", time.gmtime())
        # file_name = file_name.replace('.', '_{}.'.format(time_now))
        file_name = '_'.join([file_name, time_now]) + '.' + save_format
        dff = dff[dff['activ'] == 1]

        columns_to_save = config.get('FILES', 'COLUMNS_NAMES').split(',')
        columns_format = config.get(
                    'FILES', 'OUTPUT_COLUMNS_FORMAT').replace(',', ' ')

        logger.info('Save to {}'.format(save_format))
        logger.info('PPT calulate - {}'.format(str(ppt)))
        if ppt is True:
            dff = self.calculate_ppt(dff)
            columns_to_save[columns_to_save.index('counts')] = 'ppt'
            # columns_to_save.append('ppt')
            columns_to_save.remove('errors')
            # columns_format += ' %.8f'
            columns_format = ' '.join(columns_format.split(' ')[1:])
        #TODO zapis ppt zamiast flux!
        if save_format == 'csv':
            dff.to_csv(os.path.join(
                config.get('STATE', 'OUTPUT_PATH'), file_name), index=False,
                columns=columns_to_save,
                # float_format=columns_format
                )

        elif save_format == 'txt':
            np.savetxt(os.path.join(
                config.get('STATE', 'OUTPUT_PATH'), file_name),
                # np.c_[dff.time, dff.counts, dff.errors, dff.flags],
                np.c_[tuple(dff[column] for column in columns_to_save)],
                fmt=columns_format,
                header=' '.join(columns_to_save))

        return True

    def calculate_ppt(self, dff):
        dff_counts_mean = dff['counts'].mean()
        print(dff_counts_mean)
        dff['ppt'] = ((dff['counts'] / dff_counts_mean) - 1) * 1000

        return dff


class FitRecorder:
    def __init__(self):
        self.fit_save_path = self.path_create(config.get('FIT',
                                                         'FIT_OUTPUT_PATH'))

    def path_create(self, save_path):
        dirname = os.path.dirname(save_path)
        os.makedirs(dirname, exist_ok=True)
        return save_path

    def save_output(self, fit_func, start, end, ref_point, file_name):
        file_name = '_'.join([file_name, 'fits.dat'])

        with open(os.path.join(self.fit_save_path, file_name), 'a') as f:
            f.write('{}\n'.format(fit_func[0]))
            f.write('{}; {}; {}\n'.format(start, end, ref_point))

        return True
