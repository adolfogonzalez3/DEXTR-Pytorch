import os.path
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from evaluation.eval import eval_one_result
import dataloaders.pascal as pascal
from eval_model import eval_model

from nets import create_densenet, create_squeezenet, create_shufflenet

def compile_evals(eval_dir):
    rows = []
    for eval_file in eval_dir.iterdir():
        _, method, learning_rate = eval_file.stem.split('-')
        learning_rate = 1e-7 if learning_rate == 0 else learning_rate
        with eval_file.open() as csv:
            iou = float(csv.read())
        rows.append({'method': method, 'learning_rate': learning_rate,
                     'IOU': iou})
    return pd.DataFrame(rows).sort_values(['method', 'learning_rate'])


def compile_results(exp_root_dir):
    # Iterate through all the different methods
    #for method in method_names:
    
    data_frames = []
    for exp_dir in exp_root_dir.iterdir():
        if exp_dir.name.startswith('run'):
            _, method, learning_rate = exp_dir.name.split('-')
            learning_rate = float(learning_rate)
            learning_rate = 1e-7 if learning_rate == 0 else learning_rate
            data_frame = pd.read_csv(exp_dir / 'log.csv')
            data_frame['method'] = method
            data_frame['learning_rate'] = learning_rate
            data_frames.append(data_frame)
    result_frame = pd.concat(data_frames).sort_values('method')
    result_frame = result_frame.reset_index(drop=True)
    #print(compile_results(exp_folders, "file_name"))
    return result_frame

def apply_default(axis, title, ylabel):
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.set_xlabel('Epochs')
    chartBox = axis.get_position()
    axis.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.8, chartBox.height])
    axis.legend(loc='upper center', bbox_to_anchor=(1.2, 0.8), shadow=True, ncol=1)

def graph_results(data_frame):
    epochs = [10*i for i in range(10)]
    fig_best = plt.figure()
    best_ax = fig_best.add_subplot(111)
    best_ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    for method in ['densenet', 'shufflenet', 'squeezenet']:
        fig_train = plt.figure()
        fig_test = plt.figure()
        train_ax = fig_train.add_subplot(111)
        test_ax = fig_test.add_subplot(111)
        train_ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        test_ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        df_method = data_frame.query('method==@method')
        best = None
        for lr in [10**-i for i in range(2, 8)]:
            df = df_method.query('learning_rate == @lr')
            if best is None or best_val > df['test_loss'].min():
                best = lr
                best_val = df['test_loss'].min()
            train_ax.plot(epochs, df['train_loss'], label='{:.0e}'.format(lr))
            test_ax.plot(epochs, df['test_loss'], label='{:.0e}'.format(lr))
            test_ax.set_ylim(0, df['test_loss'].median()*2)
        df = df_method.query('learning_rate == @best')
        label = '{}-{:.0e}'.format(method, best)
        best_ax.plot(epochs, df['test_loss'], label=label)
        title = '{} Learning Rate Search'.format(method.capitalize())
        apply_default(train_ax, title, 'Train Loss')
        apply_default(test_ax, title, 'Test Loss')
        fig_train.savefig('{}-trainingloss'.format(title))
        fig_test.savefig('{}-testingloss'.format(title))
    title = 'Comparison of Best Performing'
    apply_default(best_ax, title, 'Test Loss')
    fig_best.savefig('{}-bestlr'.format(title))
    #plt.show()

if __name__ == '__main__':
    exp_root_dir = Path('RUNS', 'PARAMSEARCH')
    data_frame = compile_results(exp_root_dir)
    graph_results(data_frame)
    data_frame = compile_evals(exp_root_dir / 'eval_results')
    print(data_frame)
    data_frame.to_csv('IOU.csv', index=False)
