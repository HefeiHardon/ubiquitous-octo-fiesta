# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
#绘制验证分数
def smooth_curve(points,factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous*factor+point*(1-factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

def draw_curve(averge_mae_history):
    smooth_mae_history = smooth_curve(averge_mae_history[10:])
    plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
    plt.xlabel('Epochs')
    plt.ylabel('Validation MAE')
    plt.show()
    return list(smooth_mae_history)

def plot_history(history):
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error')
    plt.plot(history.epoch,np.array(history.history['mae']),label='Train Loss')
    plt.plot(history.epoch,np.array(history.history['val_mae']),label='Val loss')
    plt.legend()
    plt.ylim([0,5])
