import numpy as np
import csv, os, sys
import pandas as pd
np.set_printoptions(suppress=True, precision=3)
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends import backend_pdf  # saves pdfs
import pickle
from .prep_data import *
# import np_bnn as bn
from . import np_bnn as bn

def plot_nn(res,
            clim_variable=None,
            reference_mat = None, #'true_labels',
            show_rmse=False,
            t_min = None,
            t_max = None,
            filename=None):
    fig = plt.figure(figsize=(5, 5))
    y_test = res['predicted_labels']
    if reference_mat is None:
        x_test = np.arange(len(y_test))
        x_ax_name = 'Sample'
    else:
        x_test = res[reference_mat]
        if reference_mat == 'true_labels':
            x_ax_name = 'True %s' % clim_variable
        else:
            x_ax_name = reference_mat
    try:
        model_name = res['model_name']
    except(KeyError):
        model_name = "NN"

    # NN res
    if reference_mat is not None:
        plt.axline((0, 0), (1, 1), linewidth=2, linestyle='dashed', alpha=0.5, color="k")
        if show_rmse:
            rmse = np.round(np.sqrt(np.mean((y_test - x_test)**2)),2)
            lab="%s (RMSE: %s)" % (model_name, rmse)
        else:
            lab=""
        g = sns.regplot(x=x_test, y=y_test, label=lab, fit_reg=False)
        if lab != "":
            g.legend(loc=2)
        if t_min:
            g.set(ylim=(t_min, t_max), xlim=(t_min, t_max))
    else:
        g = sns.regplot(x=x_test, y=y_test, fit_reg=False)
        if t_min:
            g.set(ylim=(t_min, t_max))
    plt.xlabel(x_ax_name)
    plt.ylabel('Predicted %s' % clim_variable)
    title = "NN regression (N = %s)" % len(y_test)
    plt.title(title)
    if filename is None:
        fig.show()
    else:
        plot_div = matplotlib.backends.backend_pdf.PdfPages(filename)
        fig.tight_layout()
        fig.subplots_adjust(top=0.92)
        plot_div.savefig(fig)
        plot_div.close()
        print("Plot saved as:", filename, "\n")



def plot_bnn_training(bnn_res,
             show=True,
             filename=None,
             rescale_labels=False,
             title=None):
    x_test = bnn_res['true_labels']
    z_test = rescale_MAT_data(bnn_res['sampled_labels'], reverse=True, rescale=rescale_labels)
    mu = rescale_MAT_data(bnn_res['predicted_labels'], reverse=True, rescale=rescale_labels)

    # calc coverage
    cov = 0
    for i in range(z_test.shape[0]):
        # based on posterior samples
        m, M = bn.calcHPD(z_test[i,:], 0.95)
        # based on point estimates of mu and sigma
        # m,M = mu_est[i] - 1.96*sd_est[i], mu_est[i] + 1.96*sd_est[i]
        if x_test[i] > m and x_test[i] < M:
            cov += 1

    coverage = cov/z_test.shape[0]

    # plot w HPD
    t_min, t_max = -5, 32
    rmse = np.round(np.sqrt(np.mean((np.mean(mu, 1) - x_test.flatten()) ** 2)), 2)
    fig = plt.figure(figsize=(5, 5))
    plt.axline((0, 0), (1, 1), linewidth=2, linestyle='dashed', alpha=0.5, color="k")
    pcx = plt.violinplot(positions=x_test.flatten(), dataset=z_test.T,showmedians=False, showextrema=False)
    for pc in pcx['bodies']:
        pc.set_facecolor('#3690c0')
        pc.set_edgecolor('#3690c0')
        pc.set_alpha(0.2)
    g = sns.scatterplot(x=x_test.flatten(),y=np.mean(mu,1),
        label= "BNN-reg (RMSE: %s, Coverage: %s)" % (rmse, 100*np.round(coverage, 4)),
         color='#034e7b', s=15)
    g.set(ylim=(t_min, t_max), xlim=(t_min, t_max))
    plt.xlim(t_min, t_max)
    plt.ylim(t_min, t_max)
    plt.xlabel('True temperature')
    plt.ylabel('Predicted temperature (95% CI)')
    g.legend(loc=2)
    # plt.legend([pcx['bodies'][0]], ['flat'], loc=2)
    if title is None:
        title = "BNN model (N = %s)" % len(x_test)
    else:
        title = title + "(N = %s)" % len(x_test)
    plt.title(title)
    if show or filename is None:
        fig.show()
    else:
        plot_div = matplotlib.backends.backend_pdf.PdfPages(filename)
        fig.tight_layout()
        fig.subplots_adjust(top=0.92)
        plot_div.savefig(fig)
        plot_div.close()
        print("Plot saved as:", filename, "\n")


def plot_bnn(bnn_res,
             filename=None,
             reference_label=None,
             show_rmse=True,
             t_min=-5,
             t_max=32,
             drop_tails=False,
             use_estimated_error=True,
             variable_name='variable'
             ):
    if use_estimated_error:
          z_test = bnn_res['post_samples']
    else:
        z_test = bnn_res['post_est'].T[0]
    mu = bnn_res['point_estimates']

    if reference_label is None:
        x_test = np.arange(len(mu))
        x_ax_name = 'Sample'
    else:
        x_test = bnn_res[reference_label] + 0
        x_test = np.array(x_test).reshape(mu.shape)
        if reference_label == 'true_labels':
            x_ax_name = 'True %s' % variable_name
        else:
            x_ax_name = reference_label

    # calc coverage
    if show_rmse and reference_label:
        cov = 0
        for i in range(z_test.shape[0]):
            # based on posterior samples
            m,M = bn.calcHPD(z_test[i,:], 0.95)
            if x_test[i] > m and x_test[i] < M:
                cov += 1
            if drop_tails:
                tmp = z_test[i,:]
                ind1 = np.where(tmp < m)[0]
                ind2 = np.where(tmp > M)[0]
                ind = np.intersect1d(ind1, ind2)
                z_test[i,ind] = np.nan


        coverage = np.round(100*(cov/z_test.shape[0]), 2)
        rmse = np.round(np.sqrt(np.mean((mu - x_test) ** 2)), 2)
        lab = "RMSE: %s, Coverage: %s/100" % (rmse, coverage)
    else:
        lab=""

    mask = ~np.isnan(z_test)
    filtered_data = [d[m].T for d, m in zip(z_test, mask)]

    # plot w HPD
    fig = plt.figure(figsize=(5, 5))
    pcx = plt.violinplot(positions=x_test.flatten(), dataset=filtered_data, showmedians=False, showextrema=False)
    for pc in pcx['bodies']:
        pc.set_facecolor('#3690c0')
        pc.set_edgecolor('#3690c0')
        pc.set_alpha(0.2)

    plt.xlabel(x_ax_name)
    plt.ylabel('Predicted %s (0.95 CI)' % variable_name) 
    # plt.legend([pcx['bodies'][0]], ['flat'], loc=2)
    if 'plot_title' in bnn_res.keys():
        title = "%s (N = %s)" % (bnn_res['plot_title'], len(x_test))
        plt.title(title)

    g = sns.scatterplot(x=x_test.flatten(), y=mu,
                        label=lab,
                        color='#034e7b', s=15)

    if reference_label is not None:
        plt.axline((0, 0), (1, 1), linewidth=2, linestyle='dashed', alpha=0.5, color="k")
        if show_rmse:
            g.legend(loc=2)
        g.set(ylim=(t_min, t_max), xlim=(t_min, t_max))
        plt.xlim(t_min, t_max)
    else:
        plt.ylim(t_min, t_max)

    if filename is None:
        fig.show()
    else:
        plot_div = matplotlib.backends.backend_pdf.PdfPages(filename)
        fig.tight_layout()
        fig.subplots_adjust(top=0.92)
        plot_div.savefig(fig)
        plot_div.close()
        print("Plot saved as:", filename, "\n")




