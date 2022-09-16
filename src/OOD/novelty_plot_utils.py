import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from matplotlib import colors as mcolors
import random
import pickle

from sklearn.metrics import fbeta_score

from novelty_dfm_CL.novelty_utils import confidence_scores

def standard_error(data):
    error = np.std(data, ddof=1) / np.sqrt(np.size(data))
    return error


class ThresholdingResults():
    def __init__(self, percentiles, num_tasks):

        self.percentiles = percentiles
        self.num_tasks = num_tasks
        num_steps = percentiles.shape[0]+1

        self.thresholds = np.zeros((num_steps-1, num_tasks))
        self.accuracy_novelty = np.zeros((num_steps-1, num_tasks))
        self.precision_novelty = np.zeros((num_steps-1, num_tasks))
        self.recall_novelty = np.zeros((num_steps-1, num_tasks))
        self.fbeta_novelty = None


        self.confidences_Novel={}
        self.confidences_Old = {}

        self.clip_err = 0.0000000001

    def log_results(self, per_task_results, f1_beta=1):

        for j in range(self.num_tasks):
            tsk = j+1

            if hasattr(per_task_results, 'scores'):
                print('Has attribute scores')
                self.scores = per_task_results.scores[tsk]
            else:
                print('only has attribute scores_dist')
                self.scores = per_task_results.scores_dist[tsk].min(0)
            self.gt_novelty = per_task_results.gt_novelty[tsk]
            self.inds_old = (self.gt_novelty==0)
            self.inds_new = (self.gt_novelty==1)
            self.conf,_ = confidence_scores(per_task_results.scores_dist[tsk], metric='variance')


            self.thresholds[...,j]=np.percentile(self.scores, self.percentiles)

            self.confidences_Novel[j]={}
            self.confidences_Old[j]={}

            for i, th in enumerate(self.thresholds[...,j]):
                (self.precision_novelty[i,j], self.recall_novelty[i,j], \
                    self.confidences_Novel[j][i], self.confidences_Old[j][i], self.accuracy_novelty[i,j]) = self.compute_stats(th)

            # print(self.confidences_Novel.keys())


        self.precision_novelty = np.clip(self.precision_novelty,self.clip_err, None)
        self.recall_novelty = np.clip(self.recall_novelty,self.clip_err, None)
        self.fbeta_novelty = (((1+f1_beta**2))*self.precision_novelty*self.recall_novelty)/((f1_beta**2)*self.precision_novelty+self.recall_novelty)


    def compute_stats(self, threshold):

        inds_above = self.scores>threshold # binary classification (above th is considered to be Novelty (Positive class))
        precision_novelty = self.gt_novelty[inds_above].sum()/self.gt_novelty[inds_above].shape[0]
        recall_novelty = self.gt_novelty[inds_above].sum()/self.gt_novelty.sum()


        accuracy_novelty = sum(self.inds_new == inds_above)/self.inds_new.shape[0]

        lbls_novel = np.logical_and(self.inds_new, inds_above)
        lbls_old = np.logical_and(self.inds_old, inds_above)

        confidences_Novel = self.conf[lbls_novel]
        confidences_Old = self.conf[lbls_old]

        return (precision_novelty, recall_novelty, confidences_Novel, confidences_Old, accuracy_novelty)



def multi_task_subplots(data_mult, savepath=None, title_plot='No Title', xlabels=None, y_ticks=np.arange(0.5,1.1,0.1), 
                        fig_size_=(10.5,10),  codes=None, colors_set=None, stepsize=1, task_delim=False, turnoffX=True):

    '''data_mult is 3D - num_curves,steps,num_tasks'''

    num_tasks = data_mult.shape[-1]

    while len(data_mult.shape)<3:
        data_mult = np.expand_dims(data_mult, axis=0)

    if colors_set is not None:
        colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
        color_list = [colors[p] for p in colors_set]
    else:
        number_of_colors = data_mult.shape[0]
        color_list = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                    for i in range(number_of_colors)]

    if codes is None:
        codes = [str(i) for i in range(data_mult.shape[0])]

    tasks_plot = np.arange(0,num_tasks,stepsize)
    f, axarr = plt.subplots(tasks_plot.shape[0], sharex=True,  figsize=fig_size_)
    f.subplots_adjust(hspace=0.4)

    # print('data_mult shape:', data_mult.shape)
    alpha=0.3


    
    for i,t in enumerate(tasks_plot):
        
            
        if hasattr(axarr, 'shape'):
            axplot = axarr[i]
        else:
            axplot = axarr

        if i==0:
            axplot.set_title(title_plot)
            
        for k in range(data_mult.shape[0]):
            axplot.plot(np.arange(data_mult.shape[1]), data_mult[k,:,t], linestyle='dashdot', color=color_list[k], linewidth=2.5, markersize=2, label='%s'% (codes[k]))
        
        if task_delim:
            for b in range(num_tasks):
                axplot.axvline(x=task_delim*(b+1), alpha=alpha)
            
        axplot.set_ylabel('Task %d'%((t+1)), rotation=90, fontsize=12, fontweight='semibold')

        # y_ticks = np.arange(0.5,1.1,0.1)
        # print([str(k) for k in y_ticks])
        axplot.set_yticks(y_ticks)
        axplot.set_ylim([y_ticks[0], y_ticks[-1]])
        axplot.set_yticklabels(['%.2f'%k for k in y_ticks], fontsize=10, fontweight='semibold')
        
        # axplot.tick_params(axis = 'both', which = 'major', labelsize = 7, width=1)

        for axis in ['bottom','left']:
            axplot.spines[axis].set_linewidth(1.0)

        
        # # Hide the right and top spines
        axplot.spines['right'].set_visible(False)
        axplot.spines['top'].set_visible(False)

        # turn off x axis for all but last task
        # if turnoffX:
        #     if i<num_tasks-1:
        #         axplot.spines['bottom'].set_visible(False)
        #         axplot.set_xticks([])
        #         axplot.tick_params(bottom="off")
        # else:
        assert xlabels is not None
        # print(['%.2f'%k for k in xlabels[...,i]])
        # print(data_mult[...,i])
        # print((np.arange(xlabels[...,i].shape[0])))
        axplot.set_xticks(np.arange(xlabels[...,i].shape[0]))
        axplot.set_xticklabels(['%.2f'%k for k in xlabels[...,i]], fontsize=10, fontweight='semibold')
        

        if i==len(tasks_plot)-1:
            axplot.set_xlabel('Percentile Thresholds')
            
    #     # # # Only show ticks on the left and bottom spines
    #     axplot.yaxis.set_ticks_position('left')

    # axplot.axis([0, np.arange(data_mult.shape[1]).shape[0], 0.0, 1.0])

    lgd = axplot.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='large')
    lgd.get_frame().set_linewidth(0.0)

    plt.tight_layout()

    plt.show()

    if savepath is not None:
        # save image
        plt.savefig(savepath)





def confidence_multi_task_subplots(data_mult, thresholds, savepath=None, title_plot='No Title', 
                                                                fig_size_=(10.5,10), alpha=0.9, codes=None, stepsize_task=1, colors_set=None):

    '''data_mult is a list of nested dictionaries - each element (dict) in the list is a different type of files (ex: TP and FP)
    first dim dict= num tasks 
    third dim dict = are different values obtained via thresholding 
        Will plot mean and spreads of confidence values for TP and FP (novel and old) given each threshold value, per task.
    '''

    num_codes = len(data_mult)
    num_tasks = len(data_mult[0])
    task_vals = list(data_mult[0].keys())

    
    if colors_set is not None:
        colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
        color_list = [colors[p] for p in colors_set]
    else:
        number_of_colors = num_codes
        color_list = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                    for i in range(number_of_colors)]

    if codes is None:
        codes = [str(i) for i in range(num_codes)]



    tasks_plot = np.arange(0,num_tasks,stepsize_task)
    f, axarr = plt.subplots(tasks_plot.shape[0], sharex=True, figsize=fig_size_)
    f.subplots_adjust(hspace=0.4)
    xarray = np.arange(len(thresholds))

    width = .15 # how many bars 

    # plt.suptitle(title_plot)

    for i,t in enumerate(task_vals):
        if hasattr(axarr, 'shape'):
            axplot = axarr[i]
        else:
            axplot = axarr

        if i==0:
            axplot.set_title(title_plot)

        
        th_mean = []
        th_e = []
        for j in range(num_codes):
            # compute means/errors over all values of threshold for the task at hand
            th_mean.append([])
            th_e.append([])

            # assume vals >0
            for k, th in enumerate(thresholds):
                if data_mult[j][t][k].shape[0]>0:
                    th_mean[j].append(np.mean(data_mult[j][i][k]))
                    th_e[j].append(standard_error(data_mult[j][i][k]))
                else:
                    th_mean[j].append(0)
                    th_e[j].append(0)

            # print(th_mean[j])
            # axplot.errorbar(xarray, th_mean[j], th_e[j], linestyle='None', marker='^', color=color_list[j], label='%s'% (codes[j]), alpha=0.6)
            position = xarray + (width*(1-num_codes)/2) + j*width
            axplot.bar(position, th_mean[j], width=width, yerr=th_e[j], color=color_list[j], align='center', alpha=alpha, ecolor='black', label='%s'% (codes[j]))



        th_mean = np.array(th_mean)
        th_e = np.array(th_e)

            
        axplot.set_ylabel('Task %d'%((t)), rotation=90, fontsize=12, fontweight='semibold')

        y_ticks = np.arange(th_mean.min(), th_mean.max()+2*th_e.max(), ((th_mean.max()+2*th_e.max())-th_mean.min())/10)

        print(t, 'y_ticks', y_ticks)


        axplot.set_ylim([y_ticks[0], y_ticks[-1]])
        axplot.set_yticklabels(['%.2f'%k for k in y_ticks], fontsize=10, fontweight='semibold')
        
        # axplot.tick_params(axis = 'both', which = 'major', labelsize = 7, width=1)

        for axis in ['bottom','left']:
            axplot.spines[axis].set_linewidth(1.0)

        
        # # Hide the right and top spines
        axplot.spines['right'].set_visible(False)
        axplot.spines['top'].set_visible(False)

        if i==len(tasks_plot)-1:
            axplot.set_xticks(np.arange(thresholds.shape[0]))
            axplot.set_xticklabels(['%.2f'%k for k in thresholds], fontsize=10, fontweight='semibold')
            axplot.set_xlabel('Percentile Thresholds')
                

    
    lgd = axplot.legend(loc='lower right', fontsize='large')
    lgd.get_frame().set_linewidth(0.0)

    plt.show()

    if savepath is not None:
        # save image
        plt.savefig(savepath)

    return th_mean, th_e



def evaluate_subset_n_complement(task, results_dict, inds_subset):
    '''
    Compute precision, recall, accuracy in the two groups 
    
    '''
    gt = results_dict['gt'][task]
    inds_all = np.arange(gt.shape[0])


    num_samples_n = inds_subset.shape[0]
    inds_complement = np.setdiff1d(inds_all, inds_subset)
    num_samples_o = inds_complement.shape[0]


    # print('num_samples_n', num_samples_n, 'num_samples_o', num_samples_o)
    
    prec_n = gt[inds_subset].sum()/gt[inds_subset].shape[0]
    prec_o = gt[inds_complement].sum()/gt[inds_complement].shape[0]


    recall_n = gt[inds_subset].sum()/gt.sum()
    recall_o = gt[inds_complement].sum()/gt.sum()


    accuracy_n = (np.ones(inds_subset.shape)==gt[inds_subset]).sum()/num_samples_n
    accuracy_o = (np.zeros(inds_complement.shape)==gt[inds_complement]).sum()/num_samples_o

    # print(np.zeros(inds_subset.shape)==gt[inds_complement])

    return [prec_n, recall_n, accuracy_n, num_samples_n], [prec_o, recall_o, accuracy_o, num_samples_o]


