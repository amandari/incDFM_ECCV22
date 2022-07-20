
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import statistics 
from matplotlib import pyplot
import seaborn as sn
import pandas as pd
import matplotlib
import numpy as np
import os
import sys
import json
from argparse import Namespace




class save_novelty_results_test():
    def __init__(self, num_tasks, name_experiment, dir_save):
        self.num_tasks = num_tasks
        self.name_experiment = name_experiment
        self.dir_save = dir_save

        # save AUROC/AUPR vals to text file 
        self.novelty_txt = '%s/novelty_eval_test.txt'%(dir_save)

        output = '- %s %s %s'%('AUROC', 'AUPR', 'AUPR_NORM')
        logfile = open(self.novelty_txt, 'a+')
        logfile.write(output+"\n")
        logfile.close()


        self.auroc=[]
        self.aupr=[]
        self.aupr_norm=[]
        self.scores = []
        self.scores_dist = np.array([]) # will be converted to 2D array


    def compute(self, t, ground_truth_novelty, scores):
        '''
        ground_truth_novelty is novelty.
        Typically (for DFM), novelty=1, old=0
        For odin, novelty=0, old=1
        '''
        
        metrics_curves = scores_metrics_results(scores, ground_truth_novelty)
        
        if metrics_curves is not None:
            auroc, aupr, aupr_norm, _,_,_,_ = metrics_curves
        else:
            print('Issue with AUROC computation')
            auroc, aupr, aupr_norm = 0,0,0
            
        self.auroc.append(auroc)
        self.aupr.append(aupr)
        self.aupr_norm.append(aupr_norm)
        print('New task {}, AUROC = {:0.3f}, AUPR = {:.3f}, AUPR_NORM = {:.3f}'.format(t, auroc, aupr, aupr_norm))

        output = '%d %.4f %.4f %.4f'%(t, auroc, aupr, aupr_norm)
        logfile = open(self.novelty_txt, 'a+')
        logfile.write(output+"\n")
        logfile.close()
        
        return auroc, aupr, aupr_norm






class save_novelty_results():
    def __init__(self, num_tasks, name_experiment, dir_save):
        self.num_tasks = num_tasks
        self.name_experiment = name_experiment
        self.dir_save = dir_save

        # save AUROC/AUPR vals to text file 
        self.novelty_txt = '%s/novelty_eval.txt'%(dir_save)

        self.novelty_txt_acc_iter = '%s/novelty_accuracies_iter.txt'%(dir_save)

        output = '- %s %s %s'%('AUROC', 'AUPR', 'AUPR_NORM')
        logfile = open(self.novelty_txt, 'a+')
        logfile.write(output+"\n")
        logfile.close()


        output = '- %s %s %s %s'%('ACC', 'ACC_N', 'P_N', 'R_N')
        logfile = open(self.novelty_txt_acc_iter, 'a+')
        logfile.write(output+"\n")
        logfile.close()


        self.auroc=[]
        self.aupr=[]
        self.aupr_norm=[]
        self.scores = []
        self.scores_dist = np.array([]) # will be converted to 2D array


    def compute_from_preds(self, t, scores, threshold, ground_truth_novelty):
        
        preds = np.zeros(scores.shape[0])
        preds[scores>=threshold]=1 # new
        accuracy = accuracy_score(ground_truth_novelty, preds)

        sk_tn, sk_fp, sk_fn, sk_tp = confusion_matrix(ground_truth_novelty, preds).ravel()
        print("Accuracy Score: {}".format(accuracy))
        print("Verify our results using Sklearn confusion matrix values\n"
            "FN: {0}\n"
            "TP: {1}\n"
            "TN: {2}\n"
            "FP: {3}".format(sk_fn, sk_tp, sk_tn, sk_fp,))

        data = [[sk_tp, sk_tn],[sk_fp, sk_fn]]

        df_cm = pd.DataFrame(data, index = [i for i in 'TF'],
                  columns = [i for i in 'PN'])
        pyplot.figure(figsize = (10,7))
        sn.heatmap(df_cm, annot=True, fmt='.4g')
        pyplot.savefig(os.path.join(self.dir_save, 'Confusion_Matrix_task{}.png'.format(t)))

        return accuracy, sk_tn, sk_fp, sk_fn, sk_tp


    def compute_accuracies(self, task, iter_task, inds_pred_novel, gt_novelty):
        '''
        inds_pred_novel (ndarray): indices predicted as novel
        gt_novelty (ndarray): ground truth labels 
        '''

        labels_pred = np.zeros(gt_novelty.shape)
        
        if inds_pred_novel.shape[0]>0:
            labels_pred[inds_pred_novel]=1

            # accuracy overall (binary threshold) 
            acc_overall = np.sum(labels_pred==gt_novelty)/gt_novelty.shape[0]
            
            # accuracy novelty
            acc_novel = np.sum(labels_pred[inds_pred_novel]==gt_novelty[inds_pred_novel])/inds_pred_novel.shape[0]
            prec_novel = np.sum(gt_novelty[inds_pred_novel])/gt_novelty[inds_pred_novel].shape[0]
            recall_novel = np.sum(gt_novelty[inds_pred_novel])/np.sum(gt_novelty)
        else:
            acc_overall = np.sum(labels_pred==gt_novelty)/gt_novelty.shape[0]
            acc_novel=0
            prec_novel=0
            recall_novel=0


        # Save to txt 
        output = '%d %d %.4f %.4f %.4f %.4f'%(task, iter_task, acc_overall, acc_novel, prec_novel, recall_novel)
        logfile = open(self.novelty_txt_acc_iter, 'a+')
        logfile.write(output+"\n")
        logfile.close()


        return acc_overall, acc_novel, prec_novel, recall_novel
    



    def compute(self, t, ground_truth_novelty, scores):
        '''
        ground_truth_novelty is novelty.
        Typically (for DFM), novelty=1, old=0
        For odin, novelty=0, old=1
        '''

        
        auroc, aupr, aupr_norm, fpr, tpr, prec, recall = scores_metrics_results(scores, ground_truth_novelty)
        self.auroc.append(auroc)
        self.aupr.append(aupr)
        self.aupr_norm.append(aupr_norm)
        print('New task {}, AUROC = {:0.3f}, AUPR = {:.3f}, AUPR_NORM = {:.3f}'.format(t, auroc, aupr, aupr_norm))

        output = '%d %.4f %.4f %.4f'%(t, auroc, aupr, aupr_norm)
        logfile = open(self.novelty_txt, 'a+')
        logfile.write(output+"\n")
        logfile.close()

        pyplot.close('all')
        new_scores = scores[ground_truth_novelty == 1]
        old_scores = scores[ground_truth_novelty == 0]
        print('new_scores', new_scores)
        print('old_scores', old_scores)
        pyplot.hist([new_scores, old_scores], bins=20, density=True)
        pyplot.legend(['New', 'Old'])
        pyplot.grid()
        pyplot.title('Task {}'.format(t))
        pyplot.tight_layout()
        pyplot.savefig(os.path.join(self.dir_save, 'Distributions_Old_vs_New_Task_{}.png'.format(t)))
        # pyplot.close()

        if t == self.num_tasks-1:
            pyplot.figure()
            pyplot.title('ROC Curve for Novelty Prediction (last task)')
            pyplot.plot(fpr, tpr, marker='.', label=self.name_experiment, color='r')
            pyplot.plot(np.arange(fpr.shape[0]), np.arange(fpr.shape[0]), linestyle='--', label='B', color='b')
            pyplot.xlabel('False Positive Rate')
            pyplot.ylabel('True Positive Rate')
            pyplot.ylim([-0.01,1.01])
            pyplot.xlim([-0.01,1.01])
            pyplot.legend()
            pyplot.savefig('%s/AUROC_curve.png'%self.dir_save)

            pyplot.figure()
            pyplot.title('PR Curve for Novelty Prediction (last task)')
            pyplot.plot(recall, prec, marker='.', label=self.name_experiment, color='g')
            pyplot.ylim([-0.01,1.01])
            pyplot.xlim([-0.01,1.01])
            pyplot.xlabel('Recall')
            pyplot.ylabel('Precision')
            pyplot.legend()
            pyplot.savefig('%s/AUPR_curve.png'%self.dir_save)


            pyplot.figure()
            pyplot.title('AUPR and AUROC evolution through continual tasks')
            pyplot.plot(np.arange(1,len(self.auroc)+1), self.auroc, marker='.', label='AUROC', color='r')
            pyplot.plot(np.arange(1,len(self.aupr)+1), self.aupr, marker='.', label='AUPR', color='g')
            pyplot.plot(np.arange(1,len(self.aupr_norm)+1), self.aupr_norm, marker='.', label='AUPR_NORM', color='y')
            pyplot.ylim([-0.01,1.01])
            pyplot.xlim([-0.01+1,0.01+self.num_tasks-1])
            pyplot.xlabel('Tasks')
            pyplot.ylabel('Score')
            pyplot.legend()
            pyplot.savefig('%s/Scores_tasks_time.png'%self.dir_save)
            





def acuracy_report(y_true, y_pred, scores):

    report = classification_report(y_true, y_pred, output_dict =True)
    classes = np.unique(np.concatenate((y_true, y_pred)))

    for c in classes:
        key_c = str(c)

        inds = np.where(y_true==c)[0]

        if inds.shape[0]>0:

            # mode_c = float(statistics.mode(y_pred[inds]))
            avg_c = np.mean(scores[inds])
        else:
            avg_c=0.0
            # mode_c=0.0
            
        report[key_c]['avg_score']=avg_c
        # report[key_c]['pred_mode']=mode_c
    # include per class mode (most predicted mode)
    # include per class average value (average score)
    return report




def scores_metrics_results(scores, ground_truth_novelty, mult=1):
    '''
    ground_truth_novelty is novelty 0, 1
    '''

    try:
        fpr, tpr, _ = roc_curve(ground_truth_novelty, scores)
        prec, recall, _ = precision_recall_curve(ground_truth_novelty, scores)
        auroc = auc(fpr, tpr)
        aupr = auc(recall, prec)

        aupr_norm = aupr_normalized(ground_truth_novelty, scores, mult=mult)

    except ValueError:
        return None

    return auroc, aupr, aupr_norm, fpr, tpr, prec, recall





def aupr_normalized(ground_truth_novelty, scores, mult=1):
    '''
    ground_truth is novelty 0, 1
    '''
    new_inds = np.where(ground_truth_novelty==1)[0]
    n_new = new_inds.shape[0]

    old_inds = np.where(ground_truth_novelty==0)[0]
    n_old = old_inds.shape[0]

    old_inds_keep = old_inds[np.random.permutation(np.arange(n_old))[:min(int(n_new*mult),n_old)]]

    old_inds_keep.sort()
    inds_keep = np.concatenate((new_inds, old_inds_keep))

    ground_truth = ground_truth_novelty[inds_keep]
    scores = scores[inds_keep]
    
    prec, recall, _ = precision_recall_curve(ground_truth, scores)
    aupr_norm = auc(recall, prec)

    return aupr_norm



# ------------------ For when Old and New data are in same loader 


def scores_incomming_task(t, incomming_loader, novelty_detector, params_score):
    '''
    Gets scores for incomming task composed of Novel and IID Old Samples 
    '''
    
    gt, gt_novelty, scores, scores_dist, preds, dset_inds = novelty_detector.score(incomming_loader, params_score)


    # separate into Old IID and New 
    inds_old = np.where(gt_novelty==0)[0]
    inds_new = np.where(gt_novelty==1)[0]

    # print('true novel scores', scores[inds_new], scores[inds_new].mean())
    # print('true old scores', scores[inds_old], scores[inds_old].mean())
    
    out = {}
    out['new_scores'] = scores[inds_new]
    out['old_scores'] = scores[inds_old]
    out['scores_dist'] = scores_dist
    out['gt']= gt
    out['gt_novelty']= gt_novelty
    out['dset_inds'] = dset_inds

    return Namespace(**out)





def evaluate_simple_CL_test(t, novelty_detector, noveltyResults_test, params_score, incomming_loader, novelty_detector_name):
    
    '''
    For ODIN do: 
    
    params_score['score_func'] = score_func

    params_score['noise_magnitude'] = noise_magnitude
    
    '''
    gt, gt_novelty, scores, scores_dist, preds, dset_inds = novelty_detector.score(incomming_loader, params_score)
    
    if novelty_detector_name=='softmax' or novelty_detector_name=='odin':
        scores = -scores

    # plot results for best epsilon 
    auroc, aupr, aupr_norm = noveltyResults_test.compute(t, gt_novelty, scores)
    
    # print('Results -  Auroc %.3f, Aupr %.3f, Aupr_norm %.3f'%(auroc, aupr, aupr_norm))
    



def evaluate_simple_CL(t, novelty_detector, noveltyResults, params_score, incomming_loader, novelty_detector_name):

    results = scores_incomming_task(t, incomming_loader, novelty_detector, params_score)
    
    
    if novelty_detector_name=='softmax' or novelty_detector_name=='odin':
        results.new_scores = -results.new_scores
        results.old_scores = -results.old_scores
        results.scores_dist = -results.scores_dist

    results.scores = np.concatenate((results.new_scores, results.old_scores))
    
    metrics_curves = scores_metrics_results(results.scores, results.gt_novelty)

    if metrics_curves is not None:
        auroc, aupr, aupr_norm, _,_,_,_ = metrics_curves
    else:
        print('Issue with AUROC computation')
        auroc, aupr, aupr_norm = 0,0,0

    # plot results for best epsilon 
    noveltyResults.compute(t, results.gt_novelty, results.scores)
    dir_save = noveltyResults.dir_save

    results.auroc = auroc
    results.aupr = aupr
    results.aupr_norm = aupr_norm

    # save per class results 
    # with open('%s/per_class_report_old.txt'%dir_save, 'w') as outfile:
    #     json.dump(results.report, outfile, sort_keys = True, indent = 4)

    print('Results -  Auroc %.3f, Aupr %.3f, Aupr_norm %.3f'%(auroc, aupr, aupr_norm))
    # print('Average Accuracy per class old %.4f'% results.accuracy)

    return results



def evaluate_odin_CL(t, novelty_detector, noveltyResults, params_score, incomming_loader):
    '''
    Choose between best score_func and noise_magnitude
    There is no validation set in this CL experiment. 
    Only save overall best auroc scores and others 
    '''

    best_auroc_overall = 0
    best_aupr_overall = 0
    best_aupr_norm_overall = 0
    best_noise_magnitude = 0.0
    best_score_func = 'h'
    best_results=None

    # for score_func in ['h', 'logit', 'g']: 
    for score_func in ['h']:
        print('********Score func: %s *************'%score_func)
        params_score['score_func'] = score_func
        # should make distinction of val and test 
        for noise_magnitude in[0.005]:
        # for noise_magnitude in [0.0025, 0.005, 0.01, 0.02, 0.04, 0.08]:
            print(f'Noise magnitude {noise_magnitude:.5f}')
            params_score['noise_magnitude'] = noise_magnitude


            results = scores_incomming_task(t, incomming_loader, novelty_detector, params_score)
            results.new_scores = -results.new_scores
            results.old_scores = -results.old_scores
            results.scores_dist = -results.scores_dist

            results.scores = np.concatenate((results.new_scores, results.old_scores))
            print('scores', results.scores)
            metrics_curves = scores_metrics_results(results.scores, results.gt_novelty)


            if metrics_curves is not None:
                auroc, aupr, aupr_norm, _,_,_,_ = metrics_curves
            else:
                print('Issue with AUROC computation')
                auroc, aupr, aupr_norm = 0,0,0
                continue

            print('AUROC:', auroc, 'aupr', aupr, 'aupr_norm', aupr_norm)

            # ------- save best scores 
            if auroc >= best_auroc_overall:
                best_auroc_overall = auroc
                best_aupr_overall = aupr
                best_aupr_norm_overall = aupr_norm
                best_noise_magnitude = noise_magnitude
                best_score_func = score_func
                best_results = results
        #     break
        # break

    # results for best epsilon 
    best_results.scores = np.concatenate((best_results.new_scores, best_results.old_scores))
    noveltyResults.compute(t, best_results.gt_novelty, best_results.scores)
    dir_save = noveltyResults.dir_save
    best_results.auroc = best_auroc_overall
    best_results.aupr = best_aupr_overall
    best_results.aupr_norm = best_aupr_norm_overall
    best_results.best_score_func = best_score_func
    best_results.best_noise_magnitude = best_noise_magnitude

    print('best scores', best_results.scores)

    # Log which h,logit,g + noise yielded best result through epoch
    output = '%d %s %.4f'%(t, best_score_func, best_noise_magnitude)
    logfile = open('%sODIN_bestparams.txt'%(dir_save), 'a+')
    logfile.write(output+"\n")
    logfile.close()

    # save per class results 
    # with open('%s/per_class_report_old.txt'%dir_save, 'w') as outfile:
    #     json.dump(best_results.report, outfile, sort_keys = True, indent = 4)

    print('ODIN Best Values [Score_func %s -Noise %.3f]--> Best overall: Auroc %.3f, Aupr %.3f, Aupr_norm %.3f'%(\
        best_score_func, best_noise_magnitude, \
        best_auroc_overall, best_aupr_overall, best_aupr_norm_overall))

    return best_results



def evaluate_dfm_CL(t, novelty_detector, noveltyResults, params_score, incomming_loader):

    results = scores_incomming_task(t, incomming_loader, novelty_detector, params_score)
    
    results.scores = np.concatenate((results.new_scores, results.old_scores))
    metrics_curves = scores_metrics_results(results.scores, results.gt_novelty)

    if metrics_curves is not None:
        auroc, aupr, aupr_norm, _,_,_,_ = metrics_curves
    else:
        print('Issue with AUROC computation')
        auroc, aupr, aupr_norm = 0,0,0

    # plot results for best epsilon 
    noveltyResults.compute(t, results.gt_novelty, results.scores)
    dir_save = noveltyResults.dir_save

    results.auroc = auroc
    results.aupr = aupr
    results.aupr_norm = aupr_norm

    # save per class results 
    # with open('%s/per_class_report_old.txt'%dir_save, 'w') as outfile:
    #     json.dump(results.report, outfile, sort_keys = True, indent = 4)

    print('Results -  Auroc %.3f, Aupr %.3f, Aupr_norm %.3f'%(auroc, aupr, aupr_norm))
    # print('Average Accuracy per class old %.4f'% results.accuracy)

    return results


# -------------------------------For when old and new data are in different loaders -----------------


def get_scores_odd(t, novelty_detector, params_score, new_train_loader, old_loaders_test):
    _, _, new_scores, new_scores_dist, _,_ = novelty_detector.score(new_train_loader, params_score)

    for  task_idx, old_test_loader in enumerate(old_loaders_test[:t]):
        gt, _,scores, scores_dist, pred,_ = novelty_detector.score(old_test_loader, params_score)
        if task_idx == 0:
            old_gt = gt
            old_scores = scores
            old_preds = pred
            old_scores_dist = scores_dist
        else:
            old_gt = np.concatenate((old_gt, gt))
            old_scores = np.concatenate((old_scores, scores))
            old_preds = np.concatenate((old_preds, pred))
            old_scores_dist = np.concatenate((old_scores_dist, scores_dist), axis=1)

    print('old_scores_dist', old_scores_dist.shape)
    # rep_old = acuracy_report(old_gt, old_preds, old_scores)
    
    out = {}
    out['new_scores'] = new_scores
    out['old_scores'] = old_scores
    out['new_scores_dist'] = new_scores_dist
    out['old_scores_dist'] = old_scores_dist

    out['gt_novelty_concat']= np.ones(new_scores.shape[0]+old_scores.shape[0])
    out['gt_novelty_concat'][new_scores.shape[0]:] = 0

    # out['accuracy'] = rep_old['accuracy']
    # out['report'] = rep_old

    return Namespace(**out)



