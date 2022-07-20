
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

class save_novelty_results():
    def __init__(self, num_tasks, name_experiment, dir_save):
        self.num_tasks = num_tasks
        self.name_experiment = name_experiment
        self.dir_save = dir_save

        # save AUROC/AUPR vals to text file 
        self.novelty_txt = '%s/novelty_eval.txt'%(dir_save)
        output = '- %s %s %s'%('AUROC', 'AUPR', 'AUPR_NORM')
        logfile = open(self.novelty_txt, 'a+')
        logfile.write(output+"\n")
        logfile.close()

        self.auroc=[]
        self.aupr=[]
        self.aupr_norm=[]

    def compute_from_preds(self, t, scores, threshold, true_labels):
        
        preds = np.zeros(scores.shape[0])
        # preds[scores>threshold]=0 # predicted as new
        preds[scores<=threshold]=1 # old
        accuracy = accuracy_score(true_labels, preds)

        sk_tn, sk_fp, sk_fn, sk_tp = confusion_matrix(true_labels, preds).ravel()
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


    def compute(self, t, ground_truth, scores):

        auroc, aupr, aupr_norm, fpr, tpr, prec, recall = scores_metrics_results(scores, ground_truth)
        self.auroc.append(auroc)
        self.aupr.append(aupr)
        self.aupr_norm.append(aupr_norm)
        print('New task {}, AUROC = {:0.3f}, AUPR = {:.3f}, AUPR_NORM = {:.3f}'.format(t, auroc, aupr, aupr_norm))

        output = '%d %.4f %.4f %.4f'%(t, auroc, aupr, aupr_norm)
        logfile = open(self.novelty_txt, 'a+')
        logfile.write(output+"\n")
        logfile.close()

        pyplot.close('all')
        new_scores = scores[ground_truth == 1]
        old_scores = scores[ground_truth == 0]
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

            mode_c = float(statistics.mode(y_pred[inds]))
            avg_c = np.mean(scores[inds])
        else:
            avg_c=0.0
            mode_c=0.0
            
        report[key_c]['avg_score']=avg_c
        report[key_c]['pred_mode']=mode_c
    # include per class mode (most predicted mode)
    # include per class average value (average score)
    return report


def scores_metrics_results(scores, ground_truth, mult=1):

    fpr, tpr, _ = roc_curve(ground_truth, scores)
    prec, recall, _ = precision_recall_curve(ground_truth, scores)
    auroc = auc(fpr, tpr)
    aupr = auc(recall, prec)

    aupr_norm = aupr_normalized(ground_truth, scores, mult=mult)

    return auroc, aupr, aupr_norm, fpr, tpr, prec, recall



def aupr_normalized(ground_truth, scores, mult=1):

    new_inds = np.where(ground_truth==1)[0]
    n_new = new_inds.shape[0]

    old_inds = np.where(ground_truth==0)[0]
    n_old = old_inds.shape[0]

    old_inds_keep = old_inds[np.random.permutation(np.arange(n_old))[:min(int(n_new*mult),n_old)]]

    old_inds_keep.sort()
    inds_keep = np.concatenate((new_inds, old_inds_keep))

    ground_truth = ground_truth[inds_keep]
    scores = scores[inds_keep]
    
    prec, recall, _ = precision_recall_curve(ground_truth, scores)
    aupr_norm = auc(recall, prec)

    return aupr_norm

def get_scores_odd(t, novelty_detector, params_score, new_train_loader, old_loaders_test):
    
    gt_new, new_scores, new_preds = novelty_detector.score(new_train_loader, params_score)
    # print('gt_new',gt_new)
    # print('pred_new', new_preds)
    # print('new_scores',new_scores)
    rep_new = acuracy_report(gt_new, new_preds, new_scores)
    acc_new = rep_new['accuracy']
    # acc_new = np.sum(gt_new== new_preds)/gt_new.shape[0]
    print('accuracy new', acc_new)
    for  task_idx, old_test_loader in enumerate(old_loaders_test[:t]):
        gt, scores, pred = novelty_detector.score(old_test_loader, params_score)
        if task_idx == 0:
            old_gt = gt
            old_scores = scores
            old_preds = pred
        else:
            old_gt = np.concatenate((old_gt, gt))
            old_scores = np.concatenate((old_scores, scores))
            old_preds = np.concatenate((old_preds, pred))
    # print('gt_old',old_gt)
    # print('pred_old', old_preds)
    # print('old_scores', old_scores)

    rep_old = acuracy_report(old_gt, old_preds, old_scores)
    acc_old = rep_old['accuracy']
    # acc_old =  np.sum(old_gt== old_preds)/old_gt.shape[0]
    print('accuracy old', acc_old)
    # sys.exit()
    # print('Task {}, old scores length {}'.format(t,old_scores.shape[0]))
    scores_concat = np.concatenate((new_scores, old_scores))
    gt_concat = np.ones(scores_concat.shape[0])
    gt_concat[new_scores.shape[0]:] = 0

    accuracies = [acc_new, acc_old]
    reports =[rep_new, rep_old]

    return new_scores, old_scores, gt_concat, accuracies, reports 



def evaluate_odin(t, novelty_detector, noveltyResults, params_score, new_train_loader, old_loaders_test):
    best_auroc_overall = 0
    best_aupr_overall = 0
    best_aupr_norm_overall = 0
    best_val_score = None
    # should make distinction of val and test 
    for noise_magnitude in [0, 0.0025, 0.005, 0.01, 0.02, 0.04, 0.08]:
        print(f'Noise magnitude {noise_magnitude:.5f}')

        novelty_detector.noise_magnitude = noise_magnitude
        new_scores, old_scores, gt_concat, accs_class, per_class_reports = get_scores_odd(t, novelty_detector, params_score, new_train_loader, old_loaders_test)
        scores_concat = np.concatenate((new_scores, old_scores))
        auroc, aupr, aupr_norm, _,_,_,_ = scores_metrics_results(scores_concat, gt_concat)

        print('AUROC:', auroc, 'aupr', aupr, 'aupr_norm', aupr_norm)
        # print('Average Accuracy per class old %.4f'% accs_class[-1])
        
        best_auroc_overall = max(best_auroc_overall, auroc)
        best_aupr_overall = max(best_aupr_overall, aupr)
        best_aupr_norm_overall = max(best_aupr_norm_overall, aupr_norm)


        if best_val_score is None or old_scores.mean() > best_val_score:
            best_val_score = old_scores.mean()
            best_val_auroc = auroc
            best_val_aupr = aupr
            best_val_aupr_norm = aupr_norm
            best_noise_magnitude = noise_magnitude
            gt_concat_best = gt_concat
            scores_concat_best = scores_concat

            per_class_reports_best = per_class_reports

    # plot results for best epsilon 
    noveltyResults.compute(t, gt_concat_best, scores_concat_best)


    dir_save = noveltyResults.dir_save
    # save per class results 
    with open('%s/per_class_report_new.txt'%dir_save, 'w') as outfile:
        json.dump(per_class_reports_best[0], outfile, sort_keys = True, indent = 4)
    with open('%s/per_class_report_old.txt'%dir_save, 'w') as outfile:
        json.dump(per_class_reports_best[1], outfile, sort_keys = True, indent = 4)

    print('ODIN Best Values - Noise %.3f: Auroc %.3f, Aupr %.3f, Aupr_norm %.3f| Best overall: Auroc %.3f, Aupr %.3f, Aupr_norm %.3f'%(best_noise_magnitude, \
        best_val_auroc, best_val_aupr, best_val_aupr_norm, best_auroc_overall, best_aupr_overall, best_aupr_norm_overall))



def evaluate_dfm(t, novelty_detector, noveltyResults, params_score, new_train_loader, old_loaders_test):

    new_scores, old_scores, gt_concat, accs_class, per_class_reports = get_scores_odd(t, novelty_detector, params_score, new_train_loader, old_loaders_test)
    scores_concat = np.concatenate((new_scores, old_scores))
    auroc, aupr, aupr_norm, _,_,_,_ = scores_metrics_results(scores_concat, gt_concat)

    print('auroc', auroc)
    
    # plot results for best epsilon 
    noveltyResults.compute(t, gt_concat, scores_concat)

    dir_save = noveltyResults.dir_save

    # save per class results 
    with open('%s/per_class_report_new.txt'%dir_save, 'w') as outfile:
        json.dump(per_class_reports[0], outfile, sort_keys = True, indent = 4)
    with open('%s/per_class_report_old.txt'%dir_save, 'w') as outfile:
        json.dump(per_class_reports[1], outfile, sort_keys = True, indent = 4)

    print('DFM Results -  Auroc %.3f, Aupr %.3f, Aupr_norm %.3f'%(auroc, aupr, aupr_norm))
    print('Average Accuracy per class old %.4f'% accs_class[-1])


