import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import sys
import feature_extraction.quantization as quant

def test_main(epoch, task_num, test_loaders,  model_main, savefile_avg, savefile_pertask, OOD_class, \
    feature_name='base.8', quantizer=None, cuda=True, target_ind=1, device=0):
    
    '''
    Test: main classification results
    per task results
    average results
    '''
    print('**************TEST****************')
    model_main.model.eval()
    avg_acc = 0
    num_samples = 0
    accuracy_main = []

    for task_gt, t_loader in enumerate(test_loaders[:task_num+1]):
        print('*********************************************************')
        print('****************Testing task %d************'%(task_gt))
        print('*********************************************************')
        correct = 0
        num_tried = 0
        for count, batch in enumerate(t_loader):
            
            feats = batch[0]
            target = batch[target_ind]

            if cuda:
                feats, target = feats.to(device), target.to(device)


            _,feats = model_main(feats, base_apply=True)
            feats = feats[feature_name]
            if quantizer is not None:
                feats = quant.encode_n_decode(feats.cpu().numpy().astype(np.float32), quantizer['pq'], \
                    num_channels=quantizer['num_channels_init'], spatial_feat_dim=quantizer['spatial_feat_dim'])
                feats = torch.from_numpy(feats).to(torch.float32)
                if cuda:
                    feats = feats.to(device)


            # ======== prediction ===========
            if OOD_class.name == 'odin':
                _, output_main = OOD_class.get_loss(feats, target, base_apply=False)
            else:
                output_main,_ = model_main(feats, base_apply=False) 
                output_main = F.log_softmax(output_main, dim=1)
            pred = output_main.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            # if task_num==1 and task_gt==1:
            #     print('pred', pred.shape)
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            num_tried += pred.shape[0]


        avg_acc += correct.item()
        num_samples += num_tried

        print('\n Per-Task Test set: Accuracy: {%d}/{%d} = {%.4f}\n'%(correct, num_tried, (100. * correct.item() )/num_tried))
            
        accuracy_main[task_gt].append(correct.item()/num_tried)

        output = '%s %d %s %.4f'%(task_num+1, epoch, str(task_gt+1), accuracy_main[task_gt][-1])
        logfile = open(savefile_pertask, 'a+')
        logfile.write(output+"\n")
        logfile.close()

    avg_acc = avg_acc/num_samples
    print('\n ******Test set: Average Accuracy: %.4f*****\n'%(100.* avg_acc))
    output_avg = '%s %d %.4f'%(task_num+1, epoch, avg_acc)
    logfile_ = open(savefile_avg, 'a+')
    logfile_.write(output_avg+"\n")
    logfile_.close()
              
