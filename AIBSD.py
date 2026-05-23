import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import  models, transforms
import numpy as np
from collections import Counter
from torch.utils.data import Dataset
from PIL import Image
import tqdm
import datetime
import os
import random
from sampler import get_sampler
from dataset.get_datasets import get_dataloaders, get_datasets
from models.get_model import get_model
from optimizers.get_optimizer import get_optimizer
from losses.get_loss import get_loss_functions, calculate_loss
from losses.MixUp import mixup_data
from losses.AIBSDLoss import FeatureLoss, grlLoss
from utils.log_accuracy import *
from utils.setup_configs import *
from utils.utils import *
from utils.adv_demisclassification import grl
from models.KNNClassifier import KNNClassifier
# import wandb

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

def main():
    configs = setup_config()
    os.environ['CUDA_VISIBLE_DEVICES'] = configs.cuda.gpu_id
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    set_seed(configs.general.seed)
    print(configs)
    best_acc = 0.0
    save_name =  current_time = '%s_%s_%s_%s_%s_%s_%s_%s_%s' %(
                                                configs.datasets.imbalance_ratio, 
                                                configs.general.method, 
                                                configs.general.img_size,
                                                configs.model.model_name,
                                                configs.model.pretrained,
                                                configs.datasets.batch_size,
                                                configs.general.seed,
                                                configs.general.train_epochs,
                                                configs.datasets.transforms.train
                                                )
    configs.general.save_name = save_name
    outputs_dir = 'outputs/%s/%s/' %(configs.general.dataset_name, configs.general.save_name)
    if os.path.exists(outputs_dir):
        [os.remove(os.path.join(outputs_dir, file_name)) for file_name in os.listdir(outputs_dir)]
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)
    
    datasets = get_datasets(configs)
    cls_num_list = get_cls_num_list(datasets['train'])
    configs.datasets.cls_num_list = cls_num_list
    loss_functions =  get_loss_functions(configs, cls_num_list)
    dataloaders = get_dataloaders(datasets, configs)
    configs.general.num_classes = configs.datasets.tail - configs.datasets.medium
    teacher_model = get_model(configs)
    configs.general.num_classes = len(cls_num_list)
    model = get_model(configs)
    optimizer = get_optimizer(configs, model)
    loss_function = nn.CrossEntropyLoss()
    if configs.cuda.use_gpu:
        feature_criterion = FeatureLoss(feat_dim=model.num_features, proj_dim=768).to("cuda")
        grl_loss = grlLoss(feat_dim=model.num_features, proj_dim=192, grl=grl).to("cuda")
    feature_criterion.train()
    grl_loss.train()
    
    teacher_model.eval()

    for epoch in range(configs.general.train_epochs):
        model.train()
        train_loader_nums = len(dataloaders['train'].dataset)
        train_probs = np.zeros((train_loader_nums, configs.general.num_classes), dtype = np.float32)
        train_gt    = np.zeros((train_loader_nums, 1), dtype = np.float32)
        train_k  =0
        for data in tqdm.tqdm(dataloaders['train']):
            inputs, labels = data
            if configs.cuda.use_gpu:
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())
            else:
                inputs = Variable(inputs)
                labels = Variable(labels)
            
            tail_classes = torch.tensor(list(range(configs.datasets.medium, configs.datasets.tail))).to(labels.device)
            teacher_mask = torch.isin(labels, tail_classes)
            adv_classes = torch.tensor(list(range(configs.datasets.head, configs.datasets.tail))).to(labels.device)
            adv_mask = torch.isin(labels, tail_classes)
            
            if teacher_mask.sum() != 0:
                with torch.no_grad():
                    t_feat = teacher_model.forward_features(inputs[teacher_mask]).detach()
            s_feat = model.forward_features(inputs)
            last_feat = model.forward_head(s_feat, pre_logits=True)
            outputs = model(inputs)
            ce_loss = loss_function(outputs, labels)
            if teacher_mask.sum() != 0:
                align_loss = feature_criterion(t_feat, s_feat, teacher_mask)
                adv_loss = grl_loss(s_feat, adv_mask)
                ce_loss += configs.general.alpha*align_loss
                ce_loss += configs.general.beta*adv_loss
            optimizer.zero_grad()
            ce_loss.backward()
            optimizer.step()
            outputs =outputs.reshape(outputs.shape[0], -1)           
            labels = labels.reshape(outputs.shape[0], -1)
            train_probs[train_k: train_k + outputs.shape[0], :] = outputs.cpu().detach().numpy()
            train_gt[   train_k: train_k + outputs.shape[0]] = labels.cpu().detach().numpy()
            train_k += outputs.shape[0]
        if teacher_mask.sum() != 0:
            print(f'@@@@@@@@@ ce_loss: {ce_loss}; align_loss: {configs.general.alpha*align_loss}; adv_loss: {configs.general.beta*adv_loss}')
        else:
            print(f'@@@@@@@@@ ce_loss: {ce_loss};')
        train_pred = np.argmax(train_probs, axis=1)
        print(f"train acc:{np.sum(train_gt.squeeze() ==train_pred)/train_k}")
        
        best_acc, valid_results = eval(dataloaders, model, configs, epoch, loss_functions, best_acc)

    torch.save(model, 'outputs/%s/%s/last.pt' %(configs.general.dataset_name, configs.general.save_name))

def eval(dataloaders, model, configs, epoch, loss_functions, best_acc):
    if isinstance(model, list):
        for m in model:
            m.eval()
    else:
        model.eval()
    
#     with torch.no_grad():
#         running_loss = 0
#         correct = list(0. for i in range(configs.general.num_classes))
#         total = list(0. for i in range(configs.general.num_classes))
#         all_labels = []
#         all_outputs = []
#         for data in tqdm.tqdm(dataloaders['val']):
#             inputs, labels = data
#             if configs.cuda.use_gpu:
#                 inputs = Variable(inputs.cuda())
#                 labels = Variable(labels.cuda())
#             else:
#                 inputs, labels = Variable(inputs), Variable(labels)
#             if configs.general.method == 'BBN':
#                 feature_a, feature_b = (
#                     model.forward_features(inputs),
#                     model.forward_features(inputs),
#                 )
#                 feature_a, feature_b = (
#                     model.forward_head(feature_a, pre_logits=True),
#                     model.forward_head(feature_b, pre_logits=True),
#                 )

#                 l = 0.5
#                 configs.general.l = l
#                 mixed_feature = 2 * torch.cat((l * feature_a, (1-l) * feature_b), dim=1)
#                 outputs = model.classifier(mixed_feature)
#                 val_loss = calculate_loss(configs, outputs, labels, loss_functions['val'])
#             elif configs.general.method == 'T-Norm':
#                 feature_x = model.forward_features(inputs)
#                 feature_x = model.forward_head(feature_x, pre_logits=True)

#                 weights = model.classifier.weight

#                 ws = pnorm(weights, 2)
#                 outputs = forward(ws, feature_x)
#                 val_loss = calculate_loss(configs, outputs, labels, loss_functions['val'])
#             elif configs.general.method == 'Logits_Adjust_Posthoc':
#                 cls_num_list = get_cls_num_list(dataloaders['train'].dataset)
#                 tau = 1.0
#                 base_probs = [x/max(cls_num_list) for x in cls_num_list]
#                 base_probs = torch.Tensor(base_probs).cuda()
#                 outputs = model(inputs)
#                 outputs = outputs - torch.log((base_probs**tau) + 1e-12)
#                 val_loss = calculate_loss(configs, outputs, labels, loss_functions['val'])
#             elif configs.general.method == 'KNN':
#                 model_ft, knn = model[0], model[1]
#                 feature_x = model_ft.forward_features(inputs)
#                 feature_x = model_ft.forward_head(feature_x, pre_logits=True)
#                 outputs = knn(feature_x)[0]
#                 val_loss = calculate_loss(configs, outputs, labels, loss_functions['val'])
#             elif configs.general.method == 'RSG':
#                 outputs = model(inputs, phase_train=False)
#                 val_loss = calculate_loss(configs, outputs, labels, loss_functions['val'])
#             elif configs.general.method == 'SADE':
#                 outputs = model(inputs)
#                 outputs = outputs['output']
#                 val_loss = calculate_loss(configs, outputs, labels, loss_functions['val'])
#             else:
#                 outputs = model(inputs)
#                 val_loss = calculate_loss(configs, outputs, labels, loss_functions['val'])
#             all_outputs.append(outputs.cpu())
#             all_labels.append(labels.cpu())
#             correct, total = calculate_metrics_single(labels, outputs, correct, total)
#             running_loss += val_loss.data.item()

#         val_epoch_loss = running_loss / len(dataloaders['val'])
#         valid_results = calculate_metrics(configs, all_outputs, all_labels)
#         valid_results['epoch'] = epoch
#         valid_results['status'] = 'val'
#         valid_results['loss'] = val_epoch_loss
        
#         # wandb.log({'val_logs': results})
#         print(valid_results)
#         log_results(configs, valid_results)
        
        
    with torch.no_grad():
        running_loss = 0
        correct = list(0. for i in range(configs.general.num_classes))
        total = list(0. for i in range(configs.general.num_classes))
        all_labels = []
        all_outputs = []
        for data in tqdm.tqdm(dataloaders['test']):
            inputs, labels = data
            if configs.cuda.use_gpu:
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)
            if configs.general.method == 'BBN':
                feature_a, feature_b = (
                    model.forward_features(inputs),
                    model.forward_features(inputs),
                )
                feature_a, feature_b = (
                    model.forward_head(feature_a, pre_logits=True),
                    model.forward_head(feature_b, pre_logits=True),
                )

                l = 0.5
                configs.general.l = l
                mixed_feature = 2 * torch.cat((l * feature_a, (1-l) * feature_b), dim=1)
 
                outputs = model.classifier(mixed_feature)
                test_loss = calculate_loss(configs, outputs, labels, loss_functions['val'])
            elif configs.general.method == 'T-Norm':
                feature_x = model.forward_features(inputs)
                feature_x = model.forward_head(feature_x, pre_logits=True)
                weights = model.classifier.weight
                ws = pnorm(weights, 2)
                outputs = forward(ws, feature_x)
                test_loss = calculate_loss(configs, outputs, labels, loss_functions['val'])
            elif configs.general.method == 'Logits_Adjust_Posthoc':
                cls_num_list = get_cls_num_list(dataloaders['train'].dataset)
                tau = 1.0
                base_probs = [x/max(cls_num_list) for x in cls_num_list]
                base_probs = torch.Tensor(base_probs).cuda()
                outputs = model(inputs)
                outputs = outputs - torch.log((base_probs**tau) + 1e-12)
                test_loss = calculate_loss(configs, outputs, labels, loss_functions['val'])
            elif configs.general.method == 'KNN':
                model_ft, knn = model[0], model[1]
                feature_x = model_ft.forward_features(inputs)
                feature_x = model_ft.forward_head(feature_x, pre_logits=True)
                outputs = knn(feature_x)[0]
                test_loss = calculate_loss(configs, outputs, labels, loss_functions['val'])
            elif configs.general.method == 'RSG':
                outputs = model(inputs, phase_train=False)
                test_loss = calculate_loss(configs, outputs, labels, loss_functions['val'])
            elif configs.general.method == 'SADE':
                outputs = model(inputs)
                outputs = outputs['output']
                test_loss = calculate_loss(configs, outputs, labels, loss_functions['val'])
            else:
                outputs = model(inputs)
                test_loss = calculate_loss(configs, outputs, labels, loss_functions['val'])
            all_outputs.append(outputs.cpu())
            all_labels.append(labels.cpu())
            correct, total = calculate_metrics_single(labels, outputs, correct, total)
            running_loss += test_loss.data.item()

        test_epoch_loss = running_loss / len(dataloaders['test'])
        test_results = calculate_metrics(configs, all_outputs, all_labels)
        test_results['epoch'] = epoch
        test_results['status'] = 'test'
        test_results['loss'] = test_epoch_loss
        
        current_acc = test_results['accuracy'][3]
        if  current_acc> best_acc:
            print('Best acc: %s, current acc: %s. Saving best model...' %(round(best_acc, 4), round(current_acc, 4)))
            best_acc = current_acc
            torch.save(model, f'outputs/%s/%s/best.pt' %(configs.general.dataset_name, configs.general.save_name))
        
        # wandb.log({'val_logs': results})
        print(test_results)
        log_results(configs, test_results)
    return best_acc, test_results
    
if __name__ == '__main__':
    main()

