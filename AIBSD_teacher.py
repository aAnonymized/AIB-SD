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
from utils.log_accuracy import *
from utils.setup_configs import *
from utils.utils import *
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
    dataloaders = get_dataloaders(datasets, configs)
    configs.general.num_classes = len(cls_num_list)
    model = get_model(configs)
    optimizer = get_optimizer(configs, model)
    loss_function = nn.CrossEntropyLoss()
    
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
            
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            outputs =outputs.reshape(outputs.shape[0], -1)           
            labels = labels.reshape(outputs.shape[0], -1)
            train_probs[train_k: train_k + outputs.shape[0], :] = outputs.cpu().detach().numpy()
            train_gt[   train_k: train_k + outputs.shape[0]] = labels.cpu().detach().numpy()
            train_k += outputs.shape[0]
        train_pred = np.argmax(train_probs, axis=1)
        # print(f'@@@@@@ train_gt: {train_gt.shape}; train_pred: {train_pred.shape}')
        print(f"train acc:{np.sum(train_gt.squeeze() ==train_pred)/train_k}")
        
        test_loader_nums = len(dataloaders['test'].dataset)
        test_probs = np.zeros((test_loader_nums, configs.general.num_classes), dtype = np.float32)
        test_gt    = np.zeros((test_loader_nums, 1), dtype = np.float32)
        test_k  =0
        for data in tqdm.tqdm(dataloaders['test']):
            model.eval()
            test_inputs, test_labels = data
            if configs.cuda.use_gpu:
                test_inputs = Variable(test_inputs.cuda())
                test_labels = Variable(test_labels.cuda())
            else:
                test_inputs = Variable(test_inputs)
                test_labels = Variable(test_labels)
            with torch.no_grad():
                test_outputs = model(test_inputs)
                test_outputs = test_outputs.reshape(test_outputs.shape[0], -1)           
                test_labels = test_labels.reshape(test_outputs.shape[0], -1)
                test_probs[test_k: test_k + test_outputs.shape[0], :] = test_outputs.cpu().detach().numpy()
                test_gt[   test_k: test_k + test_outputs.shape[0]] = test_labels.cpu().detach().numpy()
                test_k += test_outputs.shape[0]
        test_pred = np.argmax(test_probs, axis=1)
        print(f"test acc:{np.sum(test_gt.squeeze() ==test_pred)/test_k}")
        
        if (np.sum(test_gt.squeeze() ==test_pred)/test_k) >= best_acc:
            print(f'saving model .........')
            best_acc = (np.sum(test_gt.squeeze() ==test_pred)/test_k)
            torch.save(model, 'outputs/%s/%s/best.pt' %(configs.general.dataset_name, configs.general.save_name))
        
if __name__ == '__main__':
    main()
    