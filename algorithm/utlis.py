import os
import sys
import random
import numpy as np
import pandas as pd
import torch
import json
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score, auc, roc_curve, roc_auc_score, confusion_matrix, \
    balanced_accuracy_score, accuracy_score, average_precision_score

from typing import Dict, List, Tuple

sys.path.append('/remote/rds/groups/idcom_imaging/Users/Jun/Vios/2024/Fairness_Unlearning_main')
from src.models.components.head import *
from dataset_loader import *
from arguments import *

args = parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ModelEnsemble(nn.Module):
    def __init__(self, model_encoder, model_classifier):
        super(ModelEnsemble, self).__init__()
        self.net = model_encoder
        self.head = model_classifier

    def forward(self, x):
        x1 = self.net(x)
        x2 = self.head(x1)
        return x2


def in_ch_dict():
    if args.arch == 'resnet101' or args.arch == 'resnext101' or args.arch == 'inception':
        in_ch = 2048
    elif args.arch == 'densenet':
        in_ch = 2208
    elif args.arch == 'enet':
        in_ch = 1536
    elif args.arch == 'resnet50':
        in_ch = 2048
    elif args.arch == 'resnet34':
        in_ch = 512
    elif args.arch == 'efficient':
        in_ch = 1536
    elif args.arch == 'swin_v2_b':
        in_ch = 1024
    else:
        in_ch = 512

    return in_ch


def load_model(state_dict):
    in_ch = in_ch_dict()

    if args.arch == 'enet':
        model_encoder = EfficientNet()
    elif args.arch == 'resnet101':
        model_encoder = ResNet101()
    elif args.arch == 'resnext101':
        model_encoder = ResNext101()
    elif args.arch == 'densenet':
        model_encoder = DenseNet()
    elif args.arch == 'inception':
        model_encoder = Inception()
    elif args.arch == 'resnet50':
        model_encoder = Extractor()
    elif args.arch == 'resnet34':
        model_encoder = ResNet34()
    elif args.arch == 'efficient':
        model_encoder = EfficientNet()
    elif args.arch == 'swin_v2_b':
        model_encoder = Swin_v2_B()
    else:
        model_encoder = ResNet18()

    model_classifier = ClassificationHead(in_features=in_ch,
                                          out_features=args.out_dim)  # Creating main classification head
    # model_encoder.load_state_dict(state_dict)
    # model_classifier.load_state_dict(state_dict)
    model = ModelEnsemble(model_encoder, model_classifier)
    model.load_state_dict(state_dict)
    # Sending classifier head to GPU
    model = model.to(device)
    return model


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return


def cal_rate(csv):
    group0_size = len(csv[args.attr] == 0)
    group1_size = len(csv[args.attr] == 1)
    group_div = group0_size / group1_size
    if group_div > 1:
        group_div = 1 / group_div
    rate = 100 - np.log(1 + group_div * 100)

    return rate
def kl_loss_sym(x, y):
    kl_loss = nn.KLDivLoss(reduction='batchmean')
    return kl_loss(nn.LogSoftmax(dim=-1)(x), y)


def bias_function(y_pred, y_true, p, metric):
    if metric == 'spd':
        return torch.mean(y_pred[p == 0]) - torch.mean(y_pred[p == 1])
    if metric == 'eodds':
        if isinstance(p, torch.Tensor):
            return torch.mean(y_pred[torch.logical_and(p == 0, y_true == 1)]) - \
                torch.mean(y_pred[torch.logical_and(p == 1, y_true == 1)])
        else:
            return torch.mean(y_pred[np.logical_and(p == 0, y_true == 1)]) - \
                torch.mean(y_pred[np.logical_and(p == 1, y_true == 1)])


def compute_empirical_bias(y_pred, y_true, priv, metric):
    """Evaluates the model's bias empirically on the given data"""

    def zero_if_nan(x):
        if isinstance(x, torch.Tensor):
            return 0. if torch.isnan(x) else x
        else:
            return 0. if np.isnan(x) else x

    gtpr_priv = zero_if_nan(y_pred[priv * y_true == 1].mean())
    gfpr_priv = zero_if_nan(y_pred[priv * (1 - y_true) == 1].mean())
    gtnr_priv = zero_if_nan(y_pred[priv * y_true == 0].mean())
    gfnr_priv = zero_if_nan(y_pred[priv * (1 - y_true) == 0].mean())
    mean_priv = zero_if_nan(y_pred[priv == 1].mean())

    gtpr_unpriv = zero_if_nan(y_pred[(1 - priv) * y_true == 1].mean())
    gfpr_unpriv = zero_if_nan(y_pred[(1 - priv) * (1 - y_true) == 1].mean())
    gtnr_unpriv = zero_if_nan(y_pred[(1 - priv) * y_true == 0].mean())
    gfnr_unpriv = zero_if_nan(y_pred[(1 - priv) * (1 - y_true) == 0].mean())
    mean_unpriv = zero_if_nan(y_pred[(1 - priv) == 1].mean())

    if metric == 'spd':
        return abs(mean_unpriv - mean_priv)
    elif metric == 'eodds':
        return 0.5 * abs((gfpr_unpriv - gfpr_priv) + (gtpr_unpriv - gtpr_priv))
    elif metric == 'eopp':
        return abs(gtpr_unpriv - gtpr_priv)
    elif metric == 'fpr_diff':
        return abs(gfpr_unpriv - gfpr_priv)
    elif metric == 'fnr_diff':
        return abs(gfnr_unpriv - gfnr_priv)
    elif metric == 'tnr_diff':
        return abs(gtnr_unpriv - gtnr_priv)


def compute_accuracy_metrics(preds_vec, labels_vec, logits_vec=None):
    """Computes predictive performance metrics from the provided predictions and ground truth labels"""

    # AUROC
    roc_auc = roc_auc_score(labels_vec, logits_vec)
    # roc_auc = roc_auc_score(labels_vec, preds_vec)

    # Accuracy
    acc = accuracy_score(labels_vec, preds_vec)

    # Average precision score
    average_precision = average_precision_score(labels_vec, preds_vec)

    # Balanced accuracy
    balanced_acc = balanced_accuracy_score(labels_vec, preds_vec)

    # F1-score
    f1_acc = f1_score(labels_vec, preds_vec)

    return acc, roc_auc, average_precision, balanced_acc, f1_acc


def eo_constraint(p, y, a):
    fpr = torch.abs(torch.sum(p * (1 - y) * a) / (torch.sum(a) + 1e-5) - torch.sum(p * (1 - y) * (1 - a)) / (
                torch.sum(1 - a) + 1e-5))
    fnr = torch.abs(torch.sum((1 - p) * y * a) / (torch.sum(a) + 1e-5) - torch.sum((1 - p) * y * (1 - a)) / (
                torch.sum(1 - a) + 1e-5))
    return fpr, fnr


def resnet_block_definition():
    if args.arch == 'resnet18':
        block_layers = [3, 6, 6, 9, 6, 9, 6, 9, 6, 2]
    return block_layers


def l1_regularization(model):
    regulation_loss = 0
    # for param in model.parameters():
    #     regulation_loss += torch.sum(torch.abs(param))
    params_vec = []
    for param in model.parameters():
        params_vec.append(param.view(-1))
    return torch.linalg.norm(torch.cat(params_vec), ord=1)


def adjust_lr(epoch):
    # base_lr * lambda
    # last additional element 0 for preventing indexing error at the last scheduler.step()
    lambda_list = [0.5, 1, 1, 1, 1, 0]  # make lr=[0.0005, 0.001, 0.001, 0.001, 0.001] for each epoch
    return lambda_list[epoch]


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_output(data_loader, net):
    net.eval()
    totalpreds = list()
    with torch.no_grad():
        for sample in data_loader:
            inputs, _, _ = sample
            inputs = inputs.to(device)
            logits = net(inputs)
            for item in logits:
                predictions = torch.sigmoid(item)
                predictions = torch.flatten(predictions)
                predictions = predictions.cpu().detach().numpy()
                totalpreds.extend(predictions)
    return totalpreds


def evaluate(data_loader, net, test_csv=None, save_path=args.save_path):
    if test_csv is None:
        print('You need to identify the test datasets')
        return
    # Get threshold
    # thresh = find_best_thresh(net, 'accuracy')
    thresh = 0.5
    # Get test output
    totalpreds = get_output(data_loader, net)
    # Print results
    eval_output(thresh, totalpreds, test_csv)
    # write CSV
    # save_predict_csv(totalpreds, test_csv, save_path)


def find_best_thresh(net, acc_metric):
    valid_csv = pd.read_csv(args.csv_dir, low_memory=False).reset_index(drop=True)
    valid_dataset = get_dataset(valid_csv, args.attr, transform=None, mode='test')
    valid_data_loader = load_dataset(valid_dataset, batch_size=args.batch_size, shuffle=False)
    totalpreds = get_output(valid_data_loader, net)
    valid_csv['logits'] = totalpreds
    threshs = np.linspace(0, 1, 101)
    y_valid = valid_csv['target']
    valid_pred_scores = valid_csv['logits']
    performances = []
    perf = 0
    for thresh in threshs:
        if acc_metric == 'balanced_accuracy':
            perf = balanced_accuracy_score(y_valid, valid_pred_scores > thresh)
        elif acc_metric == 'accuracy':
            perf = accuracy_score(y_valid, valid_pred_scores > thresh)
        elif acc_metric == 'f1_score':
            perf = f1_score(y_valid, valid_pred_scores > thresh)
        else:
            print('Accuracy metric not defined')
        performances.append(perf)
    best_thresh = threshs[np.argmax(performances)]
    return best_thresh


def eval_output(thresh, prediction, output_csv):
    output_csv['logits'] = prediction
    # output_csv['predict'] = (prediction > thresh)
    output_csv['predict'] = np.round(prediction)
    # output_csv['predict'] = prediction
    try:
        acc, roc_auc, average_precision, balanced_acc, f1_acc = compute_accuracy_metrics(output_csv['predict'],
                                                                                         output_csv['target'],
                                                                                         output_csv['logits'])
        spd = compute_empirical_bias(output_csv['predict'], output_csv['target'], output_csv[args.attr],
                                     metric='spd')
        eopp = compute_empirical_bias(output_csv['predict'], output_csv['target'], output_csv[args.attr],
                                      metric='eopp')
        eodds = compute_empirical_bias(output_csv['predict'], output_csv['target'], output_csv[args.attr],
                                       metric='eodds')
        metrics = [acc, roc_auc, average_precision, balanced_acc, f1_acc, spd, eopp, eodds]
        print(f'The accuracy is: {acc:.5f}')
        print(f'The auc is: {roc_auc:.5f}')
        print(f'The average precision is: {average_precision:.5f}')
        print(f'The balanced accuracy is: {balanced_acc:.5f}')
        print(f'The f1 score is: {f1_acc:.5f}')
        print(f'The demographics parity is: {spd:.5f}')
        print(f'The equal opportunity is: {eopp:.5f}')
        print(f'The equal odds is: {eodds:.5f}')
    except:
        print('All predictions turns into one class, some of metrics cannot be computed')


def save_json(info_dict, save_path):
    info_dict = {key: value.tolist() if torch.is_tensor(value) else value
                         for key, value in info_dict.items()}
    info_json = json.dumps(info_dict, sort_keys=False, indent=4, separators=(',', ': '))
    print(type(info_json))
    f = open(os.path.join(args.output_dir, save_path), 'w')
    f.write(info_json)


def save_predict_csv(prediction, output_csv, save_path):
    # output_csv = output_csv.insert(output_csv.shape[1], 'predict', prediction)
    output_csv.to_csv(os.path.join(args.output_dir, save_path), index=False,
                      encoding='gbk')
