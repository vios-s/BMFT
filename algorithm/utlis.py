import os
import sys
import numpy as np
import pandas as pd
import torch
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


def load_model(state_dict):
    if args.arch == 'resnet101' or args.arch == 'resnext101' or args.arch == 'inception':
        in_ch = 2048
    elif args.arch == 'densenet':
        in_ch = 2208
    else:
        in_ch = 1536

    if args.arch == 'enet':
        model_encoder = enetv2(args.enet_type)
    if args.arch == 'resnet101':
        model_encoder = ResNet101()
    if args.arch == 'resnext101':
        model_encoder = ResNext101()
    if args.arch == 'densenet':
        model_encoder = DenseNet()
    if args.arch == 'inception':
        model_encoder = Inception()

    model_classifier = ClassificationHead(in_features=in_ch,
                                          out_features=args.out_dim)  # Creating main classification head
    # model_encoder.load_state_dict(state_dict)
    # model_classifier.load_state_dict(state_dict)
    model = ModelEnsemble(model_encoder, model_classifier)
    model.load_state_dict(state_dict)
    # Sending classifier head to GPU
    model = model.to(device)
    return model

def kl_loss_sym(x, y):
    kl_loss = nn.KLDivLoss(reduction='batchmean')
    return kl_loss(nn.LogSoftmax(dim=-1)(x), y)


def compute_empirical_bias(y_pred, y_true, priv, metric):
    """Evaluates the model's bias empirically on the given data, used by the adversarial intra-processing algorithm"""
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


def compute_accuracy_metrics(preds_vec, labels_vec):
    """Computes predictive performance metrics from the provided predictions and ground truth labels"""
    # Accuracy
    acc = accuracy_score(labels_vec, preds_vec)

    # AUROC
    roc_auc = roc_auc_score(labels_vec, preds_vec)

    # Average precision score
    average_precision = average_precision_score(labels_vec, preds_vec)

    # Balanced accuracy
    balanced_acc = balanced_accuracy_score(labels_vec, preds_vec)

    # F1-score
    f1_acc = f1_score(labels_vec, preds_vec)

    return acc, roc_auc, average_precision, balanced_acc, f1_acc


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


def save_predict_csv(prediction, data_path, save_path):
    output_csv = pd.read_csv(data_path, low_memory=False).reset_index(drop=True)
    # output_csv = output_csv.insert(output_csv.shape[1], 'predict', totalpreds)
    output_csv['predict'] = prediction
    acc, roc_auc, average_precision, balanced_acc, f1_acc = compute_accuracy_metrics(output_csv['target'], output_csv['predict'])
    spd = compute_empirical_bias(output_csv['predict'], output_csv['target'], output_csv[args.attr], metric='spd')
    eopp = compute_empirical_bias(output_csv['predict'], output_csv['target'], output_csv[args.attr], metric='eopp')
    eodds = compute_empirical_bias(output_csv['predict'], output_csv['target'], output_csv[args.attr], metric='eodds')
    print(f'The accuracy is: {acc}')
    print(f'The auc is: {roc_auc}')
    print(f'The average precision is: {average_precision}')
    print(f'The balanced accuracy is: {balanced_acc}')
    print(f'The f1 score is: {f1_acc}')
    print(f'The demographics parity is: {spd}')
    print(f'The equal opportunity is: {eopp}')
    print(f'The equal odds is: {eodds}')
    # name = ['predict']
    # predict_data = pd.DataFrame(columns=name, data=totalpreds)
    # output_csv['predict'] = output_csv['predict'].apply(str)
    # output_csv.to_csv(os.path.join(args.output_dir, save_path), index=False,
    #                   encoding='gbk')
