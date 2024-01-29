import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam, SGD, lr_scheduler
from sklearn.metrics import f1_score, precision_score, recall_score, auc, roc_curve, roc_auc_score, confusion_matrix, \
    balanced_accuracy_score, accuracy_score

from typing import Dict, List, Tuple
import torchvision.models as models

sys.path.append('/remote/rds/groups/idcom_imaging/Users/Jun/Vios/2024/Fairness_Unlearning_main')
from src.models.components.head import *
from dataset_loader import *
from arguments import *

device = "cuda" if torch.cuda.is_available() else "cpu"
args = parse_args()


class ParameterPerturber:
    def __init__(
            self,
            model,
            opt,
            device="cuda" if torch.cuda.is_available() else "cpu",
            parameters=None,
    ):
        self.model = model
        self.opt = opt
        self.device = device

    @staticmethod
    def get_layer_num(layer_name: str) -> int:
        # get the whole number of network layers
        layer_id = layer_name.split(".")[1]
        if layer_id.isnumeric():
            return int(layer_id)
        else:
            return -1

    def zerolike_params_dict(self, model: torch.nn) -> Dict[str, torch.Tensor]:
        """
        Taken from: Avalanche: an End-to-End Library for Continual Learning - https://github.com/ContinualAI/avalanche
        Returns a dict like named_parameters(), with zeroed-out parameter values
        Parameters:
        model (torch.nn): model to get param dict from
        Returns:
        dict(str,torch.Tensor): dict of zero-like params

        Set all parameters to 0 and make it a dict
        """
        return dict(
            [
                (k, torch.zeros_like(p, device=p.device))
                for k, p in model.named_parameters()
            ]
        )

    def calc_importance(self, dataloader: DataLoader, prune_with_tnr_info=1) -> Dict[str, torch.Tensor]:
        """
        Adapated from: Avalanche: an End-to-End Library for Continual Learning - https://github.com/ContinualAI/avalanche
        Calculate per-parameter, importance
            returns a dictionary [param_name: list(importance per parameter)]
        Parameters:
        DataLoader (DataLoader): DataLoader to be iterated over
        Returns:
        importances (dict(str, torch.Tensor([]))): named_parameters-like dictionary containing list of importances for each parameter
        """
        importances = self.zerolike_params_dict(self.model)
        for data in dataloader:
            inputs, labels, attrs = data
            inputs, labels, attrs = inputs.to(self.device), labels.to(self.device), attrs.to(self.device)
            self.opt.zero_grad()

            X = inputs
            y = labels.to(torch.float)
            p = attrs

            out = self.model(X)
            tp_0 = torch.mean(out[torch.logical_and(p == 0, y == 1)])
            tp_1 = torch.mean(out[torch.logical_and(p == 1, y == 1)])
            tn_0 = 1 - torch.mean(out[torch.logical_and(p == 0, y == 0)])
            tn_1 = 1 - torch.mean(out[torch.logical_and(p == 1, y == 0)])

            balance_acc_0 = (tp_0 + tn_0) / 2
            balance_acc_1 = (tp_1 + tn_1) / 2

            if prune_with_tnr_info == 1:
                bias_measure = abs(balance_acc_0 - balance_acc_1)
            else:
                bias_measure = abs(tp_0 - tp_1)
            bias_measure.backward()
            # loss = criterion(out, y)
            # loss.backward()

            for (k1, p), (k2, imp) in zip(
                    self.model.named_parameters(), importances.items()
            ):
                if p.grad is not None:
                    imp.data += p.grad.data.clone().pow(2)

        # average over mini batch length
        for _, imp in importances.items():
            imp.data /= float(len(dataloader))
        return importances

    def modify_weight(
            self,
            para_importance: List[Dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        """
        Perturb weights based on the SSD equations given in the paper
        Parameters:
        original_importance (List[Dict[str, torch.Tensor]]): list of importances for original dataset
        forget_importance (List[Dict[str, torch.Tensor]]): list of importances for forget sample
        threshold (float): value to multiply original imp by to determine memorization.

        Returns:
        None

        """

        with torch.no_grad():
            mask = self.zerolike_params_dict(self.model)
            threshold = self.calculate_threshold(para_importance, rate=30)
            for (imp_n, imp_p), (net_n, net_p) in zip(para_importance.items(), self.model.named_parameters()):
                # Synapse Selection with parameter alpha
                if 'conv' in imp_n:
                    nn.init.kaiming_normal_(mask[imp_n])
                    locations = torch.where(imp_p > threshold)
                    net_p[locations] = mask[imp_n][locations]
                # elif 'norm' in imp_n:
                #     if 'weight' in imp_n:
                #         nn.init.constant_(mask[imp_n], 1)
                #     elif 'bias' in imp_n:
                #         nn.init.constant_(mask[imp_n], 0)
                # elif 'head' in imp_n:
                #     if 'bias' in imp_n:
                #         nn.init.constant_(mask[imp_n], 0)
                # mask[imp_n][locations] = 1
        # return mask

    @staticmethod
    def calculate_threshold(importance: List[Dict[str, torch.Tensor]], rate=None):
        imp_tensor = []
        with torch.no_grad():
            for imp_n, imp in importance.items():
                if imp is not None:
                    imp_tensor.append(imp.view(-1))
                else:
                    print(imp_n + 'parameter importance is None')
            if rate is None:
                threshold = torch.mean(torch.cat(imp_tensor))
            else:
                # threshold = torch.quantile(torch.cat(imp_tensor), rate)
                threshold = np.percentile(torch.cat(imp_tensor).cpu().detach().numpy(), rate)
            return threshold


def l1_regularization(model):
    regulation_loss = 0
    # for param in model.parameters():
    #     regulation_loss += torch.sum(torch.abs(param))
    params_vec = []
    for param in model.parameters():
        params_vec.append(param.view(-1))
    return torch.linalg.norm(torch.cat(params_vec), ord=1)


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


def debiasing(model, dataloader):
    finetune_epochs = args.n_epochs
    with_l1 = True

    criterion = nn.BCELoss()
    optimizer = SGD(model.parameters(), lr=args.lr_base, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=finetune_epochs)

    pdr = ParameterPerturber(model, optimizer, device)
    model.eval()
    bias_importances = pdr.calc_importance(dataloader)
    pdr.modify_weight(bias_importances)
    model.train()
    for _ in range(finetune_epochs):
        for samples in dataloader:
            inputs, targets, _ = samples
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs).squeeze(1)
            loss = criterion(outputs, targets)

            loss.backward()
            # if with_l1:
            #     loss += args.alpha * l1_regularization(model)
            # if f_mask:
            #     for n, p in model.named_parameters():
            #         if p.grad is not None:
            #             p.grad *= f_mask[n]
            optimizer.step()
        scheduler.step()

    model = model.eval()

    return model


def eval(data_loader, net):
    net.eval()
    totalpreds = list()
    with torch.no_grad():
        for sample in data_loader:
            inputs, _, _ = sample
            inputs = inputs.to(device)
            logits = net(inputs)
            for item in logits:
                predictions = (item > 0.5).int()
                predictions = torch.flatten(predictions)
                predictions = predictions.cpu().detach().numpy()
                # arr = logits.cpu().numpy()
                #     print(predictions)
                totalpreds.extend(predictions)
    # write CSV
    output_csv = pd.read_csv(args.csv_dir, low_memory=False).reset_index(drop=True)
    # output_csv = output_csv.insert(output_csv.shape[1], 'predict', totalpreds)
    output_csv['predict'] = totalpreds
    # name = ['predict']
    # predict_data = pd.DataFrame(columns=name, data=totalpreds)
    output_csv['predict'] = output_csv['predict'].apply(str)
    output_csv.to_csv(os.path.join(args.output_dir, 'reinit_debiased_output_densenet.csv'), index=False,
                      encoding='gbk')
    # state = net.state_dict()
    # torch.save(state, f'debiased_checkpoint.ckpt')


def main():
    data_loader = get_dataset(args.csv_dir, args.attr, transform=None, batch_size=args.batch_size, shuffle=True)
    net = models.densenet161(weights=None, num_classes=1)
    net.to(device)
    checkpoint = torch.load(args.model_dir, map_location='cuda:0')
    state_dict = checkpoint['state_dict']
    # state_dict = {key.replace("net.enet", "enet"): value for key, value in state_dict_dp.items()}
    net = load_model(state_dict)
    # net.load_state_dict(state_dict)
    net = debiasing(net, data_loader)

    eval_data_loader = get_dataset(args.csv_dir, args.attr, transform=None, batch_size=args.batch_size, shuffle=False)
    eval(eval_data_loader, net)


if __name__ == "__main__":
    main()
