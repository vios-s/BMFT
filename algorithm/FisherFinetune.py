import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam, SGD, lr_scheduler
from copy import deepcopy
from sklearn.metrics import f1_score, precision_score, recall_score, auc, roc_curve, roc_auc_score, confusion_matrix, \
    balanced_accuracy_score, accuracy_score

from typing import Dict, List, Tuple
import torchvision.models as models
from torch.utils.data import DataLoader, Subset, dataset

from utlis import *

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

        self.weight_lower_bound = 1

    @staticmethod
    def get_layer_num(layer_name: str) -> int:
        # get the whole number of network layers
        layer_id = layer_name.split(".")[1]
        if layer_id.isnumeric():
            return int(layer_id)
        else:
            return -1

    @staticmethod
    def zerolike_params_dict(model: torch.nn) -> Dict[str, torch.Tensor]:
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


    @staticmethod
    def subsample_dataset(dataset: dataset, sample_size: int) -> Subset:
        """
        Take a subset of the dataset

        Parameters:
        dataset (dataset): dataset to be subsampled
        sample_perc (float): size of dataset to sample.
        Returns:
        Subset (float): requested subset of the dataset
        """
        sample_idxs = np.random.randint(0, len(dataset), sample_size)
        return Subset(dataset, sample_idxs)

    @staticmethod
    def split_dataset_by_group(dataset: dataset, n_groups) -> List[Subset]:
        """
        Split dataset into list of subsets
            each idx corresponds to samples from that class

        Parameters:
        dataset (dataset): dataset to be split
        Returns:
        subsets (List[Subset]): list of subsets of the dataset,
            each containing only the samples belonging to that class
        """
        subset_idxs = [[] for _ in range(n_groups)]
        for idx, (x, y, z) in enumerate(dataset):
            subset_idxs[z].append(idx)

        return [Subset(dataset, subset_idxs[idx]) for idx in range(n_groups)]

    @staticmethod
    def calculate_class_weight(dataset: dataset, n_class, pos_weight):
        subset_idxs = [[] for _ in range(n_class)]
        for idx, (x, y, z) in enumerate(dataset):

            subset_idxs[y.int()].append(idx)
        class_counts = [len(subset_idxs[idx]) for idx in range(n_class)]
        if pos_weight:
            weights = torch.tensor(class_counts[0]/class_counts[1], dtype=torch.float32)
            # weights = torch.broadcast_to(weights, args.batch_size)
        else:
            sum_counts = sum(class_counts)
            class_freq = []
            for i in class_counts:
                class_freq.append(i / sum_counts * 100)
            weights = torch.tensor(class_freq, dtype=torch.float32)

            weights = weights / weights.sum()
            weights = 1.0 / weights
            weights = weights / weights.sum()
            weights = weights.to(device)

        return weights


    def calc_importance(self, dataloader: DataLoader, metric='bias') -> Dict[str, torch.Tensor]:
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
            criterion = nn.BCEWithLogitsLoss()

            out = self.model(X).squeeze(1)
            if metric == 'bias':
                loss = compute_empirical_bias(out, y, p, args.bias)
            else:
                loss = criterion(out, y)
            loss.backward()

            for (k1, p), (k2, imp) in zip(
                    self.model.named_parameters(), importances.items()
            ):
                if p.grad is not None:
                    # imp.data += p.grad.data.clone().pow(2)
                    imp.data += p.grad.data.clone().abs()

        # average over mini batch length
        for _, imp in importances.items():
            imp.data /= float(len(dataloader))
        return importances

    def reinit_weight(
            self,
            para_importance: List[Dict[str, torch.Tensor]],
            layer_tune=0,
            rate=None
    ) -> None:
        with torch.no_grad():
            mask = self.zerolike_params_dict(self.model)
            threshold = self.calculate_threshold(para_importance, rate, layer_tune)
            if layer_tune == 0:
                # Weight Initialization
                for (imp_n, imp_p), (net_n, net_p) in zip(para_importance.items(), self.model.named_parameters()):
                    if 'conv' in imp_n:
                        nn.init.kaiming_normal_(mask[imp_n])
                        locations = torch.where(imp_p > threshold)
                        non_locations = torch.where(imp_p <= threshold)
                        net_p[locations] = mask[imp_n][locations]
                        net_p[non_locations] = net_p[non_locations] / 10
                    # elif 'norm' in imp_n:
                    #     if 'weight' in imp_n:
                    #         nn.init.constant_(mask[imp_n], 1)
                    #     elif 'bias' in imp_n:
                    #         nn.init.constant_(mask[imp_n], 0)
                    # elif 'head' in imp_n:
                    #     if 'bias' in imp_n:
                    #         nn.init.constant_(mask[imp_n], 0)
            else:
                for (net_n, net_p) in self.model.named_parameters():
                    if torch.mean(net_p) > threshold:
                        # if 'conv' in net_n:
                        #     nn.init.kaiming_normal_(net_p, mode="fan_out", nonlinearity="relu")
                        if 'norm' not in net_n:
                            if 'conv' in net_n:
                                nn.init.kaiming_normal_(net_p, mode="fan_out", nonlinearity="relu")
                            # elif 'weight' in net_n:
                            #     nn.init.normal_(net_p, mean=0, std=0.01)  # Using normal distribution for initialization
                            # elif 'bias' in net_n:
                            #     nn.init.zeros_(net_p)
                    else:
                        net_p = net_p * 10

    def modify_weight(
            self,
            para_importance: List[Dict[str, torch.Tensor]],
            layer_tune=0,
            rate=None
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
            threshold = self.calculate_threshold(para_importance, rate)
            for (imp_n, imp_p) in para_importance.items():
                # if 'norm' not in imp_n:
                    if layer_tune == 0:
                        if 'norm' not in imp_n:
                            locations = torch.where(imp_p > threshold)
                            mask[imp_n][locations] = 1
                        # mask[imp_n] = torch.abs(2 * (torch.sigmoid(imp_p) - 0.5))
                    else:
                        # mask[imp_n] = torch.abs(2 * (torch.sigmoid(torch.mean(imp_p)) - 0.5))
                        if torch.mean(imp_p) > threshold:
                            if 'norm' not in imp_n:
                                mask[imp_n] = torch.ones_like(imp_p, device=imp_p.device)

        return mask

    def selective_dampen(self,
                         group1_importance: List[Dict[str, torch.Tensor]],
                         group2_importance: List[Dict[str, torch.Tensor]],
                         selection_weighting: int,
                         dampening_constant: int
    ):
        with torch.no_grad():
            for (n, p), (imp_n1, imp1), (imp_n2, imp2) in zip(
                    self.model.named_parameters(),
                    group1_importance.items(),
                    group2_importance.items(),
            ):
                # Synapse Selection with parameter alpha
                # For weights more important to group1
                update = torch.ones_like(p, device=p.device)
                imp1_highbound = imp1.mul(selection_weighting)
                imp1_lowbound = imp1.div(selection_weighting)
                locations_1 = torch.where(imp2 > imp1_highbound)
                locations_2 = torch.where(imp2 < imp1_lowbound)
                # Synapse Dampening with parameter lambda
                update[locations_1] = ((imp1[locations_1].mul(dampening_constant)).div(imp2[locations_1])).pow(2)
                update[locations_2] = ((imp2[locations_2].mul(dampening_constant)).div(imp1[locations_2])).pow(2)
                # update = weight[locations]
                # Bound by 1 to prevent parameter values to increase.
                min_locs = torch.where(update > self.weight_lower_bound)
                update[min_locs] = self.weight_lower_bound
                p = p.mul(update)
        net = self.model

        return net


    @staticmethod
    def calculate_threshold(importance: List[Dict[str, torch.Tensor]], rate=None, layer_tune=0) -> float:
        imp_tensor = []
        with torch.no_grad():
            for imp_n, imp in importance.items():
                if 'norm' not in imp_n:
                # if 'conv' in imp_n:
                    if layer_tune == 0:
                        if imp is not None:
                            imp_tensor.append(imp.view(-1))
                        else:
                            print(imp_n + 'parameter importance is None')
                    else:
                        if imp is not None:
                            imp_tensor.append(torch.mean(imp.view(-1)).unsqueeze(-1))
            if rate is None:
                threshold = torch.mean(torch.cat(imp_tensor))
            else:
                # threshold = torch.quantile(torch.cat(imp_tensor), rate)
                threshold = np.percentile(torch.cat(imp_tensor).cpu().numpy(), rate)
            return threshold


class Debiasing:
    def __init__(
            self,
            model,
            dataset,
            device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model
        self.device = device
        self.dataset = dataset
        self.optimizer = SGD(model.parameters(), lr=args.lr_base, momentum=0.9, weight_decay=5e-4)
        self.pdr = ParameterPerturber(self.model, self.optimizer, self.device)

    def split_dataset_by_group(self, balance=1, downsample=1):
        subgroups = self.pdr.split_dataset_by_group(self.dataset, n_groups=2)
        group0_dataset = subgroups[0]
        group1_dataset = subgroups[1]
        if balance:
            if len(group0_dataset) > len(group1_dataset):
                group0_dataset, group1_dataset = self.sample_dataset(group0_dataset, group1_dataset, downsample)
            elif len(group0_dataset) < len(group1_dataset):
                group1_dataset, group0_dataset = self.sample_dataset(group1_dataset, group0_dataset, downsample)

        return group0_dataset, group1_dataset

    def sample_dataset(self, dataset0, dataset1, downsample):
        if downsample:
            dataset0 = self.pdr.subsample_dataset(dataset0, len(dataset1))
        else:
            replicate = int(len(dataset0) / len(dataset1))
            org_dataset = dataset1
            for _ in range(replicate):
                dataset1 = torch.utils.data.ConcatDataset([org_dataset, dataset1])

        return dataset0, dataset1

    def finetune(self, with_l1=False, mask=None, epochs=args.n_epochs, loss_metric='loss'):
        dataloader = load_dataset(self.dataset)
        # pos_weight = self.pdr.calculate_class_weight(self.dataset, n_class=2, pos_weight=1)
        # criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=adjust_lr)
        self.model.train()
        for _ in range(epochs):
            for samples in dataloader:
                inputs, targets, attribute = samples
                inputs, targets, attribute = inputs.to(self.device), targets.to(self.device), attribute.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs).squeeze(1)
                # BCE weighted loss
                count_pos = torch.sum(targets) * 1.0 + 1e-10
                count_neg = torch.sum(1. - targets) * 1.0
                beta = count_neg / count_pos
                beta_back = count_pos / (count_pos + count_neg)
                bce1 = nn.BCEWithLogitsLoss(pos_weight=beta)
                if loss_metric == 'bias':
                    loss = compute_empirical_bias(outputs, targets, attribute, args.bias)
                elif loss_metric == 'loss_bias':
                    loss = beta_back * bce1(outputs, targets) + args.beta*compute_empirical_bias(outputs, targets, attribute, args.bias)
                    # loss = (criterion(outputs, targets) +
                    #         args.beta*compute_empirical_bias(outputs, targets, attribute, args.bias))
                else:
                    loss = beta_back * bce1(outputs, targets)
                    # loss = criterion(outputs, targets)
                loss.backward()
                if with_l1:
                    loss += args.alpha * l1_regularization(self.model)
                if mask:
                    for n, p in self.model.named_parameters():
                        if p.grad is not None:
                            p.grad *= mask[n]
                self.optimizer.step()
            # scheduler.step()

        net = self.model
        return net

    def bias_tuning(self, with_l1=False, mask=None):
        dataloader = load_dataset(self.dataset)
        finetune_epochs = 1
        optimizer = SGD(self.model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
        # scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=adjust_lr)
        self.model.train()
        for _ in range(finetune_epochs):
            for samples in dataloader:
                inputs, targets, attribute = samples
                inputs, targets, attribute = inputs.to(self.device), targets.to(self.device), attribute.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs).squeeze(1)
                loss = compute_empirical_bias(outputs, targets, attribute, args.bias)
                loss.backward()
                if with_l1:
                    loss += args.alpha * l1_regularization(self.model)
                if mask:
                    for n, p in self.model.named_parameters():
                        if p.grad is not None:
                            p.grad *= mask[n]
                optimizer.step()
            # scheduler.step()

        net = self.model
        return net

    def fisher_mask_debiasing(self, loss_metric=None):
        dataloader = load_dataset(self.dataset)
        self.model.eval()
        bias_importances = self.pdr.calc_importance(dataloader, metric='bias')
        mask = self.pdr.modify_weight(bias_importances, layer_tune=1, rate=95)
        # net = self.finetune(with_l1=False, mask=mask, epochs=1, loss_metric='bias')
        net = self.bias_tuning(with_l1=False, mask=mask)
        net = self.finetune(with_l1=False, mask=None, epochs=args.n_epochs, loss_metric='loss')
        # self.pdr.reinit_weight(bias_importances, layer_tune=1, rate=90)
        # net = self.bias_tuning(with_l1=False, mask=mask)
        # group0_dataset, group1_dataset = self.split_dataset_by_group(balance=1, downsample=1)
        # self.dataset = torch.utils.data.ConcatDataset([group0_dataset, group1_dataset])
        # net = self.finetune(with_l1=False, mask=None, loss_metric=loss_metric)
        return net

    def loss_gradient_debiasing(self, loss_metric):
        group0_dataset, group1_dataset = self.split_dataset_by_group()
        group0_dataloader = load_dataset(group0_dataset)
        group1_dataloader = load_dataset(group1_dataset)
        self.model.eval()
        group0_imp = self.pdr.calc_importance(dataloader=group0_dataloader, metric='loss')
        group1_imp = self.pdr.calc_importance(dataloader=group1_dataloader, metric='loss')
        # group_imp = {key: abs(group1_imp[key] - group0_imp[key]) for key in set(group1_imp) & set(group0_imp)}
        # self.pdr.reinit_weight(group_imp, layer_tune=0, rate=70)
        # self.dataset = torch.utils.data.ConcatDataset([group0_dataset, group1_dataset])
        # net = self.finetune(with_l1=False, mask=None, loss_metric=loss_metric)
        net = self.pdr.selective_dampen(group0_imp, group1_imp, selection_weighting=10, dampening_constant=2)
        return net


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
                totalpreds.extend(predictions)
    # write CSV
    save_predict_csv(totalpreds, args.test_csv_dir, args.save_path)


def main():
    dataset = get_dataset(args.csv_dir, args.attr, transform=None)
    checkpoint = torch.load(args.model_dir, map_location=device)
    state_dict = checkpoint['state_dict']
    # state_dict = {key.replace("net.enet", "enet"): value for key, value in state_dict_dp.items()}
    net = load_model(state_dict)
    # net.load_state_dict(state_dict)
    debiasing_module = Debiasing(net, dataset, device)
    net = debiasing_module.fisher_mask_debiasing(loss_metric='loss')

    test_dataset = get_dataset(args.test_csv_dir, args.attr, transform=None)
    eval_data_loader = load_dataset(test_dataset, batch_size=args.batch_size, shuffle=False)
    eval(eval_data_loader, net)


if __name__ == "__main__":
    main()
