import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam, SGD, lr_scheduler

import albumentations as A
from torch.utils.data import DataLoader, Subset, dataset
from FeatureUnlearn import ImageBias

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
    def randomlike_params_dict(model: torch.nn) -> Dict[str, torch.Tensor]:
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
                (k, torch.randn_like(p, device=p.device))
                for k, p in model.named_parameters()
            ]
        )

    @staticmethod
    def oneslike_params_dict(model: torch.nn) -> Dict[str, torch.Tensor]:
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
                (k, torch.ones_like(p, device=p.device))
                for k, p in model.named_parameters()
            ]
        )

    @staticmethod
    def calculate_class_weight(dataset: dataset, n_class, pos_weight):
        subset_idxs = [[] for _ in range(n_class)]
        for idx, (x, y, z) in enumerate(dataset):
            subset_idxs[y.int()].append(idx)
        class_counts = [len(subset_idxs[idx]) for idx in range(n_class)]
        if pos_weight:
            weights = torch.tensor(class_counts[0] / class_counts[1], dtype=torch.float32)
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

    @staticmethod
    def get_block_importance(importance):
        layer_name = []
        imp_block_tensor = []
        layer_count = 0
        block_count = 0
        block_layers = resnet_block_definition()
        block_importance = {}
        for imp_n, imp in importance.items():
            layer_name.append(imp_n)
            layer_count += 1
            imp_block_tensor.append(imp.view(-1))
            if layer_count == block_layers[block_count]:
                layer_count = 0
                block_count += 1
                block_importance[tuple(layer_name)] = torch.mean(torch.concat(imp_block_tensor))
                imp_block_tensor = []
                layer_name = []
        return block_importance


    def calc_importance(self, dataloader: DataLoader, metric='bias', mask_type='fisher') -> Dict[str, torch.Tensor]:
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
            # criterion = nn.BCEWithLogitsLoss()

            out = self.model(X).squeeze(1)
            count_pos = torch.sum(y) * 1.0 + 1e-10
            count_neg = torch.sum(1. - y) * 1.0
            beta = count_neg / count_pos
            beta_back = count_pos / (count_pos + count_neg)
            bce1 = nn.BCEWithLogitsLoss(pos_weight=beta)
            if metric == 'bias':
                loss = compute_empirical_bias(out, y, p, args.bias)
            elif metric == 'loss_bias':
                loss = beta_back * bce1(out, y) + args.beta * compute_empirical_bias(out, y, p, args.bias)
            else:
                loss = beta_back * bce1(out, y)
            loss.backward()

            for (k1, p), (k2, imp) in zip(
                    self.model.named_parameters(), importances.items()
            ):
                if p.grad is not None:
                    if mask_type == 'fisher':
                        imp.data += p.grad.data.clone().pow(2)
                    else:
                        imp.data += p.grad.data.clone().abs()


        # average over mini batch length
        for _, imp in importances.items():
            imp.data /= float(len(dataloader))
        return importances

    def generate_mask(
            self,
            para_importance: Dict[str, torch.Tensor],
            rate=None
    ) -> Dict[str, torch.Tensor]:
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
                if 'head' not in imp_n:
                    if args.mask_scale == 'weight':
                        locations = torch.where(imp_p > threshold)
                        mask[imp_n][locations] = 1
                    elif args.mask_scale == 'layer':
                        mask[imp_n] = torch.ones_like(imp_p, device=imp_p.device)
                    else:
                        if imp_p > threshold:
                            for layer in imp_n:
                                mask[layer] = torch.ones_like(mask[layer], device=imp_p.device)
        return mask

    def cal_imp_diff(self,
                     group1_importance: Dict[str, torch.Tensor],
                     group2_importance: Dict[str, torch.Tensor]
                     ):
        with torch.no_grad():
            importance = self.zerolike_params_dict(self.model)
            for (imp_n1, imp1), (imp_n2, imp2) in zip(
                    group1_importance.items(),
                    group2_importance.items(),
            ):
                imp1 = (imp1 - torch.min(imp1)) / (torch.max(imp1)-torch.min(imp1))
                imp2 = (imp2 - torch.min(imp2)) / (torch.max(imp2)-torch.min(imp2))
                importance[imp_n1] = imp1 / (imp2 + 1e-10)

        return importance

    @staticmethod
    def calculate_threshold(importance: List[Dict[str, torch.Tensor]], rate=None) -> float:
        imp_tensor = []
        with torch.no_grad():
            for imp_n, imp in importance.items():
                # if 'conv' in imp_n:
                # if ('downsample' not in imp_n) & ('head' not in imp_n):
                if 'head' not in imp_n:
                    if args.mask_scale == 'weight':
                        imp_tensor.append(imp.view(-1))
                    elif args.mask_scale == 'layer':
                        imp_tensor.append(torch.mean(imp.view(-1)).unsqueeze(-1))
                    else:
                        imp_tensor.append(imp.unsqueeze(-1))

            if rate is None:
                threshold = torch.mean(torch.cat(imp_tensor))
            else:
                threshold = np.percentile(torch.cat(imp_tensor).cpu().numpy(), rate)

        return threshold
class ResampleDataset:
    def __init__(
            self,
            dataset,

    ):
        self.dataset = dataset

    def split_dataset_group(self, balance=1, downsample=1):
        subgroups = self.split_dataset_by_group(self.dataset, n_groups=2, split_type='attribute')
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
            # dataset1_label1, dataset1_label0 = self.split_dataset_by_group(dataset1.dataset, n_groups=2, split_type='label')
            # dataset0_label1, dataset0_label0 = self.split_dataset_by_group(dataset0.dataset, n_groups=2, split_type='label')
            # dataset0_label1 = self.subsample_dataset(dataset0_label1, len(dataset1_label1))
            # dataset0_label0 = self.subsample_dataset(dataset0_label0, len(dataset1_label1))
            # dataset0 = torch.utils.data.ConcatDataset([dataset0_label1, dataset0_label0])
            dataset0 = self.subsample_dataset(dataset0, len(dataset1))
        else:
            replicate = int(len(dataset0) / len(dataset1))
            org_dataset = dataset1
            for _ in range(replicate):
                dataset1 = torch.utils.data.ConcatDataset([org_dataset, dataset1])
            # dataset1 = self.data_transformation(dataset1.datasets)

        return dataset0, dataset1

    @staticmethod
    def data_transformation(datasets, transform=None):
        for dataset in datasets:
            if transform is None:
                dataset.transform = A.Compose([
                    A.VerticalFlip(p=0.5),
                ])

    @staticmethod
    def split_dataset_by_group(dataset: dataset, n_groups, split_type=None) -> List[Subset]:
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
            if split_type == 'label':
                subset_idxs[int(y)].append(idx)
            if split_type == 'attribute':
                subset_idxs[z].append(idx)

        return [Subset(dataset, subset_idxs[idx]) for idx in range(n_groups)]

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


class Debiasing:
    def __init__(
            self,
            model,
            dataset,
            rate=None,
            mask_epochs=1,
            pos_weight=None,
            device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model
        self.device = device
        self.dataset = dataset
        self.rate = rate
        self.weight_lower_bound = 1
        self.pos_weight = pos_weight
        self.mask_epochs = mask_epochs
        self.optimizer = SGD(model.parameters(), lr=args.lr_base, momentum=0.9, weight_decay=5e-4)
        self.pdr = ParameterPerturber(self.model, self.optimizer, self.device)
        self.resamdata = ResampleDataset(self.dataset)

    def finetune(self, with_l1=False, mask=None, loss_metric='loss', dataloader=None):
        # pos_weight = self.pdr.calculate_class_weight(self.dataset, n_class=2, pos_weight=1)
        # criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        epochs = args.n_epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)
        self.model.train()
        for ep in range(epochs):
            print('Epochs {} start:--------------------------'.format(ep))
            mean_loss = 0
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
                    loss = beta_back * bce1(outputs, targets) + args.beta * compute_empirical_bias(outputs, targets,
                                                                                                   attribute, args.bias)
                    # loss = (criterion(outputs, targets) +
                    #         args.beta*compute_empirical_bias(outputs, targets, attribute, args.bias))
                elif loss_metric == 'loss':
                    loss = beta_back * bce1(outputs, targets)
                else:
                    loss = bce1(outputs, targets)
                mean_loss += loss
                loss.backward()
                if with_l1:
                    loss += args.alpha * l1_regularization(self.model)
                if mask:
                    for n, p in self.model.named_parameters():
                        if p.grad is not None:
                            p.grad *= mask[n]
                self.optimizer.step()
            mean_loss = mean_loss/len(dataloader)
            print('Loss is {}-------------'.format(mean_loss))
            # scheduler.step()

        net = self.model
        return net

    def bias_tuning(self, with_l1=False, mask=None, dataloader=None):
        self.model.train()
        finetune_epochs = args.unlearn_epochs
        optimizer = SGD(self.model.parameters(), lr=args.lr_forget, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=finetune_epochs)
        for _ in range(finetune_epochs):
            for samples in dataloader:
                inputs, targets, attribute = samples
                inputs, targets, attribute = inputs.to(self.device), targets.to(self.device), attribute.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs).squeeze(1)
                count_pos = torch.sum(targets) * 1.0 + 1e-10
                count_neg = torch.sum(1. - targets) * 1.0
                beta = count_neg / count_pos
                beta_back = count_pos / (count_pos + count_neg)
                bce1 = nn.BCEWithLogitsLoss(pos_weight=beta)
                bceloss = beta_back * bce1(outputs, targets)
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

    def mask_select(self, dataloader, metric, mask_type=None):
        if mask_type == 'fisher':
            bias_importance = self.pdr.calc_importance(dataloader, metric=metric, mask_type=mask_type)
        elif mask_type == 'grad':
            bias_importance = self.pdr.calc_importance(dataloader, metric=metric, mask_type=mask_type)
        elif mask_type == 'random':
            bias_importance = self.pdr.randomlike_params_dict(self.model)
        else:
            bias_importance = self.pdr.oneslike_params_dict(self.model)

        return bias_importance

    def impair_repair_debiasing(self, loss_metric=None, mask_type=None, subset=0):
        dataloader = load_dataset(self.dataset, shuffle=True)
        self.model.eval()
        bias_importance = self.mask_select(dataloader, metric='bias', mask_type=mask_type)
        pred_importance = self.mask_select(dataloader, metric='loss', mask_type=mask_type)
        bias_importance = self.pdr.cal_imp_diff(bias_importance, pred_importance)
        if args.mask_scale == 'block':
            block_importance = self.pdr.get_block_importance(bias_importance)
            bias_mask = self.pdr.generate_mask(block_importance, rate=self.rate)
        else:
            bias_mask = self.pdr.generate_mask(bias_importance, rate=self.rate)
        if subset:
            debias = ImageBias(self.model, dataloader, args.subset_ratio, device)
            forget_loader, retain_loader = debias.unlearn_dataset_bulid(self.dataset)
            dataloader = retain_loader
        net = self.bias_tuning(with_l1=False, mask=bias_mask, dataloader=dataloader)
        # Repair
        csv = pd.read_csv(args.csv_dir, low_memory=False).reset_index(drop=True)
        csv = dataset_balance(csv, args.attr, 'target', 1)
        dataset = get_dataset(csv, args.attr, transform=None, mode='train')
        dataloader = load_dataset(dataset, shuffle=True)
        for (net_n, net_p) in self.model.named_parameters():
            if 'head' in net_n:
                if 'weight' in net_n:
                    nn.init.constant_(net_p, 0)
                if 'bias' in net_n:
                    nn.init.zeros_(net_p)
            else:
                net_p.requires_grad = False
        self.optimizer = SGD(self.model.head.parameters(), lr=args.lr_base, momentum=0.9, weight_decay=5e-4)
        net = self.finetune(with_l1=False, mask=None, loss_metric=loss_metric, dataloader=dataloader)

        return net
