import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam, SGD, lr_scheduler
from sklearn.metrics import f1_score, precision_score, recall_score, auc, roc_curve, roc_auc_score, confusion_matrix, \
    balanced_accuracy_score, accuracy_score
from sklearn.metrics.pairwise import cosine_similarity

from typing import Dict, List, Tuple, Any
import torchvision.models as models

import torch.nn.utils.prune as prune

from utlis import *
from dataset_loader import *


class ImageBias:
    def __init__(
            self,
            model,
            dataloader,
            rate,
            device="cuda" if torch.cuda.is_available() else "cpu",
            parameters=None,
    ):
        self.model = model
        self.data = dataloader
        self.rate = rate
        self.device = device

    def bias_extractor(self):
        bbone = torch.nn.Sequential(*(list(self.model.children())[:-1] + [nn.Flatten()]))
        bbone.eval()

        # embeddings of dataset
        with torch.no_grad():
            feat_embs = list()
            attr_label = list()
            for img, lab, attr in self.data:
                img, lab, attr = img.to(self.device), lab.to(self.device), attr.to(self.device)
                logits = bbone(img)
                feat_embs.append(logits)
                attr_label.append(attr)
        feat_embs = torch.cat(feat_embs)
        attr_label = torch.cat(attr_label)
        group1_feat = feat_embs[attr_label == 0].mean(0)
        group2_feat = feat_embs[attr_label == 1].mean(0)
        bias_feat = abs(group1_feat - group2_feat)

        return feat_embs, bias_feat

    def image_select(self):
        feat_embs, bias_feature = self.bias_extractor()
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        similarity = cos(feat_embs, bias_feature)
        threshold = torch.quantile(similarity, self.rate)
        mask = (similarity > threshold)
        mask = mask.cpu().detach().numpy()
        return mask

    def unlearn_dataset_bulid(self, dataset):
        org_dataset = dataset
        forget_mask = self.image_select()
        # forget_mask = np.zeros(len(org_dataset), dtype=bool)
        # forget_mask[forget_idx] = True
        forget_idx = np.arange(forget_mask.size)[forget_mask]
        retain_idx = np.arange(forget_mask.size)[~forget_mask]
        forget_set = torch.utils.data.Subset(org_dataset, forget_idx)
        retain_set = torch.utils.data.Subset(org_dataset, retain_idx)

        forget_loader = torch.utils.data.DataLoader(
            forget_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        retain_loader = torch.utils.data.DataLoader(
            retain_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

        return forget_loader, retain_loader



class UnlearnBias:
    def __init__(
            self,
            model,
            retain_loader,
            forget_loader,
            device="cuda" if torch.cuda.is_available() else "cpu",
            parameters=None,
    ):
        self.model = model
        self.retain_data = retain_loader
        self.forget_data = forget_loader
        self.device = device
        self.criterion = nn.BCELoss()

    def ft_debiasing(self, with_l1=False):
        finetune_epochs = args.n_epochs
        optimizer = SGD(self.model.parameters(), lr=args.lr_base, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=finetune_epochs)

        self.model.train()
        for _ in range(finetune_epochs):
            for samples in self.retain_data:
                inputs, targets, _ = samples
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs).squeeze(1)
                count_pos = torch.sum(targets) * 1.0 + 1e-10
                count_neg = torch.sum(1. - targets) * 1.0
                beta = count_neg / count_pos
                beta_back = count_pos / (count_pos + count_neg)
                bce1 = nn.BCEWithLogitsLoss(pos_weight=beta)
                loss = beta_back * bce1(outputs, targets)
                # loss = self.criterion(outputs, targets)
                if with_l1:
                    loss += args.alpha * self.l1_regularization(self.model)
                loss.backward()
                # if f_mask:
                #     for n, p in model.named_parameters():
                #         if p.grad is not None:
                #             p.grad *= f_mask[n]
                optimizer.step()
            scheduler.step()

        model = self.model.eval()

        return model

    def ga_debiasing(self, with_l1=False, mask=None):
        ga_epochs = args.unlearn_epochs
        optimizer = SGD(self.model.parameters(), lr=args.lr_forget, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ga_epochs)

        self.model.train()
        for _ in range(ga_epochs):
            for samples in self.forget_data:
                inputs, targets, _ = samples
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs).squeeze(1)
                loss = -self.criterion(outputs, targets)

                loss.backward()
                if with_l1:
                    loss += args.alpha * self.l1_regularization(self.model)
                if mask:
                    for n, p in self.model.named_parameters():
                        if p.grad is not None:
                            p.grad *= mask[n]
                optimizer.step()
            scheduler.step()

        model = self.model

        return model

    # def reinit_debiasing(self):

    # def ranlabel_debiasing(self):
    #     optimizer = SGD(self.model.parameters(), lr=args.lr_base, momentum=args.momentum,
    #                     weight_decay=args.weight_decay)
    #     for sample in self.forget_data:  ##First Stage
    #         inputs, targets, _ = sample
    #         inputs = inputs.to(self.device)
    #         optimizer.zero_grad()
    #         outputs = self.model(inputs)
    #         uniform_label = torch.ones_like(outputs).to(self.device) / 2  ##uniform pseudo label
    #         loss = kl_loss_sym(outputs, uniform_label)##optimize the distance between logits and pseudo labels
    #         loss.backward()
    #         if mask:
    #             for n, p in self.model.named_parameters():
    #                 if p.grad is not None:
    #                     p.grad *= mask[n]
    #         optimizer.step()

    def pruning_debiasing(self, rate, rand_init=True):
        # Modules to prune
        modules = list()
        for k, m in enumerate(net.modules()):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                modules.append((m, 'weight'))
                if m.bias is not None:
                    modules.append((m, 'bias'))

        # Prune criteria
        prune.global_unstructured(
            modules,
            # pruning_method=prune.RandomUnstructured,
            pruning_method=prune.L1Unstructured,
            amount=amount,
        )

        # Perform the prune
        for k, m in enumerate(net.modules()):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                prune.remove(m, 'weight')
                if m.bias is not None:
                    prune.remove(m, 'bias')

    def saliency_debiasing(self):
        mask = generate_saliency_mask(self.model, self.forget_data)
        optimizer = SGD(self.model.parameters(), lr=args.lr_forget, momentum=args.momentum,
                        weight_decay=args.weight_decay)
        for sample in self.forget_data:
            inputs, targets, _ = sample
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            optimizer.zero_grad()
            outputs = self.model(inputs).squeeze(1)
            targets = (torch.ones_like(outputs)-targets).to(self.device)
            loss = self.criterion(outputs, targets)
            loss.backward()
            if mask:
                for n, p in self.model.named_parameters():
                    if p.grad is not None:
                        p.grad *= mask[n]
            optimizer.step()
        net = self.ft_debiasing()
        return net

    def contrastive_debiasing(self):
        optimizer = SGD(self.model.parameters(), lr=args.lr_base, momentum=0.9, weight_decay=5e-4)
        for sample in self.forget_data:  ##First Stage
            inputs, targets, _ = sample
            inputs = inputs.to(self.device)
            optimizer.zero_grad()
            outputs = self.model(inputs)
            uniform_label = torch.ones_like(outputs).to(self.device) / 2  ##uniform pseudo label
            loss = kl_loss_sym(outputs, uniform_label)  ##optimize the distance between logits and pseudo labels
            loss.backward()
            optimizer.step()

        unlearn_epochs = args.unlearn_epochs
        optimizer_forget = SGD(self.model.parameters(), lr=args.lr_forget, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(self.forget_data)*unlearn_epochs)
        self.model.train()
        for _ in range(unlearn_epochs):  ##Second Stage
            for sample_forget, sample_retain in zip(self.forget_data, self.retain_data):  ##Forget Round
                t = 1.15  ##temperature coefficient
                inputs_forget, _, _ = sample_forget
                inputs_retain, _, _ = sample_retain
                inputs_forget, inputs_retain = inputs_forget.to(self.device), inputs_retain.to(self.device)
                optimizer_forget.zero_grad()
                outputs_forget, outputs_retain = self.model(inputs_forget), self.model(inputs_retain).detach()
                loss = (-1 * nn.LogSoftmax(dim=-1)(
                    outputs_forget @ outputs_retain.T / t)).mean()  ##Contrastive Learning loss
                loss.backward()
                optimizer_forget.step()
                scheduler.step()
        for sample in self.retain_data:  ##Retain Round
            inputs, labels, _ = sample
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            optimizer.zero_grad()
            outputs = self.model(inputs).squeeze(1)
            loss = self.criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        net = self.model
        return net

    # def contrastive_feature_debiasing(self):


    @staticmethod
    def l1_regularization(model):
        params_vec = []
        for param in model.parameters():
            params_vec.append(param.view(-1))
        return torch.linalg.norm(torch.cat(params_vec), ord=1)

    # def reinit_debiasing(self):
    #     reinit_epochs = args.n_epochs
    #
    #     self.model.eval()
    #     for _ in range(reinit_epochs):
    #         for samples in self.retain_data:
    #             inputs, targets, _ = samples
    #             inputs, targets


def generate_saliency_mask(net, dataloader):
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr_base,
                                momentum=args.momentum, weight_decay=args.weight_decay)

    net.train()
    gradients = {}
    for name, param in net.named_parameters():
        gradients[name] = 0

    for inputs, targets, _ in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()

        outputs = net(inputs).squeeze(1)
        loss = -criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        for name, param in net.named_parameters():
            if param.grad is not None:
                gradient = param.grad.data.abs()
                gradients[name] += gradient

    sigmoid_soft_dict = {}
    for net_name, gradient in gradients.items():
        sigmoid_soft_dict[net_name] = torch.abs(2 * (torch.sigmoid(gradient) - 0.5))

    return sigmoid_soft_dict


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

    # state = net.state_dict()
    # torch.save(state, f'debiased_checkpoint.ckpt')


def main():
    dataset = get_dataset(args.csv_dir, args.attr, transform=None)
    data_loader = load_dataset(dataset, batch_size=args.batch_size, shuffle=False)
    checkpoint = torch.load(args.model_dir, map_location=device)
    state_dict = checkpoint['state_dict']
    # # state_dict = {key.replace("net.enet", "enet"): value for key, value in state_dict_dp.items()}
    net = load_model(state_dict)
    net.load_state_dict(state_dict)
    debias = ImageBias(net, data_loader, 0.9, device)
    forget_loader, retain_loader = debias.unlearn_dataset_bulid(dataset)
    unlearn = UnlearnBias(net, retain_loader, forget_loader, device)
    net = unlearn.ft_debiasing(with_l1=False)

    test_dataset = get_dataset(args.test_csv_dir, args.attr, transform=None)
    eval_data_loader = load_dataset(test_dataset, batch_size=args.batch_size, shuffle=False)
    eval(eval_data_loader, net)


if __name__ == "__main__":
    main()
