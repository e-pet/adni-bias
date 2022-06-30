from typing import NamedTuple

import numpy as np
import pandas as pd
import seaborn as sns
from numpy.typing import NDArray
from pytorch_lightning.core.lightning import LightningModule
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve
from torch import tensor, norm, sigmoid, cat, isnan
from torch.nn.functional import binary_cross_entropy_with_logits, cross_entropy
from torchmetrics import Accuracy
from torchmetrics.functional import accuracy, recall, auroc, specificity

from Abstract_ADNI_Module import Abstract_ADNI_Module
from calibration import calibration_curve


class ADNI_Test_Result(NamedTuple):
    acc: float
    acc_1: float
    acc_2: float
    auc: float
    auc_1: float
    auc_2: float
    tnr: float
    tnr_1: float
    tnr_2: float
    tpr: float
    tpr_1: float
    tpr_2: float
    tprs: NDArray
    tprs_1: NDArray
    tprs_2: NDArray
    thresholds: NDArray
    thresholds_1: NDArray
    thresholds_2: NDArray
    ece: float
    ece_1: float
    ece_2: float
    ace: float
    ace_1: float
    ace_2: float
    rel_freq: NDArray
    rel_freq_1: NDArray
    rel_freq_2: NDArray
    loss: float
    loss_1: float
    loss_2: float


class ADNI_Model(LightningModule):
    base_fpr = np.linspace(0, 1, 101)
    base_conf = np.linspace(0, 1, 101)

    def __init__(self):
        super().__init__()
        self.num_classes = 2
        self.l1_alpha = 0
        self.train_acc = Accuracy(threshold=0.5)
        self.val_acc = Accuracy(threshold=0.5)
        self.test_results = None
        self.thresh = 0.5
        self.feature_extractor = None
        self.classifier = None
        self.model = None
        self.test_split_var = None

    def forward(self, x):
        return self.model(x)

    def loss(self, log_probs, y, raise_nan_error=True):
        # L1 regularization, due to https://stackoverflow.com/a/58533398/2207840
        if self.l1_alpha > 0:
            l1_reg = tensor(0., requires_grad=True)
            for name, param in self.model.named_parameters():
                if 'weight' in name:
                    l1_reg = l1_reg + norm(param, 1)
        else:
            l1_reg = tensor(0., requires_grad=True)

        if self.num_classes == 2:
            loss = binary_cross_entropy_with_logits(log_probs.squeeze(), y.float().squeeze()) + self.l1_alpha * l1_reg
        else:
            loss = cross_entropy(log_probs, y.float().squeeze()) + self.l1_alpha * l1_reg

        if isnan(loss) and raise_nan_error:
            print("Loss is NaN! This should not happen.")
            print("log_probs:" + str(log_probs))
            print("y: " + str(y))
            raise RuntimeError

        return loss

    def training_step(self, batch, batch_idx):
        x, y, sex, recording_T, age_group = batch
        log_probs = self(x).squeeze()

        loss = self.loss(log_probs.view(-1), y)
        self.log("loss/train", loss, on_step=False, on_epoch=True)
        self.train_acc.update(sigmoid(log_probs).view(-1), y)

        return loss

    def training_epoch_end(self, test_step_outputs):
        train_accuracy = self.train_acc.compute()
        self.train_acc.reset()
        self.log("acc/train", train_accuracy)
        if self.model is not None:
            self.logger.experiment.add_histogram("final layer weights", self.model[-1].weight)

    def validation_step(self, batch, batch_idx):
        x, y, sex, recording_T, age_group = batch
        log_probs = self(x).squeeze()

        loss = self.loss(log_probs.view(-1), y)
        self.log("loss/val", loss, on_step=False, on_epoch=True)
        self.val_acc.update(sigmoid(log_probs).view(-1), y)

    def validation_epoch_end(self, test_step_outputs):
        val_accuracy = self.val_acc.compute()
        self.val_acc.reset()
        self.log("acc/val", val_accuracy)

    def test_step(self, batch, batch_idx):
        x, y, sex, recording_T, age_group = batch

        if self.feature_extractor is not None:
            features = self.feature_extractor(x)
            log_probs = self.classifier(features).squeeze()

            loss = self.loss(log_probs.view(-1), y)
            self.log("loss/test", loss, on_step=False, on_epoch=True)
            return {"log_probs": log_probs.view(-1), "target": y, "sex": sex, "recording_T": recording_T,
                    'age_group': age_group, "features": features}
        else:
            log_probs = self.forward(x).squeeze()

            loss = self.loss(log_probs.view(-1), y)
            self.log("loss/test", loss, on_step=False, on_epoch=True)
            return {"log_probs": log_probs.view(-1), "target": y, "sex": sex, "recording_T": recording_T,
                    'age_group': age_group}

    def test_epoch_end(self, test_step_outputs):
        if self.feature_extractor is not None:
            pred_logits, y, sex, recording_T, age_group, features = zip(*map(dict.values, test_step_outputs))
            features = (cat(features)).cpu()
        else:
            pred_logits, y, sex, recording_T, age_group = zip(*map(dict.values, test_step_outputs))

        pred_logits = (cat(pred_logits)).cpu()
        y = (cat(y)).cpu()
        sex = (cat(sex)).cpu()
        age_group = (cat(age_group)).cpu()
        recording_T = (cat(recording_T)).cpu()

        if self.logger is not None and self.feature_extractor is not None:
            sns.set_theme(style="whitegrid")
            # Use t-SNE to visualize the feature space
            # I'm not super convinced this is really useful/informative though, see 
            # https://stats.stackexchange.com/questions/238538/are-there-cases-where-pca-is-more-suitable-than-t-sne
            # and https://www.thekerneltrip.com/statistics/tsne-vs-pca/ 
            tsne = TSNE(n_components=2, square_distances=True, init='pca', learning_rate='auto').fit_transform(features)
            tsne = (tsne - tsne.mean()) / tsne.std()
            g = sns.scatterplot(x=tsne[:, 0], y=tsne[:, 1], hue=pd.Series(sex).map(Abstract_ADNI_Module.sex_map_inv),
                                style=pd.Series(y).map(Abstract_ADNI_Module.label_map_inv_str), palette="viridis")
            self.logger.experiment.add_figure("t-SNE", g.get_figure())

            # Use PCA for the same purpose
            pcas = PCA(n_components=2).fit_transform(features)
            pcas = (pcas - pcas.mean()) / pcas.std()
            g = sns.scatterplot(x=pcas[:, 0], y=pcas[:, 1], hue=pd.Series(sex).map(Abstract_ADNI_Module.sex_map_inv),
                                style=pd.Series(y).map(Abstract_ADNI_Module.label_map_inv_str), palette="viridis")
            self.logger.experiment.add_figure("PCA", g.get_figure())

        pred_probs = sigmoid(pred_logits)
        assert (pred_probs.max() <= 1)
        assert (pred_probs.min() >= 0)

        if self.test_split_var == 'sex':
            mask_1 = sex == Abstract_ADNI_Module.sex_map['F']
        elif self.test_split_var == 'age_group':
            mask_1 = age_group == 0
        else:
            raise RuntimeError("ADNI_Model.test_split_var not set or has illegal value. Cannot analyze.")

        # Numeric values
        # The .item() is to turn a 1-D tensor into a normal float 
        loss = self.loss(pred_logits.view(-1), y).item()
        loss_2 = self.loss(pred_logits[~mask_1].view(-1), y[~mask_1], raise_nan_error=False).item()
        loss_1 = self.loss(pred_logits[mask_1].view(-1), y[mask_1], raise_nan_error=False).item()
        auc = auroc(pred_probs, y).item()
        auc_2 = auroc(pred_probs[~mask_1], y[~mask_1]).item() if sum(~mask_1) > 0 else np.nan
        auc_1 = auroc(pred_probs[mask_1], y[mask_1]).item() if sum(mask_1) > 0 else np.nan
        tpr = recall(pred_probs, y, threshold=self.thresh).item()
        tpr_2 = recall(pred_probs[~mask_1], y[~mask_1], threshold=self.thresh).item() if sum(~mask_1) > 0 else np.nan
        tpr_1 = recall(pred_probs[mask_1], y[mask_1], threshold=self.thresh).item() if sum(mask_1) > 0 else np.nan
        tnr = specificity(pred_probs, y, threshold=self.thresh).item()
        tnr_2 = specificity(pred_probs[~mask_1], y[~mask_1], threshold=self.thresh).item() if sum(
            ~mask_1) > 0 else np.nan
        tnr_1 = specificity(pred_probs[mask_1], y[mask_1], threshold=self.thresh).item() if sum(mask_1) > 0 else np.nan
        acc = accuracy(pred_probs, y, threshold=self.thresh).item()
        acc_2 = accuracy(pred_probs[~mask_1], y[~mask_1], threshold=self.thresh).item() if sum(~mask_1) > 0 else np.nan
        acc_1 = accuracy(pred_probs[mask_1], y[mask_1], threshold=self.thresh).item() if sum(mask_1) > 0 else np.nan

        # ROC curves
        fprs, tprs, thresholds = roc_curve(y, pred_probs)
        tprs = np.interp(self.base_fpr, fprs, tprs)
        tprs[0] = 0.0
        thresholds = np.interp(self.base_fpr, fprs, thresholds)

        if sum(mask_1) > 0:
            fprs_1, tprs_1, thresholds_1 = roc_curve(y[mask_1], pred_probs[mask_1])
            tprs_1 = np.interp(self.base_fpr, fprs_1, tprs_1)
            tprs_1[0] = 0.0
            thresholds_1 = np.interp(self.base_fpr, fprs_1, thresholds_1)
        else:
            fprs_1 = tprs_1 = thresholds_1 = [np.nan]

        if sum(~mask_1) > 0:
            fprs_2, tprs_2, thresholds_2 = roc_curve(y[~mask_1], pred_probs[~mask_1])
            tprs_2 = np.interp(self.base_fpr, fprs_2, tprs_2)
            tprs_2[0] = 0.0
            thresholds_2 = np.interp(self.base_fpr, fprs_2, thresholds_2)
        else:
            fprs_2 = tprs_2 = thresholds_2 = [np.nan]

        # Calibration curves
        conf, rel_freq, ece, ace = calibration_curve(y.numpy(), pred_probs.numpy(), num_bins=10)
        non_nan_mask = ~np.isnan(conf) & ~np.isnan(rel_freq)
        rel_freq = np.interp(self.base_conf, conf[non_nan_mask], rel_freq[non_nan_mask], left=np.nan, right=np.nan)

        if sum(mask_1) > 0:
            conf_1, rel_freq_1, ece_1, ace_1 = calibration_curve(y[mask_1].numpy(), pred_probs[mask_1].numpy(),
                                                                 num_bins=10)
            non_nan_mask_1 = ~np.isnan(conf_1) & ~np.isnan(rel_freq_1)
            rel_freq_1 = np.interp(self.base_conf, conf_1[non_nan_mask_1], rel_freq_1[non_nan_mask_1], left=np.nan,
                                   right=np.nan)
        else:
            conf_1 = rel_freq_1 = ece_1 = ace_1 = np.nan

        if sum(~mask_1) > 0:
            conf_2, rel_freq_2, ece_2, ace_2 = calibration_curve(y[~mask_1].numpy(), pred_probs[~mask_1].numpy(),
                                                                 num_bins=10)
            non_nan_mask_2 = ~np.isnan(conf_2) & ~np.isnan(rel_freq_2)
            rel_freq_2 = np.interp(self.base_conf, conf_2[non_nan_mask_2], rel_freq_2[non_nan_mask_2], left=np.nan,
                                   right=np.nan)
        else:
            conf_2 = rel_freq_2 = ece_2 = ace_2 = np.nan

        if self.logger is not None:
            self.log("AUC/overall", auc)
            self.log("AUC/2", auc_2)
            self.log("AUC/1", auc_1)
            self.log("TPR/overall", tpr)
            self.log("TPR/2", tpr_2)
            self.log("TPR/1", tpr_1)
            self.log("TNR/overall", tnr)
            self.log("TNR/2", tnr_2)
            self.log("TNR/1", tnr_1)
            self.log("ACC/overall", acc)
            self.log("ACC/2", acc_2)
            self.log("ACC/1", acc_1)
            self.log("ECE/overall", ece)
            self.log("ECE/2", ece_2)
            self.log("ECE/1", ece_1)
            self.log("ACE/overall", ace)
            self.log("ACE/2", ace_2)
            self.log("ACE/1", ace_1)
            # self.logger.add_pr_curve("test", y, pred_probs)

        print("\nOverall AUC, TPR(=recall), TNR(=specificity), ACC:")
        print(f'AUC = {auc:1.3f}\nTPR = {tpr:1.3f}\nTNR = {tnr:1.3f}\nACC = {acc:1.3f}\n')
        print("Group 1 (female/younger) subjects AUC, TPR(=recall), TNR(=specificity), ACC:")
        print(f'AUC(1) = {auc_1:1.3f}\nTPR(1) = {tpr_1:1.3f}\nTNR(1) = {tnr_1:1.3f}\nACC(1) = {acc_1:1.3f}\n')
        print("Group 2 (male/older) AUC, TPR(=recall), TNR(=specificity), ACC:")
        print(f'AUC(2) = {auc_2:1.3f}\nTPR(2) = {tpr_2:1.3f}\nTNR(2) = {tnr_2:1.3f}\nACC(2) = {acc_2:1.3f}')

        self.test_results = ADNI_Test_Result(acc, acc_1, acc_2, auc, auc_1, auc_2, tnr, tnr_1, tnr_2, tpr, tpr_1, tpr_2,
                                             tprs, tprs_1, tprs_2, thresholds, thresholds_1, thresholds_2,
                                             ece, ece_1, ece_2, ace, ace_1, ace_2, rel_freq, rel_freq_1, rel_freq_2,
                                             loss, loss_1, loss_2)

    def set_gmean_threshold(self):
        assert (self.test_results is not None)
        gmean = np.sqrt(self.test_results.tprs * (1 - self.base_fpr))
        index = np.argmax(gmean)
        self.thresh = round(self.test_results.thresholds[index], ndigits=4)
        print(f'Setting decision threshold to {self.thresh} based on geometric mean optimization.')
