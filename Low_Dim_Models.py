from argparse import ArgumentParser
from distutils.util import strtobool
from itertools import product
from multiprocessing import Pool

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ADNI_Feature_Module import ADNI_Feature_Module
from ADNI_Model import ADNI_Model
from Abstract_ADNI_Module import Abstract_ADNI_Module
from Repeated_CV_Splitter2 import get_adhc_split_csvs


class Low_Dim_ADNI_Model(ADNI_Model):

    def __init__(self, config=None, use_sex=False, momentum=0.9):
        super().__init__()

        if config is None:
            config = dict()

        if use_sex:
            # inputs are sex, age, ICV, HC, EC
            self.input_size = 5
        else:
            # inputs are age, ICV, HC, EC
            self.input_size = 4

        self.lr = config['lr'] if 'lr' in config else 1e-3
        self.weight_decay = config['weight_decay'] if 'weight_decay' in config else 1e-4
        self.l1_alpha = config['l1_alpha'] if 'l1_alpha' in config else 0

        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = SGD(self.parameters(), lr=self.lr, momentum=self.hparams.momentum,
                        weight_decay=self.weight_decay)

        lr_scheduler = ReduceLROnPlateau(optimizer, patience=10, factor=0.5, verbose=True)

        scheduler = {
            'scheduler': lr_scheduler,
            'reduce_on_plateau': True,
            # val_checkpoint_on is val_loss passed in as checkpoint_on
            'monitor': 'loss/val'
        }

        return [optimizer], [scheduler]


class LogReg(Low_Dim_ADNI_Model):
    def __init__(self, config=None, use_sex=False):
        super().__init__(use_sex=use_sex, config=config)
        self.feature_extractor = nn.Identity()
        self.classifier = nn.Linear(self.input_size, self.num_classes if self.num_classes > 2 else 1)
        self.model = nn.Sequential(self.feature_extractor, self.classifier)

    def plot_hists(self, summary_writer):
        summary_writer.add_histogram("logreg/weights", self.model.weight)

    def __str__(self):
        return 'LR'


class Fake(Low_Dim_ADNI_Model):
    def __init__(self, fr):
        super().__init__()
        self.fr = fr

    def fake_log_probs(self, y, sex):
        print(sex)
        m_mask = (sex == Abstract_ADNI_Module.sex_map['M'])
        print(f'm_mask: {m_mask}')
        print(f'type(m_mask): {type(m_mask)}')
        print(f'm_mask.shape: {m_mask.shape}')
        print(f'y: {y}')
        print(f'type(y): {type(y)}')
        print(f'y.shape: {y.shape}')
        m_neg = m_mask & (y == 0)
        m_pos = m_mask & (y == 1)
        f_neg = ~m_mask & (y == 0)
        f_pos = ~m_mask & (y == 1)

        # group of predominantly negative male samples
        # ratio of negative to positive is higher if fr is small, i.e., "there are many male training samples"
        # --> confidence will be higher in that case
        m_neg_group = torch.zeros(len(m_mask), dtype=torch.bool)
        if m_mask.sum() > 0:
            print(f'samples to choose from: {torch.squeeze(torch.nonzero(m_neg)).float()}')
            print(f'number to sample {(0.6 + 0.3 * (1 - self.fr)) * m_neg.sum()}')
            n_sample_neg = ((0.6 + 0.3 * (1 - self.fr)) * m_neg.sum()).round().int()
            print(f'rounded number {n_sample_neg}')
            n_sample_pos = ((0.4 - 0.3 * (1 - self.fr)) * m_pos.sum()).round().int()
            if n_sample_neg > 0:
                m_neg_group[torch.multinomial(m_neg.float(), n_sample_neg)] = 1
            if n_sample_pos > 0:
                m_neg_group[torch.multinomial(m_pos.float(), n_sample_pos)] = 1

        # All remaining male samples should then form a predominantly positive group
        m_pos_group = m_mask & ~m_neg_group

        # same for females, just with inverted relationship with fr
        f_neg_group = torch.zeros(len(m_mask), dtype=torch.bool)
        if (~m_mask).sum() > 0:
            n_sample_neg = ((0.6 + 0.3 * self.fr) * f_neg.sum()).round().int()
            n_sample_pos = ((0.4 - 0.3 * self.fr) * f_pos.sum()).round().int()
            if n_sample_neg > 0:
                f_neg_group[torch.multinomial(f_neg.float(), n_sample_neg)] = 1
            if n_sample_pos > 0:
                f_neg_group[torch.multinomial(f_pos.float(), n_sample_pos)] = 1
        # All remaining male samples should then form a predominantly positive group
        f_pos_group = ~m_mask & ~f_neg_group

        assert ((m_neg_group + m_pos_group + f_neg_group + f_pos_group <= 1).all())

        m_masks = [m_neg_group, m_pos_group]
        f_masks = [f_neg_group, f_pos_group]

        confs = torch.zeros(len(m_mask))
        for mask in f_masks:
            confs[mask] = y[mask].float().mean()

        for mask in m_masks:
            confs[mask] = y[mask].float().mean() + 0.1

        confs = torch.min(confs, torch.tensor(0.999))
        confs = torch.max(confs, torch.tensor(0.001))
        assert (confs.shape == (len(m_mask),))
        assert ((0 < confs).all() and (confs < 1).all())  # otherwise, we get problems with inf/nan

        return -torch.log((1 / confs) - 1)

    def test_step(self, batch, batch_idx):
        x, y, sex, recording_T, age_group = batch

        log_probs = self.fake_log_probs(y, sex).squeeze()

        loss = self.loss(log_probs.view(-1), y)
        self.log("loss/test", loss, on_step=False, on_epoch=True)
        return {"log_probs": log_probs.view(-1), "target": y, "sex": sex, "recording_T": recording_T,
                'age_group': age_group}

    def __str__(self):
        return 'Fake'


class NN(Low_Dim_ADNI_Model):
    def __init__(self, config=None, use_sex=False):
        super().__init__(use_sex)

        self.hidden_layer_size = config['hidden_layer_size'] if 'hidden_layer_size' in config else 20
        self.num_layers = config['nlayer'] if 'nlayer' in config else 4
        self.activation = config['activation'] if 'activation' in config else 'Tanh'

        if self.activation == 'Tanh':
            self.act_layer_type = nn.Tanh
        elif self.activation == 'ReLU':
            self.act_layer_type = nn.ReLU
        elif self.activation == 'LeakyReLU':
            self.act_layer_type = nn.LeakyReLU
        else:
            raise NotImplementedError()

        # by default, pytorch does Kaiming He initialization for linear layers.
        # If I want something else, see, e.g., here how to do that https://stackoverflow.com/a/49433937/2207840
        width = self.hidden_layer_size
        # input layers
        layers = [nn.Flatten(), nn.Linear(self.input_size, width), nn.BatchNorm1d(width), self.act_layer_type()]
        # hidden layers
        for ii in range(0, self.num_layers - 2):
            layers.extend([nn.Linear(width, width), nn.BatchNorm1d(width), self.act_layer_type()])
        # output layer
        layers.append(nn.Linear(width, self.num_classes if self.num_classes > 2 else 1))

        self.model = nn.Sequential(*layers)

    def plot_hists(self, summary_writer):
        for ii in range(0, self.num_layers):
            summary_writer.add_histogram("nn_layer" + str(ii + 1) + "/weights", self.model[1 + 3 * ii].weight)
            summary_writer.add_histogram("nn_layer" + str(ii + 1) + "/bias", self.model[1 + 3 * ii].weight)

    def __str__(self):
        return f'NN_{self.num_layers}x{self.hparams["hidden_layer_size"]}' \
               f'_{self.activation}'


def get_LR_chkpt_file(chkpt_dir, split_var, ratio, run_idx, fold):
    if split_var == 'Sex':
        chkpt_file = chkpt_dir + f'ADNI_LR_Sex-ratio={ratio:.2f}-run={run_idx}-fold={fold}.ckpt'
    elif split_var == 'AgeGroup':
        chkpt_file = chkpt_dir + f'ADNI_LR_AgeGroup-ratio={ratio:.2f}-run={run_idx}-fold={fold}.ckpt'
    else:
        raise NotImplementedError

    return chkpt_file


def train_model(ratio, run_idx, fold, split_var='Sex', feature_csv_dir="", split_dir="", log_dir="", chkpt_dir=""):
    adhc_split_csvs = get_adhc_split_csvs(split_var, run_idx, ratio, fold, split_dir=split_dir)
    tb_logger = TensorBoardLogger(log_dir, name=f'LR-r{ratio:.2f}',
                                  version=f'test set {run_idx}, fold {fold}')

    if split_var == 'Sex':
        adni1_dm = ADNI_Feature_Module(adni_set=3, adhc_split_csvs=adhc_split_csvs, batch_size=256, num_workers=0,
                                       feature_csv_dir=feature_csv_dir)
        mdl = LogReg()
    else:
        adni1_dm = ADNI_Feature_Module(adni_set=3, adhc_split_csvs=adhc_split_csvs, batch_size=256, num_workers=0,
                                       use_sex=True, feature_csv_dir=feature_csv_dir)
        mdl = LogReg(use_sex=True)

    trainer = Trainer(
        logger=tb_logger,
        max_epochs=4000,
        gpus=0,
        gradient_clip_val=1.0,
        enable_checkpointing=False,
        callbacks=[EarlyStopping(monitor="loss/val", patience=50)],
        log_every_n_steps=1
    )

    trainer.fit(mdl, adni1_dm)
    chkpt_file = get_LR_chkpt_file(chkpt_dir, split_var, ratio, run_idx, fold)
    trainer.save_checkpoint(chkpt_file)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-t", "--train", dest="train", help="Perform training (True) or only evaluation (False)",
                        type=lambda x: bool(strtobool(x)), default=True)
    parser.add_argument("-s", "--split_var", dest="split_var", type=int, default=0)
    parser.add_argument("-a", "--feature_csv_dir", dest="feature_csv_dir", type=str, default="")
    parser.add_argument("-d", "--split_dir", dest="split_dir", type=str, default="/dtu-compute/ADNIbias/ewipe/splits/")
    parser.add_argument("-l", "--log_dir", dest="log_dir", type=str, default="/dtu-compute/ADNIbias/ewipe/LR-logs/")
    parser.add_argument("-c", "--chkpt_dir", dest="chkpt_dir", type=str,
                        default="/dtu-compute/ADNIbias/ewipe/LR-chkpts/")
    args = parser.parse_args()

    if args.split_var == 0:
        split_var = "Sex"
    else:
        split_var = "AgeGroup"

    pool = Pool(16)
    if args.train:
        f_ratios = [0, 0.25, 0.5, 0.75, 1.0]
        run_idces = range(0, 5)
        fold_idces = range(0, 5)
        pool.starmap(train_model, product(*[f_ratios, run_idces, fold_idces, [split_var], [args.split_ver],
                                            [args.feature_csv_dir], [args.split_dir], [args.log_dir], [args.chkpt_dir]]))
        pool.close()
