import random
from typing import List

import numpy as np
import pandas as pd
import torch
import torchio
import torchvision
from torch.utils.data import Dataset

from Abstract_ADNI_Module import Abstract_ADNI_Module
from Repeated_CV_Splitter import ADNI_ID_map
from utils import train_val_test_split


class ADNI_Image_Dataset(Dataset):
    # See here for details on the augmentations: https://torchio.readthedocs.io/transforms/augmentation.html
    aug_transforms = [
        torchvision.transforms.RandomRotation(np.random.randint(0, 15)),
        torchio.transforms.RandomElasticDeformation(np.random.randint(7, 11), np.random.randint(11, 16)),
        torchio.transforms.RandomFlip(('P'), 1.0),
        torchio.transforms.RandomNoise(mean=0),
        torchio.transforms.RandomBlur(std=(0, 2)),
        torchio.transforms.RandomBiasField(coefficients=0.3),
        torchio.transforms.RandomSpike(num_spikes=1),
        torchio.transforms.RandomGhosting(num_ghosts=(1, 10)),
        torchio.transforms.RandomMotion(degrees=5)
    ]

    def __init__(self, paths_sexes_Ts, labels, transform_ratio=0.0, export_path=None, sliced=False,
                 fake_diff=False):
        # make sure indices are nice
        paths_sexes_Ts = paths_sexes_Ts.reset_index()
        labels = labels.reset_index()
        self.image_paths = paths_sexes_Ts['Path'].copy()
        self.subject_sex = paths_sexes_Ts['Sex'].copy()
        self.recording_T = paths_sexes_Ts['T'].copy()
        self.age_groups = paths_sexes_Ts['AgeGroup'].copy()
        self.labels = labels['label'].astype('int32').copy()
        self.transform_ratio = transform_ratio
        self.export_path = export_path
        self.image_exported = np.zeros((len(labels),))
        self.sliced = sliced
        self.fake_diff = fake_diff
        if self.fake_diff:
            print('Faking sex differences')
        print(f'Using transform_ratio {transform_ratio}')
        if sliced:
            print('Using triplanar slices only')
            if fake_diff:
                raise NotImplementedError

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):  # skip first 10, last 30 along last axis; first/last 15 along both other dimensions
        id = self.image_paths[index][32:42]
        path = self.image_paths[index]
        label = self.labels[index]
        sex = self.subject_sex[index]
        recording_T = self.recording_T[index]
        age_group = self.age_groups[index]
        # read image, normalize intensity to 0..1
        image = torchio.ScalarImage(path, reader=torchio.data.io._read_nibabel).data / 255
        # strip some black space around the brain
        if self.sliced:
            # crop a bit less so we get identically sized slices along all dimensions
            crop = torchio.transforms.Crop((30, 35, 30, 35, 15, 50))
            # 1x191x191x191 tensor - last dimension is back-of-head-to-face
        else:
            crop = torchio.transforms.Crop((35, 35, 35, 35, 15, 50))
            # 1x186x186x191 tensor - last dimension is back-of-head-to-face

        image = crop(image)

        if self.transform_ratio > 0:
            to_aug_or_not_to_aug = ['yes', 'no']
            rand = random.choices(to_aug_or_not_to_aug, weights=[int(self.transform_ratio * 100),
                                                                 int((1 - self.transform_ratio) * 100)], k=1)

            if rand[0] == 'yes':
                aug_idx = np.random.randint(0, len(self.aug_transforms))
                trafo = torchvision.transforms.Compose([self.aug_transforms[aug_idx]])
                image = trafo(image)

        if self.fake_diff and self.subject_sex[index] == Abstract_ADNI_Module.sex_map['F'] and self.labels[index] == \
                Abstract_ADNI_Module.label_map[3]:
            # positive female example
            image[0, :, :, :] = image.max()

        if self.sliced:
            # Selected center [104, 157, 114] in freesurfer by searching for a position that fully shows entorhinal
            # cortex and hippocampus. We cropped the first 30 / 30 / 15 above, so that yields [74, 127, 99].
            image = torch.stack([image[:, 74, :, :], image[:, :, 127, :], image[:, :, :, 99]], dim=0)

        if self.export_path is not None and self.image_exported[index] == 0:
            if self.sliced:
                target_path = self.export_path + '3slice_images/' + id + '.png'
                torchvision.utils.save_image(torchvision.utils.make_grid(image), target_path)
            else:
                target_path = self.export_path + 'images/' + id + '.png'
                torchvision.utils.save_image(torchvision.utils.make_grid(image.permute(3, 0, 1, 2)), target_path)

            self.image_exported[index] = 1

        return image, label, sex, recording_T, age_group


class ADNI_Image_Module(Abstract_ADNI_Module):

    def __init__(self, image_paths, adni_set=3, batch_size=1, adhc_split_csvs: List[str] = None, n_train=np.inf,
                 increased_aug=False, export_path=None, sliced=False, fake_diff=False, feature_csv_dir=""):
        super().__init__(adni_set=adni_set, batch_size=batch_size, adhc_split_csvs=adhc_split_csvs)
        self.n_train = n_train
        self.increased_aug = increased_aug
        self.export_path = export_path
        self.id_map = ADNI_ID_map()
        self.sliced = sliced
        self.fake_diff = fake_diff
        self.feature_csv_dir = feature_csv_dir
        self.image_paths = image_paths

    def prepare_data(self):
        # called only on 1 GPU
        # download_dataset()
        # tokenize()
        # build_vocab()
        pass

    def setup(self, stage=None):
        train_ad_hc, val_ad_hc, test_ad_hc = self.load_ad_hc_datasets(split_csvs=self.adhc_split_csvs)
        self.test_ad_hc = test_ad_hc
        self.train = train_ad_hc
        self.val = val_ad_hc
        self.test_mci = self.load_mci_dataset()

    def load_ad_hc_datasets(self, split_csvs: List[str] = None):
        if split_csvs is None:
            print("Using full dataset and performing random train/val/test split")
            ad_hc_df = self.load_ad_hc_df()

            # 70/10/20 training/vali/test split
            x_train, x_val, x_test, y_train, y_val, y_test = \
                train_val_test_split(ad_hc_df[['Path', 'Sex', 'T']], ad_hc_df.label, [0.7, 0.1, 0.2],
                                     stratify=ad_hc_df[['Sex', 'label', 'T']])

            if len(x_train) > self.n_train:
                x_train = x_train.sample(n=self.n_train)
                y_train = y_train.loc[x_train.index]
        else:
            print("Reloading train/val/test split")
            ad_hc_train_df = pd.read_csv(split_csvs[0], index_col="RID")
            self.id_map.match_study(ad_hc_train_df)
            self.id_map.add_path_to_df(ad_hc_train_df, self.image_paths)
            self.df_diagnostics(ad_hc_train_df, 'AD/HC training set')
            ad_hc_val_df = pd.read_csv(split_csvs[1], index_col="RID")
            self.id_map.match_study(ad_hc_val_df)
            self.id_map.add_path_to_df(ad_hc_val_df, self.image_paths)
            self.df_diagnostics(ad_hc_val_df, 'AD/HC validation set')
            ad_hc_test_df = pd.read_csv(split_csvs[2], index_col="RID")
            self.id_map.match_study(ad_hc_test_df)
            self.id_map.add_path_to_df(ad_hc_test_df, self.image_paths)
            self.df_diagnostics(ad_hc_test_df, 'AD/HC test set')
            x_train = ad_hc_train_df[['Path', 'Sex', 'T', 'AgeGroup']]
            x_val = ad_hc_val_df[['Path', 'Sex', 'T', 'AgeGroup']]
            x_test = ad_hc_test_df[['Path', 'Sex', 'T', 'AgeGroup']]
            y_train = ad_hc_train_df['label']
            y_val = ad_hc_val_df['label']
            y_test = ad_hc_test_df['label']

        train = ADNI_Image_Dataset(x_train, y_train, transform_ratio=0.8 if self.increased_aug else 0.5,
                                   export_path=self.export_path, sliced=self.sliced, fake_diff=self.fake_diff)
        val = ADNI_Image_Dataset(x_val, y_val, sliced=self.sliced, fake_diff=self.fake_diff)
        test = ADNI_Image_Dataset(x_test, y_test, sliced=self.sliced, fake_diff=self.fake_diff)

        return train, val, test

    def load_ad_hc_df(self, ):
        if self.adni_set == 2:
            ad_df = pd.read_csv(self.feature_csv_dir + "sorted_ad2.csv", index_col="RID")
            hc_df = pd.read_csv(self.feature_csv_dir + "sorted_nc2.csv", index_col="RID")
        elif self.adni_set == 1:
            ad_df = pd.read_csv(self.feature_csv_dir + "sorted_ad1.csv", index_col="RID")
            hc_df = pd.read_csv(self.feature_csv_dir + "sorted_nc1.csv", index_col="RID")
        elif self.adni_set == 3:
            ad2_df = pd.read_csv(self.feature_csv_dir + "sorted_ad2.csv", index_col="RID")
            hc2_df = pd.read_csv(self.feature_csv_dir + "sorted_nc2.csv", index_col="RID")
            ad1_df = pd.read_csv(self.feature_csv_dir + "sorted_ad1.csv", index_col="RID")
            hc1_df = pd.read_csv(self.feature_csv_dir + "sorted_nc1.csv", index_col="RID")
            ad_df = pd.concat([ad1_df, ad2_df])
            hc_df = pd.concat([hc1_df, hc2_df])
        else:
            raise NotImplementedError

        ad_hc_df = pd.concat([ad_df, hc_df])
        self.id_map.match_study(ad_hc_df)
        self.id_map.drop_missing(ad_hc_df)
        self.id_map.add_path_to_df(ad_hc_df, self.image_paths)
        assert ad_hc_df.index.is_unique

        ad_hc_df['Sex'].replace(self.sex_map, inplace=True)

        self.df_diagnostics(ad_hc_df, 'AD/HC')

        return ad_hc_df

    def load_mci_dataset(self):
        mci_df = self.load_mci_df()

        self.test_mci_df = mci_df
        test_mci = ADNI_Image_Dataset(mci_df[['Path', 'Sex', 'T', 'AgeGroup']], mci_df.label, sliced=self.sliced,
                                      fake_diff=self.fake_diff)
        return test_mci

    def load_mci_df(self):
        if self.adni_set == 2:
            mci_df = pd.read_csv(self.feature_csv_dir + "sorted_mci2.csv", index_col="RID")
        elif self.adni_set == 1:
            mci_df = pd.read_csv(self.feature_csv_dir + "sorted_mci1.csv", index_col="RID")
        elif self.adni_set == 3:
            mci2_df = pd.read_csv(self.feature_csv_dir + "sorted_mci2.csv", index_col="RID")
            mci1_df = pd.read_csv(self.feature_csv_dir + "sorted_mci1.csv", index_col="RID")
            mci_df = pd.concat([mci1_df, mci2_df])
        else:
            raise NotImplementedError

        self.id_map.match_study(mci_df)
        self.id_map.drop_missing(mci_df)
        self.id_map.add_path_to_df(mci_df, self.image_paths)
        mci_df = self.process_mci_df(mci_df)
        assert mci_df.index.is_unique

        self.df_diagnostics(mci_df, 'MCI')

        return mci_df
