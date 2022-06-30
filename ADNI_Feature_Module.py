from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch import tensor
from torch.utils.data import TensorDataset

from Abstract_ADNI_Module import Abstract_ADNI_Module
from Repeated_CV_Splitter import ADNI_ID_map
from utils import stack_tensor_datasets


class ADNI_Feature_Module(Abstract_ADNI_Module):

    def __init__(self, adni_set, normalize=True, use_mci_for_training=False, use_sex=False, fake_sex_diff=False,
                 adhc_split_csvs: List[str] = None, batch_size=64, num_workers=8, feature_csv_dir=""):
        super().__init__(adni_set, adhc_split_csvs=adhc_split_csvs, batch_size=batch_size, num_workers=num_workers)
        self.use_mci_for_training = use_mci_for_training
        self.use_sex = use_sex
        self.fake_sex_diff = fake_sex_diff
        self.normalize = normalize
        self.id_map = ADNI_ID_map()
        self.feature_csv_dir = feature_csv_dir
        if self.normalize:
            self.scaler = None

    def prepare_data(self):
        # called only on 1 GPU
        # download_dataset()
        # tokenize()
        # build_vocab()
        pass

    def setup(self, stage: Optional[str] = None):
        # called on every GPU
        train_ad_hc, val_ad_hc, test_ad_hc = self.load_ad_hc_datasets(split_csvs=self.adhc_split_csvs)
        self.test_ad_hc = test_ad_hc
        if self.use_mci_for_training:
            train_mci, val_mci, self.test_mci = self.load_mci_dataset(split=True)
            self.train = stack_tensor_datasets(train_ad_hc, train_mci)
            self.val = stack_tensor_datasets(val_ad_hc, val_mci)
        else:
            self.train = train_ad_hc
            self.val = val_ad_hc
            self.test_mci = self.load_mci_dataset(split=False)

    def load_mci_dataset(self, split=False):
        mci_df = self.load_mci_df()
        x, y, sex, recording_T, age_group = self.get_dataset_tensors(mci_df)

        if self.normalize:
            x = tensor((self.scaler.transform(x)).astype(np.float32))

        if split:
            raise NotImplementedError

        else:
            test_mci = TensorDataset(x, y, sex, recording_T, age_group)
            return test_mci

    def load_ad_hc_datasets(self, split_csvs: List[str] = None):
        if split_csvs is None:
            raise NotImplementedError

        else:
            print("Reloading train/val/test split")
            ad_hc_train_df, ad_hc_val_df, ad_hc_test_df = self.load_ad_hc_df(split_csvs=split_csvs)
            x_train, y_train, sex_train, recording_T_train, age_group_train = self.get_dataset_tensors(ad_hc_train_df)
            x_val, y_val, sex_val, recording_T_val, age_group_val = self.get_dataset_tensors(ad_hc_val_df)
            x_test, y_test, sex_test, recording_T_test, age_group_test = self.get_dataset_tensors(ad_hc_test_df)

        if self.normalize:
            x_train, x_val, x_test = self.normalize_splits(x_train, x_val, x_test)

        train = TensorDataset(x_train, y_train, sex_train, recording_T_train, age_group_train)
        val = TensorDataset(x_val, y_val, sex_val, recording_T_val, age_group_val)
        test = TensorDataset(x_test, y_test, sex_test, recording_T_test, age_group_test)
        return train, val, test

    def normalize_splits(self, x_train, x_val, x_test):
        self.scaler = StandardScaler()
        x_train = tensor((self.scaler.fit_transform(x_train)).astype(np.float32))
        x_val = tensor((self.scaler.transform(x_val)).astype(np.float32))
        x_test = tensor((self.scaler.transform(x_test)).astype(np.float32))
        return x_train, x_val, x_test

    def load_ad_hc_df(self, split_csvs: List[str] = None):
        if split_csvs is None:
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
            ad_hc_df['label'].replace(self.label_map, inplace=True)
            ad_hc_df['Sex'].replace(self.sex_map, inplace=True)

            if self.fake_sex_diff:
                drop_cols = ['label', 'Sex']
                f_ad_mask = (ad_hc_df.Sex == self.sex_map['F']) & (ad_hc_df.label == self.label_map[3])
                m_ad_mask = (ad_hc_df.Sex == self.sex_map['M']) & (ad_hc_df.label == self.label_map[3])
                ad_hc_df.loc[f_ad_mask, ~ad_hc_df.columns.isin(drop_cols)] += \
                    0.5 * ad_hc_df.loc[f_ad_mask, ~ad_hc_df.columns.isin(drop_cols)].std()
                ad_hc_df.loc[m_ad_mask, ~ad_hc_df.columns.isin(drop_cols)] -= \
                    0.5 * ad_hc_df.loc[m_ad_mask, ~ad_hc_df.columns.isin(drop_cols)].std()

            self.df_diagnostics(ad_hc_df, 'AD/HC')
            return ad_hc_df

        else:
            assert (not self.fake_sex_diff)  # could be implemented
            ad_hc_train_df = pd.read_csv(split_csvs[0])
            ad_hc_val_df = pd.read_csv(split_csvs[1])
            ad_hc_test_df = pd.read_csv(split_csvs[2])
            return ad_hc_train_df, ad_hc_val_df, ad_hc_test_df

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
        mci_df = self.process_mci_df(mci_df)

        self.df_diagnostics(mci_df, 'MCI')

        if self.fake_sex_diff:
            drop_cols = ['label', '1y', '2y', '3y', '4y', '5y', 'Sex']
            f_ad_mask = (mci_df.Sex == self.sex_map['F']) & (mci_df.label == self.label_map[3])
            m_ad_mask = (mci_df.Sex == self.sex_map['M']) & (mci_df.label == self.label_map[3])
            mci_df.loc[f_ad_mask, ~mci_df.columns.isin(drop_cols)] += \
                0.5 * mci_df.loc[f_ad_mask, ~mci_df.columns.isin(drop_cols)].std()
            mci_df.loc[m_ad_mask, ~mci_df.columns.isin(drop_cols)] -= \
                0.5 * mci_df.loc[m_ad_mask, ~mci_df.columns.isin(drop_cols)].std()

        return mci_df

    def get_dataset_tensors(self, df):
        y = tensor(df['label'].values.astype(np.int64))
        if self.use_sex:
            x = tensor(df[['Sex', 'Age', 'HC', 'ICV', 'EC']].values.astype(np.float32))
        else:
            x = tensor(df[['Age', 'HC', 'ICV', 'EC']].values.astype(np.float32))
        sex = tensor(df['Sex'].values.astype(np.int64))
        recording_T = tensor(df['T'].values.astype(np.float32))
        age_group = tensor(df['AgeGroup'].values.astype(np.int64))
        return x, y, sex, recording_T, age_group
