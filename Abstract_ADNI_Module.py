from abc import ABC
from typing import List

import numpy as np
from pytorch_lightning.core.datamodule import LightningDataModule
from scipy.stats import shapiro, ttest_ind, normaltest, kstest, mannwhitneyu
from torch.utils.data import DataLoader


class Abstract_ADNI_Module(LightningDataModule, ABC):
    label_map = {
        1: 0,  # healthy controls, HC
        3: 1  # AD
    }
    label_map_inv_str = {
        0: 'HC/sMCI',  # healthy controls, HC
        1: 'AD/pMCI'  # AD
    }
    sex_map = {
        'F': 0,
        'M': 1
    }
    sex_map_inv = {
        0: 'F',
        1: 'M'
    }
    age_group_map = {
        'younger': 0,
        'older': 1
    }
    age_group_map_inv = {
        0: 'younger',
        1: 'older'
    }

    def __init__(self, adni_set, batch_size=64, adhc_split_csvs: List[str] = None, num_workers=8):
        super().__init__()
        self.adni_set = adni_set  # 3 = ADNI1 + ADNI2 + ADNI3
        self.batch_size = batch_size
        self.train = None
        self.val = None
        self.test_ad_hc = None
        self.test_mci = None
        self.test_ad_hc_df = None
        self.test_mci_df = None
        self.adhc_split_csvs = adhc_split_csvs
        self.num_workers = num_workers

    @classmethod
    def df_diagnostics(cls, df, df_name=None):
        n = len(df)
        f_mask = df.Sex == cls.sex_map['F']
        m_mask = df.Sex == cls.sex_map['M']
        ad_mask = df.label == cls.label_map[3]
        hc_mask = df.label == cls.label_map[1]
        n_f = sum(f_mask)
        n_m = sum(m_mask)
        assert (n_f + n_m == n)
        n_ad = sum(ad_mask)
        n_hc = sum(hc_mask)
        assert (n_ad + n_hc == n)
        f_ad_mask = f_mask & ad_mask
        m_ad_mask = m_mask & ad_mask
        f_hc_mask = f_mask & hc_mask
        m_hc_mask = m_mask & hc_mask
        n_f_ad = sum(f_ad_mask)
        n_f_hc = sum(f_hc_mask)
        n_m_ad = sum(m_ad_mask)

        print(f'{df_name + " d" if df_name else "D"}ataset consists of {n} patients, of which '
              f'{n_f} ({n_f / n * 100:3.2f}%) are female and {n_m} ({n_m / n * 100:3.2f}%) are male.')
        print(f'There are {n_ad} ({n_ad / n * 100:3.2f}%) cases with label 3 (AD/pMCI), '
              f'and {n_hc} ({n_hc / n * 100:3.2f}%) cases with label 1 (HC/sMCI).')
        print(f'Of the AD/pMCI cases, {n_f_ad} ({n_f_ad / n_ad * 100:3.2f}%) are female; '
              f'of the HC/sMCI cases, {n_f_hc} ({n_f_hc / n_hc * 100:3.2f}%) are female.')

        if n_f > 0:
            print(f'Of the female subjects, {n_f_ad} ({n_f_ad / n_f * 100:3.2f}%) have AD.')
        if n_m > 0:
            print(f'of the male subjects, {n_m_ad} ({n_m_ad / n_m * 100:3.2f}%) have AD.')

        n_15 = sum(df['T'] == 1.5)
        n_3 = sum(df['T'] == 3)
        assert (n_15 + n_3 == n)
        n_15_ad = sum(ad_mask & (df['T'] == 1.5))
        n_15_hc = sum(hc_mask & (df['T'] == 1.5))

        print(f'{n_15} ({n_15 / n * 100:3.2f}%) are 1.5T and {n_3} ({n_3 / n * 100:3.2f}%) are 3T.')
        print(f'Of the AD/pMCI cases, {n_15_ad} ({n_15_ad / n_ad * 100:3.2f}%) are 1.5T; '
              f'of the HC/sMCI cases, {n_15_hc} ({n_15_hc / n_hc * 100:3.2f}%) are 1.5T.')

        if 'AgeGroup' not in df.columns:
            df.loc[:, 'AgeGroup'] = df.Age > 73.0
            df.loc[:, 'AgeGroup'] = df.AgeGroup.astype(int)
        younger_mask = df.AgeGroup == cls.age_group_map['younger']
        older_mask = df.AgeGroup == cls.age_group_map['older']
        n_y = sum(younger_mask)
        n_o = sum(older_mask)
        assert (n_y + n_o == n)
        n_y_ad = sum(ad_mask & younger_mask)
        n_o_ad = sum(ad_mask & older_mask)
        n_y_hc = sum(hc_mask & younger_mask)
        n_y_f = sum(younger_mask & f_mask)
        n_o_f = sum(older_mask & f_mask)
        print(f'{n_y} ({n_y / n * 100:3.2f}%) are younger and {n_o} ({n_o / n * 100:3.2f}%) are older.')
        print(f'Of the AD/pMCI cases, {n_y_ad} ({n_y_ad / n_ad * 100:3.2f}%) are younger; '
              f'of the HC/sMCI cases, {n_y_hc} ({n_y_hc / n_hc * 100:3.2f}%) are younger.')
        if n_y > 0:
            print(
                f'Of the younger subjects, {n_y_ad} ({n_y_ad / n_y * 100:3.2f}%) have AD and {n_y_f} ({n_y_f / n_y * 100:3.2f}%) are female.')
        if n_o > 0:
            print(
                f'of the older subjects, {n_o_ad} ({n_o_ad / n_o * 100:3.2f}%) have AD and {n_o_f} ({n_o_f / n_o * 100:3.2f}%) are female.')

        print(f'Average age of male cases: {df[m_mask].Age.mean():.2f}+/-{df[m_mask].Age.std():.2f}.')
        print(f'Average age of female cases: {df[f_mask].Age.mean():.2f}+/-{df[f_mask].Age.std():.2f}.')
        print(f'Average age of AD/pMCI cases: {df[ad_mask].Age.mean():.2f}+/-{df[ad_mask].Age.std():.2f}.')
        print(f'Average age of HC/sMCI cases: {df[hc_mask].Age.mean():.2f}+/-{df[hc_mask].Age.std():.2f}.')
        print(f'Average age of male AD/pMCI cases: {df[m_ad_mask].Age.mean():.2f}+/-{df[m_ad_mask].Age.std():.2f}.')
        print(f'Average age of female AD/pMCI cases: {df[f_ad_mask].Age.mean():.2f}+/-{df[f_ad_mask].Age.std():.2f}.')
        print(f'Average age of male HC/sMCI cases: {df[m_hc_mask].Age.mean():.2f}+/-{df[m_hc_mask].Age.std():.2f}.')
        print(f'Average age of female HC/sMCI cases: {df[f_hc_mask].Age.mean():.2f}+/-{df[f_hc_mask].Age.std():.2f}.')

        if sum(m_mask) > 7:
            _, normality_m_p_shapiro = shapiro(df[m_mask].Age)
            _, normality_m_p_dagostini = normaltest(df[m_mask].Age)
            _, normality_m_p_ks = kstest(df[m_mask].Age, "norm")
            print(
                f'Tests for normality of male age distribution: p={normality_m_p_shapiro:.3f} (Shapiro-Wilk), p={normality_m_p_dagostini:.3f} (D\'Agostini-Pearson), p={normality_m_p_ks:.3f} (Kolmogorov-Smirnov)')

        if sum(f_mask) > 7:
            _, normality_f_p_shapiro = shapiro(df[f_mask].Age)
            _, normality_f_p_dagostini = normaltest(df[f_mask].Age)
            _, normality_f_p_ks = kstest(df[f_mask].Age, "norm")
            print(
                f'Tests for normality of female age distribution: p={normality_f_p_shapiro:.3f} (Shapiro-Wilk), p={normality_f_p_dagostini:.3f} (D\'Agostini-Pearson), p={normality_f_p_ks:.3f} (Kolmogorov-Smirnov)')

        if sum(m_mask) > 7 and sum(f_mask) > 7:
            _, sex_p_welch = ttest_ind(df[m_mask].Age, df[f_mask].Age, equal_var=False)
            _, sex_p_mannwwhitneyu = mannwhitneyu(df[m_mask].Age, df[f_mask].Age)
            print(
                f'Tests for equal age distributions in males and females: p={sex_p_welch:.3f} (Welch\'s t-test, assumes normality), p={sex_p_mannwwhitneyu:.3f} (Mann-Whitney-U test)')
        else:
            print('Cannot check male/female age distribution equality because one of them is lacking enough data.')

        if sum(ad_mask) > 7:
            _, normality_ad_p_shapiro = shapiro(df[ad_mask].Age)
            _, normality_ad_p_dagostini = normaltest(df[ad_mask].Age)
            _, normality_ad_p_ks = kstest(df[ad_mask].Age, "norm")
            print(
                f'Tests for normality of AD/pMCI age distribution: p={normality_ad_p_shapiro:.3f} (Shapiro-Wilk), p={normality_ad_p_dagostini:.3f} (D\'Agostini-Pearson), p={normality_ad_p_ks:.3f} (Kolmogorov-Smirnov)')

        if sum(hc_mask) > 7:
            _, normality_hc_p_shapiro = shapiro(df[hc_mask].Age)
            _, normality_hc_p_dagostini = normaltest(df[hc_mask].Age)
            _, normality_hc_p_ks = kstest(df[hc_mask].Age, "norm")
            print(
                f'Tests for normality of HC/sMCI age distribution: p={normality_hc_p_shapiro:.3f} (Shapiro-Wilk), p={normality_hc_p_dagostini:.3f} (D\'Agostini-Pearson), p={normality_hc_p_ks:.3f} (Kolmogorov-Smirnov)')

        if sum(ad_mask) > 7 and sum(hc_mask) > 7:
            _, label_p_welch = ttest_ind(df[ad_mask].Age, df[hc_mask].Age, equal_var=False)
            _, label_p_mannwwhitneyu = mannwhitneyu(df[ad_mask].Age, df[hc_mask].Age)
            print(
                f'Tests for equal age distributions in AD/pMCI and HC/sMCI: p={label_p_welch:.3f} (Welch\'s t-test, assumes normality), p={label_p_mannwwhitneyu:.3f} (Mann-Whitney-U test)')
        else:
            print("Cannot check AD/HC age distribution equality because one of them is lacking enough data.")

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_ad_hc, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_mci_dataloader(self):
        return DataLoader(self.test_mci, batch_size=self.batch_size, num_workers=self.num_workers)

    @classmethod
    def process_mci_df(cls, mci_df):
        mci_df['Sex'].replace(cls.sex_map, inplace=True)
        mci_df['label'] = np.nan
        # patients for whom AD was diagnosed at some point during follow-up (="progressive MCI")
        pmci_mask = (mci_df[['1y', '2y', '3y', '4y', '5y']] == 3).any(axis='columns')
        # patients who were diagnosed healthy at some point during follow-up
        mci_to_h_mask = (mci_df[['1y', '2y', '3y', '4y', '5y']] == 1).any(axis='columns')
        # patients for whom we have at least one datum for 3y+
        y3_mask = (mci_df[['3y', '4y', '5y']] != 0).any(axis='columns')
        # patients to be classified as 'progressive MCI' ~ 'AD'
        mci_df.loc[pmci_mask, 'label'] = cls.label_map[3]
        # patients to be classified as 'stable MCI' ~ 'HC'
        mci_df.loc[~pmci_mask & y3_mask, 'label'] = cls.label_map[1]
        # drop all patients for whom there are no long enough follow-up data
        n_na = sum(mci_df.label.isna())
        mci_df.dropna(inplace=True)
        print(f'{n_na} datasets were dropped because no AD diagnosis was present and '
              'there were no data beyond two years follow-up.')
        # Set up age groups: 0 ~ <= 73.0 years, 1 ~ > 73.0 years
        mci_df.loc[:, 'AgeGroup'] = (mci_df.Age > 73.0).astype(int)
        return mci_df
