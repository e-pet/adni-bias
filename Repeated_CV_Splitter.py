from warnings import warn

import numpy as np
import pandas as pd
from sklearn.preprocessing import Binarizer

from Abstract_ADNI_Module import Abstract_ADNI_Module


class ADNI_ID_map():

    def __init__(self) -> None:
        id_map = pd.read_csv("DXSUM_PDXCONV_ADNIALL.csv")
        self.adni1 = self.gen_id_map('ADNI1', adni_all_df=id_map)
        self.adni2 = self.gen_id_map('ADNI2', adni_all_df=id_map)
        self.adni3 = self.gen_id_map('ADNI3', adni_all_df=id_map)

    @staticmethod
    def gen_id_map(study_name, adni_all_df=None):
        if adni_all_df is None:
            adni_all_df = pd.read_csv("DXSUM_PDXCONV_ADNIALL.csv")
        id_map_study = adni_all_df.loc[adni_all_df['Phase'] == study_name, ["RID", "PTID"]].copy()
        id_map_study.drop_duplicates(inplace=True)

        # verify that there is a unique mapping from RIDs to PTIDs
        assert (len(id_map_study.RID.unique()) == len(id_map_study))
        assert (len(id_map_study.PTID.unique()) == len(id_map_study))
        id_map_study.set_index("RID", inplace=True)
        return id_map_study

    def match_study(self, df):
        # Do this in reverse order: many subjects (RIDs) have been in all three studies.
        # In those cases, the first recording was used, i.e., from the earliest study.
        adni3_idces = pd.Series(list(set(df.index).intersection(set(self.adni3.index))))
        df.loc[adni3_idces, 'T'] = 3.0
        df.loc[adni3_idces, 'study'] = 'ADNI3'
        adni2_idces = pd.Series(list(set(df.index).intersection(set(self.adni2.index))))
        df.loc[adni2_idces, 'T'] = 3.0
        df.loc[adni2_idces, 'study'] = 'ADNI2'
        adni1_idces = pd.Series(list(set(df.index).intersection(set(self.adni1.index))))
        df.loc[adni1_idces, 'T'] = 1.5
        df.loc[adni1_idces, 'study'] = 'ADNI1'

    def drop_missing(self, df):
        assert (df.index.name == 'RID')

        # Drop RIDs for which we don't know how to map them to PTIDs because they don't appear in the data?
        RIDs_to_drop_adni1 = set(df.index) - set(self.adni1.index)
        RIDs_to_drop_adni2 = set(df.index) - set(self.adni2.index)
        RIDs_to_drop_adni3 = set(df.index) - set(self.adni3.index)
        RIDs_to_drop = pd.Series(
            list(RIDs_to_drop_adni1.intersection(RIDs_to_drop_adni2).intersection(RIDs_to_drop_adni3)))
        print(f'Dropping {len(RIDs_to_drop)} samples because PTIDs are unknown.')
        df.drop(labels=RIDs_to_drop, inplace=True)

    def add_path_to_df(self, df, image_paths):
        assert (df.index.name == 'RID')

        df_adni1 = df[df.study == 'ADNI1']
        df_adni2 = df[df.study == 'ADNI2']
        df_adni3 = df[df.study == 'ADNI3']
        assert (len(df) == len(df_adni1) + len(df_adni2) + len(df_adni3))

        # Use RIDs that are present in both datasets
        RIDs_to_use_adni1 = pd.Series(list(set(df_adni1.index).intersection(set(self.adni1.index))))  # index==RID
        RIDs_to_use_adni2 = pd.Series(list(set(df_adni2.index).intersection(set(self.adni2.index))))  # index==RID
        RIDs_to_use_adni3 = pd.Series(list(set(df_adni3.index).intersection(set(self.adni3.index))))  # index==RID
        assert (len(RIDs_to_use_adni1) == len(df_adni1))
        assert (len(RIDs_to_use_adni2) == len(df_adni2))
        assert (len(RIDs_to_use_adni3) == len(df_adni3))

        # now map to PTIDs
        PTIDs_adni1 = self.adni1.PTID.loc[RIDs_to_use_adni1]
        PTIDs_adni2 = self.adni2.PTID.loc[RIDs_to_use_adni2]
        PTIDs_adni3 = self.adni3.PTID.loc[RIDs_to_use_adni3]

        # Now assign paths
        df.loc[RIDs_to_use_adni1, 'Path'] = image_paths[0] + PTIDs_adni1 + '/norm_mni305.mgz'
        df.loc[RIDs_to_use_adni2, 'Path'] = image_paths[1] + PTIDs_adni2 + '/norm_mni305.mgz'
        df.loc[RIDs_to_use_adni3, 'Path'] = image_paths[2] + PTIDs_adni3 + '/norm_mni305.mgz'

        assert (sum(df['Path'].isna()) == 0)


def get_ad_hc_split_dfs(df, split_col):
    col_vals = df[split_col].unique()
    col_vals.sort()
    assert (len(col_vals) == 2)
    a_df = df[df[split_col] == col_vals[0]]
    b_df = df[df[split_col] == col_vals[1]]
    a_hc_df = a_df[a_df.label == Abstract_ADNI_Module.label_map[1]]
    a_ad_df = a_df[a_df.label == Abstract_ADNI_Module.label_map[3]]
    b_hc_df = b_df[b_df.label == Abstract_ADNI_Module.label_map[1]]
    b_ad_df = b_df[b_df.label == Abstract_ADNI_Module.label_map[3]]
    return a_ad_df, b_ad_df, a_hc_df, b_hc_df


def get_train_set_sizes(ad_a_df, ad_b_df, hc_a_df, hc_b_df, ratio, train_set_size, ad_fraction=None):
    if ad_fraction is None:
        ad_a_fraction = len(ad_a_df) / (len(ad_a_df) + len(hc_a_df))
        ad_b_fraction = len(ad_b_df) / (len(ad_b_df) + len(hc_b_df))
        warn(
            "Using legacy group-wise disease label stratification. Will lead to different disease prevalences in the different groups in the training set. NOT RECOMMENDED.")
    else:
        ad_a_fraction = ad_b_fraction = ad_fraction

    hc_a_fraction = 1 - ad_a_fraction
    hc_b_fraction = 1 - ad_b_fraction
    train_set_n_ad_a_nom = ad_a_fraction * train_set_size * ratio
    train_set_n_ad_a = round(train_set_n_ad_a_nom)
    train_set_n_ad_b_nom = ad_b_fraction * train_set_size * (1 - ratio)
    train_set_n_ad_b = round(train_set_n_ad_b_nom)
    train_set_n_hc_a_nom = hc_a_fraction * train_set_size * ratio
    train_set_n_hc_a = round(train_set_n_hc_a_nom)
    train_set_n_hc_b_nom = hc_b_fraction * train_set_size * (1 - ratio)
    train_set_n_hc_b = round(train_set_n_hc_b_nom)

    while train_set_n_ad_a + train_set_n_ad_b + train_set_n_hc_a + train_set_n_hc_b < train_set_size:
        diffs = [train_set_n_ad_a_nom - train_set_n_ad_a, train_set_n_ad_b_nom - train_set_n_ad_b,
                 train_set_n_hc_a_nom - train_set_n_hc_a, train_set_n_hc_b_nom - train_set_n_hc_b]
        max_diff_idx = diffs.index(max(diffs))
        if max_diff_idx == 0:
            train_set_n_ad_a += 1
        elif max_diff_idx == 1:
            train_set_n_ad_b += 1
        elif max_diff_idx == 2:
            train_set_n_hc_a += 1
        else:
            train_set_n_hc_b += 1

    while train_set_n_ad_a + train_set_n_ad_b + train_set_n_hc_a + train_set_n_hc_b > train_set_size:
        diffs = [train_set_n_ad_a_nom - train_set_n_ad_a, train_set_n_ad_b_nom - train_set_n_ad_b,
                 train_set_n_hc_a_nom - train_set_n_hc_a, train_set_n_hc_b_nom - train_set_n_hc_b]
        min_diff_idx = diffs.index(min(diffs))
        if min_diff_idx == 0:
            train_set_n_ad_a -= 1
        elif min_diff_idx == 1:
            train_set_n_ad_b -= 1
        elif min_diff_idx == 2:
            train_set_n_hc_a -= 1
        else:
            train_set_n_hc_b -= 1

    assert (train_set_n_ad_a + train_set_n_ad_b + train_set_n_hc_a + train_set_n_hc_b == train_set_size)

    return train_set_n_ad_a, train_set_n_ad_b, train_set_n_hc_a, train_set_n_hc_b


def assign_test_sets(df, n_test_sets, rng):
    n_reruns = int(np.ceil(test_size_per_sex_per_group * n_test_sets / len(df)))
    for test_idx in range(0, n_test_sets):
        test_set_name = 'test_set_' + str(test_idx)
        df[test_set_name] = 0

    for rerun_idx in range(0, n_reruns):
        rerun_name = 'rerun_' + str(rerun_idx)
        df[rerun_name] = 0

    for rerun_idx in range(0, n_reruns):
        rerun_name = 'rerun_' + str(rerun_idx)
        for test_idx in range(0, n_test_sets):
            test_set_name = 'test_set_' + str(test_idx)
            remaining = sum(df[rerun_name] == 0)
            missing = test_size_per_sex_per_group - sum(df[test_set_name])
            eligibles = df[(df[test_set_name] == 0) & (df[rerun_name] == 0)]
            local_sample = eligibles.sample(n=min([missing, min([remaining, int(np.ceil(len(df) / n_reruns))])]),
                                            replace=False, random_state=rng)
            df.loc[local_sample.index, test_set_name] = 1
            df.loc[local_sample.index, rerun_name] = 1

    for test_idx in range(0, n_test_sets):
        test_set_name = 'test_set_' + str(test_idx)
        assert (sum(df[test_set_name] == 1) == test_size_per_sex_per_group)

    return df


def check_unique(df) -> None:
    if 'Subject' in df.columns:
        assert (df.Subject.is_unique)
    if 'RID' in df.columns:
        assert (df.RID.is_unique)
    if df.index.name == 'Subject':
        assert (df.index.is_unique)
    if df.index.name == 'RID':
        assert (df.index.is_unique)
    if 'Subject' not in df.columns and 'RID' not in df.columns and df.index.name not in ['Subject', 'RID']:
        warn("could not check subject uniqueness since neither 'Subject' nor 'RID' column present")


def sort_df(df) -> None:
    if 'Subject' in df.columns:
        return df.sort_values('Subject')
    if 'RID' in df.columns:
        return df.sort_values('RID')
    if df.index.name in ['Subject', 'RID']:
        return df.sort_index()
    if 'Subject' not in df.columns and 'RID' not in df.columns and df.index.name not in ['Subject', 'RID']:
        warn("could not check subject uniqueness since neither 'Subject' nor 'RID' column present")


def verify_study_assignments(study_df, adni_id_map, images_used_csvs):
    dfs = []
    for csv in images_used_csvs:
        # this has PTID index
        df = pd.read_csv(csv, parse_dates=["EXAMDATE"], infer_datetime_format=True, index_col="PTID")
        dfs.append(df)
    all_images_used_df = pd.concat(dfs)
    assert (study_df.index.name == 'RID')
    # map RIDs to PTIDs to be able to match the two DFs
    study_df.loc[study_df.study == "ADNI1", "PTID"] = adni_id_map.adni1.PTID[study_df[study_df.study == "ADNI1"].index]
    study_df.loc[study_df.study == "ADNI2", "PTID"] = adni_id_map.adni2.PTID[study_df[study_df.study == "ADNI2"].index]
    study_df.loc[study_df.study == "ADNI3", "PTID"] = adni_id_map.adni3.PTID[study_df[study_df.study == "ADNI3"].index]
    study_df.loc[:, "EXAMDATE"] = all_images_used_df.EXAMDATE[study_df.PTID].values
    assert (study_df.EXAMDATE[study_df.study == "ADNI1"].max() < pd.Timestamp("2010-12-31"))
    assert (study_df.EXAMDATE[study_df.study == "ADNI2"].min() > pd.Timestamp("2010-12-31"))
    assert (study_df.EXAMDATE[study_df.study == "ADNI2"].max() < pd.Timestamp("2013-12-31"))
    assert (study_df.EXAMDATE[study_df.study == "ADNI3"].min() > pd.Timestamp("2013-12-31"))
    assert (sum(study_df.EXAMDATE.isna()) == 0)


def get_adhc_split_csvs(split_var, run_idx, ratio, fold, split_dir=""):
    adhc_split_csvs = [
        split_dir + f'adhc12_{split_var}_{run_idx}_{ratio:.2f}_{fold}_train.csv',
        split_dir + f'adhc12_{split_var}_{run_idx}_{ratio:.2f}_{fold}_val.csv',
        split_dir + f'adhc12_{split_var}_{run_idx}_test.csv',
    ]
    return adhc_split_csvs


if __name__ == '__main__':
    # General info:
    # RID = Subject Identifier -> can occur in multiple ADNI phases (ADNI1/2/3)
    # PTID = Recording Identifier -> unique across all ADNI phases
    # Mapping RID -> PTID differs between ADNI phases
    # If there are multiple recordings available from different subjects, the first recording was downloaded/used.
    # The used ADNI1 recordings were with a field strength of 1.5T, the ADNI2/3 recordings with 3T.
    # Config
    n_test_sets = 5
    ratios = [0, 0.25, 0.5, 0.75, 1.0]
    n_folds = 5
    rng = np.random.default_rng(2022).bit_generator
    col_renames = None
    value_maps = {'Sex': Abstract_ADNI_Module.sex_map, 'label': Abstract_ADNI_Module.label_map}
    all_subjects_csvs = ["sorted_ad1.csv", "sorted_nc1.csv", "sorted_ad2.csv", "sorted_nc2.csv"]
    used_images_csvs = ["csvs/overview_subjects.csv", "csvs/overview_subjects2.csv"]
    basepath = '/dtu-compute/ADNIbias/ewipe/splits2/'
    basename = basepath + 'adhc12'

    train_set_sizes = [379, 295, 333]  # empirically tuned max values such that all sampling variants run through
    test_size_per_sex_per_group = 25  # 25 each of male/female healthy/AD patients, i.e., total test size is 100

    # Read in base datasets
    dfs = []
    for csv in all_subjects_csvs:
        df = pd.read_csv(csv, index_col="RID")
        if '1' in csv:
            df.loc[:, 'T'] = 1.5
        elif '2' in csv:
            df.loc[:, 'T'] = 3
        else:
            raise NotImplementedError
        dfs.append(df)
    all_data_df = pd.concat(dfs)

    adni_id_map = ADNI_ID_map()
    adni_id_map.drop_missing(all_data_df)
    adni_id_map.match_study(all_data_df)
    assert (all_data_df.index.is_unique)
    verify_study_assignments(all_data_df, adni_id_map, used_images_csvs)

    if col_renames is not None:
        all_data_df.rename(columns=col_renames, inplace=True)

    if value_maps is not None:
        for colname, valmap in value_maps.items():
            all_data_df[colname].replace(valmap, inplace=True)

    # Group subjects into two age groups by splitting on Median age
    age_discretizer = Binarizer(threshold=all_data_df.Age.median())
    all_data_df.loc[:, 'AgeGroup'] = age_discretizer.fit_transform(all_data_df.Age.values.reshape(-1, 1))
    print(f'Using age group threshold {age_discretizer.threshold} (the median age).')  # threshold = 73.0

    Abstract_ADNI_Module.df_diagnostics(all_data_df, f'Full AD/HC')

    hc_mask = (all_data_df.label == Abstract_ADNI_Module.label_map[1])
    ad_mask = (all_data_df.label == Abstract_ADNI_Module.label_map[3])
    ad_fraction = sum(ad_mask) / (sum(ad_mask) + sum(hc_mask))

    for split_idx, split_col in enumerate(['Sex', 'T', 'AgeGroup']):
        # a=f/1.5/low age, b=m/3/high age
        ad_a_df, ad_b_df, hc_a_df, hc_b_df = get_ad_hc_split_dfs(all_data_df, split_col)

        # Compute cross-validation fold sizes
        fold_size_base, fold_size_rem = divmod(train_set_sizes[split_idx], n_folds)
        fold_sizes = []
        for ii in range(0, n_folds):
            if ii < fold_size_rem:
                fold_sizes.append(fold_size_base + 1)
            else:
                fold_sizes.append(fold_size_base)
        assert (sum(fold_sizes) == train_set_sizes[split_idx])

        # Set up the desired test sets
        ad_a_df = assign_test_sets(ad_a_df, n_test_sets, rng)
        hc_a_df = assign_test_sets(hc_a_df, n_test_sets, rng)
        ad_b_df = assign_test_sets(ad_b_df, n_test_sets, rng)
        hc_b_df = assign_test_sets(hc_b_df, n_test_sets, rng)

        for test_idx in range(0, n_test_sets):
            test_set_name = 'test_set_' + str(test_idx)

            test_df = sort_df(pd.concat([ad_a_df[ad_a_df[test_set_name] == 1],
                                         hc_a_df[hc_a_df[test_set_name] == 1],
                                         ad_b_df[ad_b_df[test_set_name] == 1],
                                         hc_b_df[hc_b_df[test_set_name] == 1]]))
            check_unique(test_df)
            Abstract_ADNI_Module.df_diagnostics(test_df, f'Test {test_idx}')
            test_df.to_csv(basename + f'_{split_col}_{test_idx}_test.csv')

            # Mark that nothing has been used in the training / validation sets belong to this test set yet
            ad_a_df.loc[:, 'used_with_curr_test'] = 0
            ad_b_df.loc[:, 'used_with_curr_test'] = 0
            hc_a_df.loc[:, 'used_with_curr_test'] = 0
            hc_b_df.loc[:, 'used_with_curr_test'] = 0

            ratios.sort()
            # Set up the training and validation sets for each test set
            for ratio in ratios:
                # ----- Determine (based on ratio and train_set_size) how many AD/HC F/M/1.5/3 there should be and
                # split into the four corresponding DFs.
                train_set_n_ad_a, train_set_n_ad_b, train_set_n_hc_a, train_set_n_hc_b = \
                    get_train_set_sizes(ad_a_df, ad_b_df, hc_a_df, hc_b_df, ratio, train_set_sizes[split_idx],
                                        ad_fraction=ad_fraction)

                # Compose a training + validation dataset with the desired sex ratio from the remaining non-test data
                # Reuse samples that have been used for earlier ratios wherever possible to minimize training set
                # variations across ratios.
                if ratio == min(ratios):
                    # first ratio, just sample from scratch
                    train_ad_a_df = ad_a_df[ad_a_df[test_set_name] == 0].sample(n=train_set_n_ad_a, random_state=rng)
                    train_ad_b_df = ad_b_df[ad_b_df[test_set_name] == 0].sample(n=train_set_n_ad_b, random_state=rng)
                    train_hc_a_df = hc_a_df[hc_a_df[test_set_name] == 0].sample(n=train_set_n_hc_a, random_state=rng)
                    train_hc_b_df = hc_b_df[hc_b_df[test_set_name] == 0].sample(n=train_set_n_hc_b, random_state=rng)
                else:
                    # We work with increasing ratios, i.e., we now have less males and more females than for the
                    # previous ratio.
                    # Draw males only from the ones that have been used so far
                    train_ad_b_df = ad_b_df[ad_b_df['used_with_curr_test'] == 1].sample(n=train_set_n_ad_b,
                                                                                        random_state=rng)
                    train_hc_b_df = hc_b_df[hc_b_df['used_with_curr_test'] == 1].sample(n=train_set_n_hc_b,
                                                                                        random_state=rng)
                    # Use all females used so far + draw new ones as needed
                    n_prev = ad_a_df['used_with_curr_test'].sum()
                    train_ad_a_df = pd.concat([ad_a_df[ad_a_df['used_with_curr_test'] == 1],
                                               ad_a_df[(ad_a_df[test_set_name] == 0) & (
                                                           ad_a_df['used_with_curr_test'] == 0)].sample(
                                                   n=train_set_n_ad_a - n_prev, random_state=rng)])
                    n_prev = hc_a_df['used_with_curr_test'].sum()
                    train_hc_a_df = pd.concat([hc_a_df[hc_a_df['used_with_curr_test'] == 1],
                                               hc_a_df[(hc_a_df[test_set_name] == 0) & (
                                                           hc_a_df['used_with_curr_test'] == 0)].sample(
                                                   n=train_set_n_hc_a - n_prev, random_state=rng)])

                    # Mark which ones we have used so far with the current test set + this and previous ratios
                ad_a_df.loc[train_ad_a_df.index, 'used_with_curr_test'] = 1
                ad_b_df.loc[train_ad_b_df.index, 'used_with_curr_test'] = 1
                hc_a_df.loc[train_hc_a_df.index, 'used_with_curr_test'] = 1
                hc_b_df.loc[train_hc_b_df.index, 'used_with_curr_test'] = 1

                train_and_vali_df = sort_df(pd.concat([train_ad_a_df, train_ad_b_df, train_hc_a_df, train_hc_b_df]))

                # Set up folds for cross-validation
                train_and_vali_df['fold'] = np.nan
                for fold_idx, fold_size in enumerate(fold_sizes):
                    train_and_vali_df.loc[train_and_vali_df[train_and_vali_df.fold.isna()].sample(n=fold_size,
                                                                                                  random_state=rng).index, 'fold'] = fold_idx

                # check that everything looks nice
                check_unique(pd.concat([test_df, train_and_vali_df]))
                assert (~train_and_vali_df.fold.isna().any())
                assert (len(train_and_vali_df) == train_set_sizes[split_idx])
                assert (np.abs(
                    sum(train_and_vali_df[split_col] == all_data_df[split_col].min()) - ratio * train_set_sizes[
                        split_idx]) <= 2)
                assert (sum(train_and_vali_df.label) > 0.1 * train_set_sizes[split_idx])
                assert (sum(train_and_vali_df.label) < 0.9 * train_set_sizes[split_idx])

                for fold_idx in range(0, n_folds):
                    train_df = train_and_vali_df[train_and_vali_df.fold != fold_idx]
                    val_df = train_and_vali_df[train_and_vali_df.fold == fold_idx]
                    all_df = pd.concat([train_df, val_df, test_df])
                    check_unique(all_df)

                    Abstract_ADNI_Module.df_diagnostics(train_df, f'Training {test_idx}-{fold_idx}')
                    train_df.to_csv(basename + f'_{split_col}_{test_idx}_{ratio:.2f}_{fold_idx}_train.csv')
                    Abstract_ADNI_Module.df_diagnostics(val_df, f'Validation {test_idx}-{fold_idx}')
                    val_df.to_csv(basename + f'_{split_col}_{test_idx}_{ratio:.2f}_{fold_idx}_val.csv')
