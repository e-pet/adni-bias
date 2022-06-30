import re
from argparse import ArgumentParser
from distutils.util import strtobool
from os.path import exists

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from scipy.stats import wilcoxon
from statsmodels.formula.api import ols

from Slice_CNN import ADNI_3slice_CNN
from ADNI_Feature_Module import ADNI_Feature_Module
from ADNI_Image_Module import ADNI_Image_Module
from ADNI_Model import ADNI_Model
from CNN_Model import get_CNN_chkpt_file, ADNI_CNN_Model
from Low_Dim_Models import LogReg, Fake, get_LR_chkpt_file
from Repeated_CV_Splitter import get_adhc_split_csvs


def get_results_df(result, val_result, model, f_ratio, dataset='adhc', run_idx=None, fold=None):
    test_results_df = pd.DataFrame(
        columns=['f_ratio', 'dataset', 'model', 'split_var', 'age_group', 'run_idx', 'fold', 'auc', 'val_auc', 'tprs',
                 'ece', 'ace'
                        'conf', 'rel_freq', 'acc', 'loss'])
    test_results_df = test_results_df.append(
        {'f_ratio': f_ratio, 'dataset': dataset, 'model': model, 'split_var': 1, 'run_idx': run_idx, 'fold': fold,
         'auc': result.auc_1, 'val_auc': val_result.auc,
         'tprs': result.tprs_1, 'ece': result.ece_1, 'ace': result.ace_1, 'rel_freq': result.rel_freq_1,
         'acc': result.acc_1, 'loss': result.loss_1},
        ignore_index=True)
    test_results_df = test_results_df.append(
        {'f_ratio': f_ratio, 'dataset': dataset, 'model': model, 'split_var': 2, 'run_idx': run_idx, 'fold': fold,
         'auc': result.auc_2, 'val_auc': val_result.auc,
         'tprs': result.tprs_2, 'ece': result.ece_2, 'ace': result.ace_2, 'rel_freq': result.rel_freq_2,
         'acc': result.acc_2, 'loss': result.loss_2},
        ignore_index=True)

    return test_results_df


def test_model(gpus, mdl, adni_dm, mdl_name, split_var, ratio, log_dir):
    exp_name = get_experiment_name(mdl_name, split_var, ratio)
    adni_dm.setup()
    trainer = Trainer(gpus=gpus, logger=None)
    trainer.test(mdl, dataloaders=adni_dm.val_dataloader())
    val_results = mdl.test_results
    mdl.set_gmean_threshold()  # set threshold based on val performance

    tb_logger = TensorBoardLogger(log_dir + 'ADHC-test/',
                                  name=exp_name,
                                  version=f'test set {run_idx}, fold {fold}')
    trainer = Trainer(gpus=gpus, logger=tb_logger)
    trainer.test(mdl, dataloaders=adni_dm.test_dataloader())
    ADHC_results_df = get_results_df(mdl.test_results, val_results, mdl_name, ratio, run_idx=run_idx, fold=fold)

    tb_logger = TensorBoardLogger(log_dir + 'MCI-test/',
                                  name=exp_name,
                                  version=f'test set {run_idx}, fold {fold}')
    trainer = Trainer(gpus=gpus, logger=tb_logger)
    trainer.test(mdl, dataloaders=adni_dm.test_mci_dataloader())
    MCI_results_df = get_results_df(mdl.test_results, val_results, mdl_name, ratio, dataset='mci', run_idx=run_idx,
                                    fold=fold)

    return ADHC_results_df, MCI_results_df


def get_experiment_name(mdl_name, split_var, ratio):
    if split_var == 0:
        exp_name = f'{mdl_name}_sex-r{ratio:.2f}'
    elif split_var == 1:
        exp_name = f'{mdl_name}_age-r{ratio:.2f}'
    else:
        raise NotImplementedError
    return exp_name


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("-g", "--gpu", dest="gpu", help="GPU to use", type=int, default=4)
    parser.add_argument("-r", "--reload", dest="reload", type=lambda x: bool(strtobool(x)), default=True)
    parser.add_argument("-t", "--test", dest="test", type=lambda x: bool(strtobool(x)), default=True)
    parser.add_argument("-s", "--split_var", dest="split_var", type=int, default=0)
    parser.add_argument("-l", "--log_dir", dest="log_dir", type=str, default="/dtu-compute/ADNIbias/ewipe")
    args = parser.parse_args()
    gpus = [args.gpu]
    print(args.reload)

    # Yes, the ADNI3 images are in the ADNI1 directory for some reason
    image_paths = ["/scratch/ewipe/freesurfer_ADNI1",
                   "/scratch/ewipe/freesurfer_ADNI2",
                   "/scratch/ewipe/freesurfer_ADNI1"]

    if args.split_var == 0:
        split_var = 'Sex'
        analyses_csv = "analysis_results_sex"
    elif args.split_var == 1:
        split_var = 'AgeGroup'
        analyses_csv = "analysis_results_age"
    else:
        raise NotImplementedError

    if args.test:
        analyses_csv = analyses_csv + "_fake"

    plot_file_stem = analyses_csv
    analyses_csv = analyses_csv + ".csv"

    if args.reload:
        print("Reloading analysis results from file " + analyses_csv + ".")
        test_results_df = pd.read_csv(analyses_csv)

        # The tpr, prob_true and prob_pred columns have arrays in each cell and are written+read as strings

        def make_array(text):
            # DataFrame.to_csv apparently adds lots of whitespace for nice formatting
            # Moreover, np.fromstring cannot deal with opening/closing brackets?!
            return np.fromstring(re.sub(r'[\n\t ]+', ' ', text.strip('[]')), sep=' ')


        test_results_df['tprs'] = test_results_df['tprs'].apply(make_array)
        test_results_df['rel_freq'] = test_results_df['rel_freq'].apply(make_array)
    else:
        results_dfs_list = []

        # CNN results
        for ratio in [0.0, 0.25, 0.5, 0.75, 1.0]:
            for run_idx in range(0, 5):
                for fold in range(0, 5):
                    for CNN_type in ['3sliceCNN', 'CNN']:
                        print(
                            f'------- F_RATIO {ratio}, RUN_IDX {run_idx}, FOLD {fold}, {CNN_type}, FAKE {args.test} -------')

                        chkpt_file = get_CNN_chkpt_file(CNN_type, args.split_ver, args.split_var, ratio, run_idx, fold,
                                                        args.test)

                        if exists(chkpt_file):
                            print("Analyzing chkpt " + chkpt_file)
                            if CNN_type == 'CNN':
                                mdl = ADNI_CNN_Model.load_from_checkpoint(chkpt_file)
                            else:
                                mdl = ADNI_3slice_CNN.load_from_checkpoint(chkpt_file)

                            adhc_split_csvs = get_adhc_split_csvs(split_var, run_idx, ratio, fold)

                            adni_dm = ADNI_Image_Module(image_paths=image_paths, adni_set=3, batch_size=1,
                                                        adhc_split_csvs=adhc_split_csvs,
                                                        sliced=True if CNN_type == '3sliceCNN' else False,
                                                        fake_diff=args.test)

                            if args.split_var == 0:
                                mdl.test_split_var = 'sex'
                            elif args.split_var == 1:
                                mdl.test_split_var = 'age_group'
                            else:
                                raise NotImplementedError

                            ADHC_results_df, MCI_results_df = test_model(gpus, mdl, adni_dm, CNN_type, args.split_var,
                                                                         ratio, args.log_dir + CNN_type + "_logs/")
                            results_dfs_list.append(ADHC_results_df)
                            results_dfs_list.append(MCI_results_df)

                        else:
                            print("Skipped checkpoint because file could not be found: " + chkpt_file)

        # LR results
        for ratio in [0, 0.25, 0.5, 0.75, 1.0]:
            for run_idx in range(0, 5):
                for fold in range(0, 5):
                    print(f'------- RATIO {ratio}, RUN_IDX {run_idx}, FOLD {fold}, LR, FAKE {args.test} -------')
                    # reload pre-computed data splits
                    adhc_split_csvs = get_adhc_split_csvs(args.split_ver, split_var, run_idx, ratio, fold)
                    adni_dm = ADNI_Feature_Module(adni_set=3, adhc_split_csvs=adhc_split_csvs, batch_size=256,
                                                  use_sex=False if args.split_var == 0 else True)

                    if args.test:
                        mdl = Fake(fr=ratio)
                    else:
                        chkpt_file = get_LR_chkpt_file(args.split_ver, split_var, ratio, run_idx, fold)
                        mdl = LogReg.load_from_checkpoint(chkpt_file)

                    if args.split_var == 0:
                        mdl.test_split_var = 'sex'
                    elif args.split_var == 1:
                        mdl.test_split_var = 'age_group'
                    else:
                        raise NotImplementedError

                    ADHC_results_df, MCI_results_df = test_model(0, mdl, adni_dm, 'LR', args.split_var, ratio,
                                                                 args.log_dir + "LR_logs/")
                    results_dfs_list.append(ADHC_results_df)
                    results_dfs_list.append(MCI_results_df)

                    # Put everything together
        test_results_df = pd.concat(results_dfs_list)
        # would like to do this elsewhere, but well...
        test_results_df['auc'] = test_results_df['auc'].astype(float)
        print("Saving analysis results to file " + analyses_csv + ".")
        test_results_df.to_csv(analyses_csv)

    num_models = len(test_results_df.model.unique())
    num_datasets = len(test_results_df.dataset.unique())

    # --------- PLOTTING RESULTS
    tex_fonts = {
        # Use LaTeX to write all text
        "text.usetex": True,
        "font.family": "serif",
        # "font.serif": ["Times", "Palatino", "New Century Schoolbook", "Bookman", "Computer Modern Roman"],
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "font.size": 9,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        'text.latex.preamble': [r'\usepackage{siunitx}',
                                r'\usepackage{upgreek}',
                                r'\usepackage{lmodern}',
                                r'\DeclareMathAlphabet{\mathsfit}{T1}{\sfdefault}{\mddefault}{\sldefault}']
    }

    plt.rcParams.update(tex_fonts)
    sns.set_theme(style="whitegrid", rc=tex_fonts)
    textwidth_in = 4.8
    golden_ratio = 1.4
    numcol = num_datasets * 2
    numrow = num_models
    tile_width = textwidth_in / numcol
    tweak_factor = 0.9
    tile_height = textwidth_in / numcol / golden_ratio * 2 * tweak_factor

    # rename some columns + values so that it looks nicer in the plots
    test_results_df.rename(columns=
                           {"auc": "AUC", "f_ratio": "FR", "model": "Model", "acc": "ACC", "dataset": "Dataset",
                            "ace": "ACE", 'loss': 'NLL'},
                           inplace=True)
    test_results_df.loc[test_results_df.Model == "tCNN", "Model"] = "CNN"
    test_results_df.loc[test_results_df.Dataset == "mci", "Dataset"] = "pMCI/sMCI"
    test_results_df.loc[test_results_df.Dataset == "adhc", "Dataset"] = "AD/HC"

    if split_var == 'Sex':
        split_var_map = {1: 'f', 2: 'm'}
        x_var = 'FR'
    elif split_var == 'AgeGroup':
        split_var_map = {1: 'y', 2: 'o'}
        test_results_df.rename(columns={'FR': 'YR'}, inplace=True)
        x_var = 'YR'
    else:
        raise NotImplementedError

    test_results_df.loc[:, 'Dataset+SplitVar'] = \
        test_results_df.Dataset + ', ' + test_results_df.split_var.replace(split_var_map)
    test_results_df.loc[:, 'run_idx+' + x_var + '+fold'] = \
        test_results_df.run_idx.astype(str) + ', ' + test_results_df[x_var].astype(
            str) + ', ' + test_results_df.fold.astype(str)

    # T tests
    for model in test_results_df.Model.unique():
        for dataset in test_results_df.Dataset.unique():
            subset = test_results_df[(test_results_df.Model == model) & (test_results_df.Dataset == dataset)]
            subset_1_sorted = subset[subset.split_var == 1].sort_values(by=['run_idx', x_var, 'fold'])
            subset_2_sorted = subset[subset.split_var == 2].sort_values(by=['run_idx', x_var, 'fold'])
            _, pval_auc = wilcoxon(x=subset_1_sorted.AUC, y=subset_2_sorted.AUC, alternative='two-sided')
            _, pval_acc = wilcoxon(x=subset_1_sorted.ACC, y=subset_2_sorted.ACC, alternative='two-sided')
            _, pval_ace = wilcoxon(x=subset_1_sorted.ACE, y=subset_2_sorted.ACE, alternative='two-sided')
            _, pval_nll = wilcoxon(x=subset_1_sorted.NLL, y=subset_2_sorted.NLL, alternative='two-sided')
            for split_var_val in test_results_df.split_var.unique():
                subset_split_var_val = subset[subset.split_var == split_var_val]
                print(
                    f'---- MEAN AUC FOR {model} ON {split_var_val} ON {dataset}: {subset_split_var_val.AUC.mean()} +- {subset_split_var_val.AUC.std()} ----')
                print(
                    f'---- MEAN ACC FOR {model} ON {split_var_val} ON {dataset}: {subset_split_var_val.ACC.mean()} +- {subset_split_var_val.ACC.std()} ----')
                print(
                    f'---- MEAN ACE FOR {model} ON {split_var_val} ON {dataset}: {subset_split_var_val.ACE.mean()} +- {subset_split_var_val.ACE.std()} ----')
                print(
                    f'---- MEAN NLL FOR {model} ON {split_var_val} ON {dataset}: {subset_split_var_val.NLL.mean()} +- {subset_split_var_val.NLL.std()} ----')
            # Bonferroni correction: we look at two measures, AUC and ACC, hence factor=2
            corr_factor = 2
            print(
                f'---- P-VALUE (CORRECTED) FOR M-F AUC DIFFERENCES FOR {model} ON {dataset}: {pval_auc * corr_factor}')
            print(
                f'---- P-VALUE (CORRECTED) FOR M-F ACC DIFFERENCES FOR {model} ON {dataset}: {pval_acc * corr_factor}')
            print(f'---- P-VALUE (UNCORRECTED) FOR M-F ACE DIFFERENCES FOR {model} ON {dataset}: {pval_ace}')
            print(f'---- P-VALUE (UNCORRECTED) FOR M-F NLL DIFFERENCES FOR {model} ON {dataset}: {pval_nll}')

    subset_CNN = test_results_df[(test_results_df.Model == 'CNN')].sort_values(by=['run_idx', x_var, 'fold'])
    subset_LR = test_results_df[(test_results_df.Model == 'LR') &
                                (test_results_df['run_idx+' + x_var + '+fold'].isin(
                                    subset_CNN['run_idx+' + x_var + '+fold'].unique()))].sort_values(
        by=['run_idx', x_var, 'fold'])

    if len(subset_CNN) > 0 and len(subset_LR) > 0:
        _, pval_auc = wilcoxon(x=subset_LR.AUC, y=subset_CNN.AUC, alternative='two-sided')
        _, pval_acc = wilcoxon(x=subset_LR.ACC, y=subset_CNN.ACC, alternative='two-sided')
        _, pval_ace = wilcoxon(x=subset_LR.ACE, y=subset_CNN.ACE, alternative='two-sided')
        _, pval_nll = wilcoxon(x=subset_LR.NLL, y=subset_CNN.NLL, alternative='two-sided')
        # Bonferroni correction: we look at two measures, AUC and ACC, hence factor=2
        corr_factor = 2
        print(f'---- P-VALUE (CORRECTED) FOR AUC DIFFERENCES BETWEEN LR AND CNN: {pval_auc * corr_factor}')
        print(f'---- P-VALUE (CORRECTED) FOR ACC DIFFERENCES BETWEEN LR AND CNN: {pval_acc * corr_factor}')
        print(f'---- P-VALUE (UNCORRECTED) FOR ACE DIFFERENCES BETWEEN LR AND CNN: {pval_ace}')
        print(f'---- P-VALUE (UNCORRECTED) FOR NLL DIFFERENCES BETWEEN LR AND CNN: {pval_nll}')


    def gen_plot_panel(data, quantity, **kws):
        mdl = ols(formula=quantity + ' ~ ' + x_var, data=data)
        results = mdl.fit()
        slope = results.params[1]
        slope_std = results.bse[1]
        P_value = results.pvalues[1]
        # Bonferroni correction: we perform eight tests per dataset (AD/HC, pMCI/sMCI): LR/CNN * m/f * AUC/ACC
        corr_factor = 8
        p_corr = min(1, P_value * corr_factor)
        ax = plt.gca()
        slope_str = f"$m{{=}}${slope:.3f}$\pm${slope_std:.3f}\n"
        if p_corr > 0.001:
            p_str = "$p_{\\mathrm{corr}}{{=}}$" + f"{p_corr:.3f}\n"
        else:
            p_str = "$p_{\\mathrm{corr}}{{=}}$" + f"{p_corr:.1e}\n"
        mu_str = f"$\mu{{=}}${data[quantity].mean():.3f}$\pm${data[quantity].std():.3f}"
        text_str = slope_str + p_str + mu_str
        t = ax.text(.07, .1, text_str, transform=ax.transAxes, fontsize=7)
        t.set_bbox(dict(facecolor='white', alpha=0.5))
        sns.regplot(x=x_var, y=quantity, data=data, ax=ax, scatter=False, fit_reg=True, color="black",
                    line_kws={'linewidth': 1})
        data = data.copy()
        data.loc[:, x_var + '_raw'] = data[x_var]
        data.loc[:, x_var] = data[x_var] + np.random.uniform(-0.08, 0.08, data[x_var].shape)
        sns.scatterplot(x=x_var, y=quantity, data=data, hue=x_var + "_raw", s=10, ax=ax, palette="viridis", linewidth=0,
                        alpha=0.7)


    def beautify_fig(g):
        g.set_titles(col_template="{col_name}", row_template="{row_name}")
        g.set(xticks=(0, 0.25, 0.5, 0.75, 1.0))
        g.set_xticklabels([0, '', 0.5, '', 1.0])
        g.fig.set_dpi(600)
        return g


    # --- AUC plots
    g = sns.FacetGrid(col="Dataset+SplitVar", row="Model", palette="viridis", data=test_results_df,
                      height=tile_height, aspect=golden_ratio / 2 / tweak_factor, margin_titles=True,
                      row_order=['LR', 'CNN'])
    g.map_dataframe(lambda data, **kws: gen_plot_panel(data, 'AUC', **kws))
    g.set(ylim=(0.3, 1))
    g.set(yticks=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0))
    beautify_fig(g)
    g.savefig(plot_file_stem + "-auc.pdf", bbox_inches="tight", transparent=True)

    # # --- ACC plots
    g = sns.FacetGrid(col="Dataset+SplitVar", row="Model", palette="viridis", data=test_results_df,
                      height=tile_height, aspect=golden_ratio / 2 / tweak_factor, margin_titles=True,
                      row_order=['LR', 'CNN'])
    g.map_dataframe(lambda data, **kws: gen_plot_panel(data, 'ACC', **kws))
    g.set(ylim=(0.3, 1))
    g.set(yticks=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0))
    beautify_fig(g)
    g.savefig(plot_file_stem + "-acc.pdf", bbox_inches="tight", transparent=True)

    # # --- ACE plots
    g = sns.FacetGrid(col="Dataset+SplitVar", row="Model", palette="viridis", data=test_results_df,
                      height=tile_height, aspect=golden_ratio / 2 / tweak_factor, margin_titles=True,
                      row_order=['LR', 'CNN'])
    g.map_dataframe(lambda data, **kws: gen_plot_panel(data, 'ACE', **kws))
    g.set(ylim=(0.2, 0.6))
    g.set(yticks=(0.2, 0.3, 0.4, 0.5, 0.6))
    beautify_fig(g)
    g.savefig(plot_file_stem + "-ace.pdf", bbox_inches="tight", transparent=True)

    # --- NLL plots
    g = sns.FacetGrid(col="Dataset+SplitVar", row="Model", palette="viridis", data=test_results_df,
                      height=tile_height, aspect=golden_ratio / 2 / tweak_factor, margin_titles=True,
                      row_order=['LR', 'CNN'])
    g.map_dataframe(lambda data, **kws: gen_plot_panel(data, 'NLL', **kws))
    beautify_fig(g)
    g.savefig(plot_file_stem + "-nll.pdf", bbox_inches="tight", transparent=True)

    # --- ROC + reliability plots
    N_roc = len(ADNI_Model.base_fpr)
    N_rel = len(ADNI_Model.base_conf)
    roc_data = pd.DataFrame(columns=['split_var', x_var, 'Dataset', 'Model', 'fpr', 'tpr'])
    rel_data = pd.DataFrame(columns=['split_var', x_var, 'Dataset', 'Model', 'conf', 'rel_freq'])
    for idx, row in test_results_df.iterrows():
        local_roc_df = pd.DataFrame({'split_var': np.tile(row.split_var, (N_roc,)),
                                     x_var: np.tile(row[x_var], (N_roc,)),
                                     'Dataset': np.tile(row.Dataset, (N_roc,)),
                                     'Model': np.tile(row.Model, (N_roc,)),
                                     'fpr': ADNI_Model.base_fpr,
                                     'tpr': row.tprs})
        roc_data = roc_data.append(local_roc_df, ignore_index=True)

        local_rel_df = pd.DataFrame({'split_var': np.tile(row.split_var, (N_rel,)),
                                     x_var: np.tile(row[x_var], (N_rel,)),
                                     'Dataset': np.tile(row.Dataset, (N_rel,)),
                                     'Model': np.tile(row.Model, (N_rel,)),
                                     'conf': ADNI_Model.base_conf,
                                     'rel_freq': row.rel_freq})
        rel_data = rel_data.append(local_rel_df, ignore_index=True)

    f_roc, axes_roc = plt.subplots(num_models, 2 * num_datasets, figsize=(tile_width * numcol, tile_height * numrow),
                                   dpi=600)
    axes_roc = axes_roc.flatten()
    f_rel, axes_rel = plt.subplots(num_models, 2 * num_datasets, figsize=(tile_width * numcol, tile_height * numrow),
                                   dpi=600)
    axes_rel = axes_rel.flatten()
    subplot_idx = 0
    for model in roc_data.Model.unique():
        for split_var_val in roc_data.split_var.unique():
            for dataset in roc_data.Dataset.unique():
                # ROC plot
                h = sns.lineplot(data=roc_data[(roc_data.Model == model) &
                                               (roc_data.split_var == split_var_val) &
                                               (roc_data.Dataset == dataset)],
                                 x="fpr", y="tpr", hue=x_var, palette="viridis",
                                 ax=axes_roc[subplot_idx], legend=False)
                axes_roc[subplot_idx].set_title(', '.join([model, str(split_var_val), dataset]))
                h.set_ylim(0, 1)
                h.set_xlim(0, 1)
                # Reliability plot
                h = sns.lineplot(data=rel_data[(rel_data.Model == model) &
                                               (rel_data.split_var == split_var_val) &
                                               (rel_data.Dataset == dataset)],
                                 x="conf", y="rel_freq", hue=x_var, palette="viridis",
                                 ax=axes_rel[subplot_idx], legend=False)
                axes_rel[subplot_idx].set_title(', '.join([model, str(split_var_val), dataset]))
                h.set_ylim(0, 1)
                h.set_xlim(0, 1)
                subplot_idx += 1

    handles, labels = axes_roc[0].get_legend_handles_labels()
    f_roc.legend(handles, labels, loc='center right')
    handles, labels = axes_rel[0].get_legend_handles_labels()
    f_rel.legend(handles, labels, loc='center right')
    f_roc.savefig(plot_file_stem + "-roc.png", bbox_inches="tight")
    f_rel.savefig(plot_file_stem + "-rel.png", bbox_inches="tight")
