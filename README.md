### Feature robustness and sex differences in medical imaging: a case study in MRI-based Alzheimer's disease detection

This repository contains the code used for producing the results presented in Petersen et al., MICCAI 2022, [Feature robustness and sex differences in medical imaging: a case study in MRI-based Alzheimer's disease detection](https://link.springer.com/chapter/10.1007/978-3-031-16431-6_9) [[arxiv]](https://arxiv.org/abs/2204.01737).

Provided are the implementations of
- CNN-based detection of Alzheimer's disease (AD) using 3D MRI volumes,
- Logistic regression-based detection of AD using manually extracted volumetric features, and
- detailed performance analysis with respect to training datasets containing varying amounts of males/females and younger/older subjects

as described in the paper.

Preprocessing (using FreeSurfer and SPM12) is not provided here; see below for instructions and feel free to contact us for help with the reproduction of our experiments.

For access to the raw MRI data, please refer to the Alzheimer's disease neuroimaging initiative (ADNI), https://adni.loni.usc.edu/.

The main files to run are:
1. Repeated_CV_Splitter.py (for generating dataset splits as described in the paper)
2. Low_Dim_Models.py (for training logistic regression models using manually extracted volumetric features, age, and sex)
3. CNN_Model.py (for training CNN models using raw 3D MRI volumes as inputs)
4. analysis.py (for pulling results together and conducting performance analyses)

##### Data selection and preprocessing
- A single T1-weighted MP-RAGE recording per subject was selected from ADNI1-3. If multiple recordings were available, the first one was used.
- Recordings for which either FreeSurfer or SPM threw errors were discarded.
- For the MCI analysis, recordings for which a classification into stable or progressive MCI according to our definition was not possible were discarded as well. (This was the case when there was no AD diagnosis during follow-up _and_ there was no follow-up data beyond 2 years.)
- SPM processing was done using Matlab R2019a, FreeSurfer was v. 7.1.1.
- For the logistic regression analysis, run [recon-all](https://surfer.nmr.mgh.harvard.edu/fswiki/recon-all) followed by [asegstats2table](https://surfer.nmr.mgh.harvard.edu/fswiki/asegstats2table) and [aparcstats2table](https://surfer.nmr.mgh.harvard.edu/fswiki/aparcstats2table) in FreeSurfer, then use SPM12 and their volume quantification to [get ICV](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4265726/). The FreeSurfer command will take a long time to run, i.e., 12-16 hours per recording (!) on our system.
- For the CNN analysis, use the recon-all (see above) output norm.mgz and transform it into MNI305 using [mri_vol2vol](https://surfer.nmr.mgh.harvard.edu/fswiki/mri_vol2vol) and the tailarach transform given by FreeSurfer, e.g., using `mri_vol2vol --mov norm.mgz --lta transforms/talairach.lta --o norm_mni305.nii --targ mri/mni305.cor.mgz`. The output is a file called `norm_mni305.mgz` that should be used as the input for the CNN.

--

[Eike Petersen](https://e-pet.github.io/) on behalf of the remaining authors: [Aasa Feragen](http://www2.compute.dtu.dk/~afhar/), Maria Luise da Costa Zemsch, Anders Henriksen, Oskar Eiler Wiese Christensen, and [Melanie Ganz](https://sites.google.com/view/melanieganz/home), 2022.

This research was supported by Danmarks Frie Forskningsfond (9131-00097B, project [Bias and fairness in medicine](http://fairmed.compute.dtu.dk/)), the Novo Nordisk Foundation through the [Center for Basic Machine Learning Research in Life Science](https://www.mlls.dk/) (NNF20OC0062606) and the [Pioneer Centre for AI](https://www.aicentre.dk/), DNRF grant number P1.
