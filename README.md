### Feature robustness and sex differences in medical imaging: a case study in MRI-based Alzheimer's disease detection

This repository contains the code used for producing the results presented in Petersen et al., MICCAI 2022, "Feature robustness and sex differences in medical imaging: a case study in MRI-based Alzheimer's disease detection."

Provided are the implementations of
- CNN-based detection of Alzheimer's disease (AD) using 3D MRI volumes,
- Logistic regression-based detection of AD using manually extracted volumetric features, and
- detailed performance analysis with respect to training datasets containing varying amounts of males/females and younger/older subjects

as described in the paper.

Preprocessing (using FreeSurfer and SPM12) is not provided here; see the paper for details and feel free to contact us for help with the reproduction of our experiments.

For access to the raw MRI data, please refer to the Alzheimer's disease neuroimaging initiative (ADNI), https://adni.loni.usc.edu/.

The main files to run are:
1. Repeated_CV_Splitter.py (for generating dataset splits as described in the paper)
2. Low_Dim_Models.py (for training logistic regression models using manually extracted volumetric features, age, and sex)
3. CNN_Model.py (for training CNN models using raw 3D MRI volumes as inputs)
4. analysis.py (for pulling results together and conducting performance analyses)

--

[Eike Petersen](https://e-pet.github.io/) on behalf of the remaining authors: [Aasa Feragen](http://www2.compute.dtu.dk/~afhar/), Maria Luise da Costa Zemsch, Anders Henriksen, Oskar Eiler Wiese Christensen, and [Melanie Ganz](https://sites.google.com/view/melanieganz/home), 2022.

This research was supported by Danmarks Frie Forskningsfond (9131-00097B, project [Bias and fairness in medicine](http://fairmed.compute.dtu.dk/)), the Novo Nordisk Foundation through the [Center for Basic Machine Learning Research in Life Science](https://www.mlls.dk/) (NNF20OC0062606) and the [Pioneer Centre for AI](https://www.aicentre.dk/), DNRF grant number P1.