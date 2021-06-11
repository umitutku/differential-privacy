# Differential-privacy & influence based pruning

R&amp;D project on the implications of influence-based pruning methods on differential-privacy. The project uses the Adult and the CIFAR-10 datasets to demonstrate whether removing highly inflential examples—those examples that correlate with a negative impact on model accuracy—can improve the privacy budget (epsilon). The project heavily relies on tensorflow and DP-SGD.

## Primary Libraries

```bash
tensorflow 2.4.1
```

```bash
tensorflow-datasets
```

```bash
tensorflow-privacy
```

## Scripts
The project is executed through multiple jupyter notebooks which are written to train dpsgd and non-dpsgd models, optimise hyperparameters and analyse results.

**adult_training.ipynb**
> A script which loads the pre-sorted adult dataset and trains a simple neural network and calculates influence scores for each training examples.
> Influence scores are calculated using the -run_self_influence- function. The script created an influence-based ordered dataset which is saved as a pickle file 
> and later used for analysis

**adult_analysis.ipynb**
> This script is used to load the influence-based ordered dataset and analyse the implicatons of the influence method and influence-evolution across epochs. The last section is used to create lists of pruned dataset indices, which are generated for a given fraction and later used during dp-sgd training.

**adult_dpsgd_training.ipynb**
> This script uses the results from the dp-sgd hyperparameter optimisation training and analysis, and the pruned dataset indices, to run multiple experiments for a given scenario of hyperparameters.
