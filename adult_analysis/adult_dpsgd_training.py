#!/usr/bin/env python
# coding: utf-8

# # Adult DP-SGD Training
# This notebook is used to train pruned and non-pruned datasets by using pre-selected noise multiplier and clipping norm pairs. The script produces a pandas dataframe for accuracy, validation accuracy and epsilon for a list of epochs.

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import feature_column
import tensorflow_datasets as tfds
import os
from os import path
import pickle
import time
import datetime


# In[ ]:


# import tf-privacy libraries
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy_lib
from tensorflow_privacy.privacy.analysis.rdp_accountant import compute_rdp
from tensorflow_privacy.privacy.analysis.rdp_accountant import get_privacy_spent

from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasSGDOptimizer
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasAdamOptimizer
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras_vectorized import VectorizedDPKerasSGDOptimizer
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras_vectorized import VectorizedDPKerasAdagradOptimizer


# ### Pruning methods

# In[ ]:


# method to remove random entries, rounding to a dataset divisible by 100
def prune_random_entries(df, prune_list, batch_size):
    new_df = df.drop(np.random.choice(df.index, len(prune_list), replace=False))
    return new_df.drop(np.random.choice(new_df.index, (new_df.shape[0]%batch_size), replace=False))


# In[ ]:


# method to remove influential entries
# > remainder 100 entries are removes randomly thereafter
def prune_influential_entries(df, prune_list, batch_size):
    new_df = df.drop(df.index[prune_list])
    return new_df.drop(np.random.choice(new_df.index, (new_df.shape[0]%batch_size), replace=False))


# ### Epsilon calculator

# In[ ]:


# method which takes in steps and returns privacy spent in steps taken
# > STEPS
# > NOISE_MULTIPLIER
# > BATCH_SIZE
# > DELTA
def compute_epsilon(steps, batch_size, num_training_examples, noise_multiplier, delta):
    """Computes epsilon value for given hyperparameters."""
    if noise_multiplier == 0.0:
        return float('inf')
    orders = [1 + x / 10. for x in range(1, 100)] + list(range(12, 64))
    sampling_probability = batch_size / num_training_examples
    rdp = compute_rdp(q=sampling_probability,
                      noise_multiplier=noise_multiplier,
                      steps=steps,
                      orders=orders)
    # Delta is set to approximate 1 / (number of training points).
    return get_privacy_spent(orders, rdp, target_delta=delta)[0]


# ### Load dataset

# In[ ]:


# load training, test and validation datasets
train_df = pd.read_csv("data/train-one-hot.csv")

test_df = pd.read_csv("data/test-one-hot.csv")
test_target_df = test_df.pop('salary')

val_df = pd.read_csv("data/val-one-hot.csv")
val_target_df = val_df.pop('salary')


# load pruned indices
#> dictionary of 

#file = 'pruned_ds_cpave_bs500_upper'
#file = 'pruned_ds_cp50_bs500_upper'
file = 'pruned_ds_cpave_bs500_2'

with open('results/{}.pickle'.format(file), 'rb') as handle:
    prune_indices_list = pickle.load(handle)
    print ("Running experiments from file: {}".format(file))


# ### Hyperparameters


#NUM_TRAIN_EXAMPLES=len(train_target_df.values)
EPOCHS=100
BATCH_SIZE=100
N_MICROBATCHES=100
LEARNING_RATE=0.001
DELTA=1e-5
NOISE_MULTIPLIER = 2.5
L2_NORM_CLIP = 10
epoch_scan = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
#epoch_scan = list(range(1, 101))


# ### Training
# ### Scanning batch_size


def run_dpsgd_scan_batch_size(l2_norm_clip, 
                              noise_multiplier, 
                              epoch_scan, 
                              prune_type, 
                              prune_indices_list,
                              BATCH_SIZE_ARRAY,
                              EXPERIMENTS):
    start = time.time()
    total_loops = len(prune_indices_list["frac_list"])*EXPERIMENTS*len(BATCH_SIZE_ARRAY)
    current_loop = 0
    
    
    # creating a dataframe to store information
    columns = ['run_number', 'prune_type', 'prune_frac', 'batch_size', 'steps', 'epochs', 'noise_multiplier', 'clipping_norm', 'acc', 'val_acc', 'epsilon']
    df = pd.DataFrame(columns=columns)
    data = []
    
    for i, prune_frac in enumerate(prune_indices_list["frac_list"]):        
        for batch_size in BATCH_SIZE_ARRAY:
            
            N_MICROBATCHES=batch_size
            
            # get influential prune list
            prune_list = prune_indices_list["prune_indices"][i]
            
            # prune dataset
            if prune_type == "random":
                pruned_train_df = prune_random_entries(train_df, prune_list, batch_size)
                pruned_train_target_df = pruned_train_df.pop('salary')
            else:
                pruned_train_df = prune_influential_entries(train_df, prune_list, batch_size)
                pruned_train_target_df = pruned_train_df.pop('salary')
            
            # total number of training examples
            NUM_TRAIN_EXAMPLES = len(pruned_train_df.values)

            for run in range(EXPERIMENTS):
                # reset tf session
                tf.keras.backend.clear_session()
                # set optimiser options
                optimizer = VectorizedDPKerasSGDOptimizer(
                    l2_norm_clip=L2_NORM_CLIP,
                    noise_multiplier=NOISE_MULTIPLIER,
                    num_microbatches=N_MICROBATCHES,
                    learning_rate=LEARNING_RATE
                )
                # define model
                model = tf.keras.Sequential([
                    tf.keras.Input(shape=(63,)),
                    tf.keras.layers.Dense(128, activation="relu"),
                    tf.keras.layers.Dense(64, activation="relu"),
                    tf.keras.layers.Dense(32, activation="relu"),
                    tf.keras.layers.Dense(1)]
                )
                # compile model
                model.compile(optimizer=optimizer,
                              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True,
                                                                      reduction=tf.losses.Reduction.NONE),
                              metrics=['accuracy'])
                # start training
                history = model.fit(pruned_train_df.values,
                                    pruned_train_target_df.values,
                                    validation_data=(val_df.values, val_target_df.values),
                                    batch_size=batch_size,
                                    epochs=EPOCHS, 
                                    verbose=0)
                # calculate and append information required: 
                for epoch in epoch_scan:
                    STEPS = epoch * NUM_TRAIN_EXAMPLES / batch_size
                    values = [run,
                              "none" if prune_frac==0 else "random" if prune_type == "random" else "influential",
                              prune_frac, 
                              batch_size, 
                              STEPS,
                              epoch,
                              NOISE_MULTIPLIER, 
                              L2_NORM_CLIP, 
                              history.history["accuracy"][epoch-1],
                              history.history["val_accuracy"][epoch-1],
                              compute_epsilon(STEPS,
                                              batch_size,
                                              NUM_TRAIN_EXAMPLES,
                                              NOISE_MULTIPLIER, 
                                              DELTA)]
                    zipped = zip(columns, values)
                    a_dictionary = dict(zipped)
                    data.append(a_dictionary)
                # printing information loop information
                loss, acc = model.evaluate(val_df.values, val_target_df.values, verbose=0)
                current_loop += 1
                print ("# {} out of {} || Prune frac: {} -- Prune Type: {} -- Accuracy {}".format(current_loop,
                                                                                                total_loops, 
                                                                                                prune_frac, 
                                                                                                "random" if prune_type == "random" else "influential", 
                                                                                                acc))
                print("Elapsed time:", datetime.timedelta(seconds=time.time() - start))
            
    end = datetime.timedelta(seconds=time.time() - start)
    df = df.append(data, True)
    print("Completed {} experiments in {}".format(total_loops, end))
    return df     


# In[ ]:


# training options
PRUNE_TYPE = "influential"
EXPERIMENTS = 6
BATCH_SIZE_ARRAY = [25, 30, 35, 40]

df = run_dpsgd_scan_batch_size(L2_NORM_CLIP, 
                               NOISE_MULTIPLIER, 
                               epoch_scan,
                               PRUNE_TYPE,
                               prune_indices_list,
                               BATCH_SIZE_ARRAY,
                               EXPERIMENTS)


# save output
df.to_csv('results/dp_sgd_pruning_results_infl_cpave_bs500_2.csv', index=False)