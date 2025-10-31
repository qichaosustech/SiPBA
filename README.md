# Single-loop Pessimistic Bilevel Algorithm (SiPBA)

## Overview

All experiments in this project are conducted using Python **3.12**.

## Synthetic Example

1. **Convergence Curve**&#x20;

   To draw the results in Figure 1, please run

   &#x20;`Synthetic example/AdaProx_PD.py`, `Synthetic example/AdaProx_SG.py`, `Synthetic example/SiPBA.py` and  `Synthetic example/plot.py`

2. **Results in Table 1**

   To get the results in Table 1, please run

   `compactschotles.py` and `detailedschotles.py`

3. **Ablation Analysis**&#x20;

   To get  the results for **ablation analysis**, execute: `Synthetic example/ablation.py`

4. **Time and Iter. v.s. Dimensions**
   To study the relationship between computation time, iterations, and problem dimensions, run:
   `Synthetic example/dimension.py`

## Spam Classification

Please download the required datasets from the web and extract them into the folder `Spam/SPAMDATA`. Below are the sources for the datasets:

* [TREC 2006 dataset](https://plg.uwaterloo.ca/cgi-bin/cgiwrap/gvcormac/foo06)

* [TREC 2007 dataset](https://plg.uwaterloo.ca/cgi-bin/cgiwrap/gvcormac/foo07)

* [Enron_Spam dataset](https://www.cs.cmu.edu/~enron/)

* [LingSpam dataset](https://www.aueb.gr/users/ion/data/lingspam_public.tar.gz)

After preparing the datasets, preprocess all of them by running the following scripts:
`Spam/dataloader06.py`, `Spam/dataloader07.py`, `Spam/dataloaderEnron.py`, `Spam/dataloaderLingSpam.py`

Next, set the training dataset and run the following scripts to train the models:

`Spam/SVM.py`  `Spam/Logistic.py` `Spam/PBO-train.py` `Spam/SQP_hinge.py``Spam/SQP_crossentropy.py`

Finally, to obtain results in Table 2, run the training script for the corresponding dataset: `Spam/show_results.py`

## Hyper-Representation

This repository is based on [this codebase](https://github.com/sowmaster/esjacobians).

1. **Linear hyper-representation**

&#x20;      To draw the results in Figure 2, execute: `HR/hyper.py`

2. **Deep hyper-representation**

   To draw the results in Figure 3, execute:&#x20;

   `Deep HR/AID-CG.py` `Deep HR/AID-FP.py` `Deep HR/PZOBO.py` `Deep HR/SiPBA.py` and `Deep HR/plot.py`

