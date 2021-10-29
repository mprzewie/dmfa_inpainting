# dmfa_inpainting

Source code for:
* [MisConv: Convolutional Neural Networks for Missing Data](https://arxiv.org/abs/2110.14010) (to be published at WACV 2022)
* [Estimating conditional density of missing values using deep Gaussian mixture model](https://arxiv.org/abs/2010.02183) (ICONIP 2020)

## Requirements

Python 3.8 or higher is required.
Models have been implemented with PyTorch.

To install the requirements, running:

```bash
pip install -r requirements.txt
```

should suffice.

## Running

To train the DMFA model, see the script:

```bash
python scripts/train_inpainter.py --h
````

To run classifier / WAE experiments, see the scripts:

```bash
python scripts/train_classifier_v2.py --h
python scripts/train_wae_v2.py --h
```
respectively.

Moreover, in the `scripts/` directory we provide the `*.sh` scripts which run the model trainings 
with the same parameters as used in the paper.

All experiments are runnable on a single Nvidia GPU.

### Inpainters used with classifiers and WAE
In order to run a classifier / WAE with DMFA, one must train the DMFA model first with the above script.

For some of the inpainters we compare our approach to, additional repositories must be cloned or installed:
* Partial Convolutions: https://github.com/NVIDIA/partialconv
* ACFlow: https://github.com/lupalab/ACFlow
* k-NN: either sklearn (runs very slowly) or [cuml](https://github.com/rapidsai/cuml) + [cudf](https://github.com/rapidsai/cudf)
* classical MFA: https://github.com/mareksmieja/gmm_missing

## DMFA Weights

We provide DMFA training results (among which are JSONs, weights and training arguments) [here](https://drive.google.com/drive/folders/1o_qgbJNfh8HLlQEq0CPmXXCvCwgeqcH4?usp=sharing).

We provide results for following models, trained on complete and incomplete data:

* MNIST - linear heads 
* SVHN - fully convolutional 
* CIFAR-10 - fully convolutional
* CelebA - fully convolutional, trained on 64x64 images

### Notebooks

There are several Jupyter Notebooks in the [notebooks](https://github.com/mprzewie/gmms_inpainting/tree/master/notebooks) directory. 
They were used for initial experiments with the DMFA models, as well as analysis of the results and calculating metrics reported in the paper.

The notebooks are not guaranteed to run 100% correctly due to the subsequent code refactor.

## Citation

If you find our work useful, please consider citing us!

```
@misc{przewięźlikowski2021misconv,
      title={MisConv: Convolutional Neural Networks for Missing Data}, 
      author={Marcin Przewięźlikowski and Marek Śmieja and Łukasz Struski and Jacek Tabor},
      year={2021},
      eprint={2110.14010},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
```
@article{Przewiezlikowski_2020,
   title={Estimating Conditional Density of Missing Values Using Deep Gaussian Mixture Model},
   ISBN={9783030638368},
   ISSN={1611-3349},
   url={http://dx.doi.org/10.1007/978-3-030-63836-8_19},
   DOI={10.1007/978-3-030-63836-8_19},
   journal={Lecture Notes in Computer Science},
   publisher={Springer International Publishing},
   author={Przewięźlikowski, Marcin and Śmieja, Marek and Struski, Łukasz},
   year={2020},
   pages={220–231}
}
```

