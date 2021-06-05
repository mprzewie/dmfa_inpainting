# dmfa_inpainting

Source code for [Estimating conditional density of missing values using deep Gaussian mixture model](https://arxiv.org/abs/2010.02183).

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
```

## Weights

We provide DMFA training results (among which are JSONs, weights and training arguments) [here](https://drive.google.com/drive/folders/1o_qgbJNfh8HLlQEq0CPmXXCvCwgeqcH4?usp=sharing).

We provide results for following models, trained on complete and incomplete data:

* MNIST - linear heads 
* SVHN - fully convolutional 
* CIFAR-10 - fully convolutional
* CelebA - fully convolutional, trained on 64x64 images

### Classical MFA baseline

The implementation of classical MFA baseline we compare ourselves to in the paper can be found [here](https://github.com/mareksmieja/gmm_missing).

### Notebooks

There are several Jupyter Notebooks in the [notebooks](https://github.com/mprzewie/gmms_inpainting/tree/master/notebooks) directory. 
They were used for initial experiments with the DMFA models, as well as analysis of the results and calculating metrics reported in the paper.

The notebooks are not guaranteed to run 100% correctly due to the subsequent code refactor.

## Citation

If you find our work useful, please cite us!
 
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

