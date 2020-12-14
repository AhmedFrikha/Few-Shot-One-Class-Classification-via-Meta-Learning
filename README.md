## Few-Shot One-Class Classification via Meta-Learning (AAAI 2021)

This code repository contains the implementations used for the experiments of the paper 
"Few-Shot One-Class Classification via Meta-Learning" accepted at AAAI 2021 (https://arxiv.org/abs/2007.04146).


Bibtex:
```
@article{frikha2020few,
  title={Few-shot one-class classification via meta-learning},
  author={Frikha, Ahmed and Krompa{\ss}, Denis and K{\"o}pken, Hans-Georg and Tresp, Volker},
  journal={arXiv preprint arXiv:2007.04146},
  year={2020}
}
```

The repository is organized as follows.

The meta-learning algorithms MetaOptNet, Meta-SGD (including their one-class versions) and One-Way Prototypical Networks have separate folders.

For the one-class and class-balanced versions of MAML, FOMAML and Reptile, the experiments of each dataset are in a separate folder. Each dataset folder contains two folders, one for the OCC baselines and one for the meta-learning algorithms, and a script to preprocess/generate the data, e.g. generate (meta-training) tasks.

The files containing the Sawtooth and Sine Synthetic Time-Series datasets proposed as benchmarks for few-shot one-class classification in the time-series domains can be found in the "Data" folders. 

In each folder there is an exemplary script to run the experiments (usually called run_example.sh). We provide some exemplary configuration files (.json). All hyper-parameters (e.g. size of the dapatation set K, CIR, etc...) can be easily changed from the configuration file. Feel free to do so to reproduce the results reported in the paper. The reported results were run with the seeds {1,2,3,4,5}.

Some modules of the pyMeta library (https://github.com/spiglerg/pyMeta) were used to conduct experiments on image datasets.
