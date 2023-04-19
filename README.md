<h2 align='center'>pytorch-neuralssm-template</h2>
<h3 align='center'>A modular and approachable PyTorch-Lightning template for <br>quick-starting research with Neural State-Space Models (Neural SSMs)
</h3>

<a name="about"></a>
## About this Repository

<!-- CITATION -->
<a name="citation"></a>
## Citation
If you use portions of this repo in research development, please consider citing the following work:
```
@inproceedings{jiangsequentialLVM,
  title={Sequential Latent Variable Models for Few-Shot High-Dimensional Time-Series Forecasting},
  author={Jiang, Xiajun and Missel, Ryan and Li, Zhiyuan and Wang, Linwei},
  booktitle={The Eleventh International Conference on Learning Representations}
}
```

<a name="toc"></a>
## Table of Contents
- [About](#about)
- [Citation](#citation)
- [Table of Contents](#toc)
- [What are Neural SSMs?](#neuralSSMwhat)
- [Implementation](#implementation)
  - [Data](#data)
  - [Models](#models)
  - [Metrics](#metrics)
- [Contributions](#contributions)
- [References](#references)

<!-- Neural SSM INTRO -->
<a name="neuralSSMwhat"></a>
## What are Neural SSMs?
An extension of classic state-space models, <i>neural</i> state-space models - at their core - consist of a dynamic function of some latent states <b>z_k</b> and their emission to observations <b>x_k</b>, realized through the equations:
<a name="ssmEQ"></a>
<p align='center'><img src="https://user-images.githubusercontent.com/32918812/169743189-057f52a5-8a08-4616-9516-3c60aca86b28.png" alt="neural ssm equations" )/></p>
where <b>θ</b><sub>z</sub> represents the parameters of the latent dynamic function. The precise form of these functions can vary significantly - from deterministic or stochastic, linear or non-linear, and discrete or continuous.
<p> </p>
Due to their explicit differentiation of transition and emission and leveraging of structured equations, they have found success in learning interpretable latent dynamic spaces<sup>[1,2,3]</sup>, identifying physical systems from non-direct features<sup>[4,5,6]</sup>, and uses in counterfactual forecasting<sup>[7,8,14]</sup>.
<p> </p>
Given the fast pace of progress in latent dynamics modeling over recent years, many models have been presented under a variety of terminologies and proposed frameworks - examples being variational latent recurrent models<sup>[5,9,10,11,12,22]</sup>, deep state-space models<sup>[1,2,3,7,13,14]</sup>, and deterministic encoding-decoding models<sup>[4,15,16]</sup>. Despite differences in appearance, they all adhere to the same conceptual framework of latent variable modeling and state-space disentanglement. As such, they can be unified under the terminology of Neural SSMs and differentiated into the two base choices of probabilistic graphical models that they adhere to: <i>state estimation</i> (Fig 1A) and <i>system identification</i> (Fig. 1B). Our example in this repo focuses on the <i>system identification</i> setting (<u>B2</u> in Figure 1), in which a single initial state is used to propagate forward the entire sequence, but note that this repo is flexible and can be used in the other settings just as easily.

<a name="latentSchematic"></a>
<p align='center'><img src="https://user-images.githubusercontent.com/32918812/203471172-6dcbb898-d2fb-411f-b486-ed153b95bfc6.png" alt="latent variable schematic" /></p>
<p align='center'>Fig 1. Schematic of latent variable PGMs in Neural SSMS.</p>

<!-- IMPLEMENTATION -->
<a name="implementation"></a>
# Implementation

In this section, specifics on model implementation and the datasets/metrics used are detailed. Specific data generation details are available in the URLs provided for each dataset. The models and datasets used throughout this repo are solely grayscale physics datasets with underlying Hamiltonian laws. Extensions to color images and non-pixel-based tasks (or even graph-based data!) are easily done in this framework, as the only architecture change needed is the structure of the encoder and decoder networks as the state propagation happens solely in a latent space.

The project's folder structure is as follows:
<a name="folderStructure"></a>
```
  torchssm/
  │
  ├── main.py                       # Entry point for starting training or testing runs
  ├── README.md                     # What you're reading right now :^)
  ├── requirements.txt              # Anaconda requirements file to enable easy setup
  |
  ├── data/
  |   ├── <dataset_type>            # Name of the stored dynamics dataset
  |   ├── generate_bouncingball.py  # Dataset generation script for Bouncing Ball
  ├── experiments/
  |   └── <model_name>              # Name of the dynamics model run
  |       └── <experiment_type>     # Given name for the ran experiment
  |           └── <version_x>/      # Each experiment type has its sequential lightning logs saved
  ├── lightning_logs/
  |   ├── version_0/                # Placeholder lightning log folder
  |   └── ...                       # Subsequent runs
  ├── models/
  │   ├── CommonDynamics.py         # Abstract PyTorch-Lightning Module to handle train/test loops
  │   ├── CommonVAE.py              # Shared encoder/decoder Modules for the VAE portion
  │   ├── dynamics_models/ 
  │       └── ...                   # Specific implementations of different latent dynamics functions
  ├── utils/
  │   ├── dataloader.py             # WebDataset class to return the dataloaders used in train/val/testing
  │   ├── layers.py                 # PyTorch Modules that represent general network layers
  │   ├── metrics.py                # Metric functions for evaluation
  │   ├── plotting.py               # Plotting functions for visualizatin
  |   └── utils.py                  # General utility functions (e.g. argparsing, experiment number tracking, etc)
  └──
```

<!-- DATA -->
<a name="data"></a>
## Example Data

Example datasets to run on the base neural ODE are available for download <a href="">here</a> here on Google Drive, in which they already come in their .npz forms. Additionally, we provide a dataloader and generation scripts for the standard latent dynamics 
dataset of bouncing balls, modified from the implementation in 
<a href="https://github.com/simonkamronn/kvae/tree/master/kvae/datasets">KVAE</a>. It consists of a ball or multiple 
balls moving within a bounding box while being affected by potential external effects, e.g. gravitational 
forces, pong, and interventions. The starting position, angle, and velocity of the ball(s) are sampled uniformly between a set range. It is generated with the 
<a href="https://github.com/viblo/pymunk">PyMunk</a> and <a href="https://www.pygame.org/news">PyGame</a> libraries. 
In this repository, we consider a simple set of one gravitational force. We generate <code>20000</code> training and 
<code>5000</code> testing trajectories, sampled at <code>Δt = 0.1</code> intervals.

<p align='center'><img src="https://user-images.githubusercontent.com/32918812/171948373-ad692ecc-bfac-49dd-86c4-137a2a5e4b73.gif" alt="bouncing ball examples" /></p>
<p align='center'>Fig 2. Single Gravity Bouncing  Ball Example.</p>

<!-- MODELS -->
<a name="models"></a>    
## Models

Here, details on how the model implementation is structured and running experiments locally are given. As well, 
an overview of the abstract class implementation for a general Neural SSM and its types are explained.

### Implementation Structure
Provided within this repository is a PyTorch class structure in which an abstract PyTorch-Lightning Module is shared 
across all the given models, from which the specific VAE and dynamics functions inherit and override the relevant 
forward functions for training and evaluation. As the implementation is provided in 
<a href="https://pytorch-lightning.readthedocs.io/en/latest/">PyTorch-Lightning</a>, an optimization and boilerplate 
library for PyTorch, it is recommended to be familiar at face-level.

<p> </p>
For every model run, a new <code>lightning_logs/</code> version folder is created as well as a new experiment version 
under `experiments` related to the passed in naming arguments. During training and validation sequences, all of the metrics below are automatically tracked and saved into a Tensorboard instance which can be used to compare different model runs following. Inside each dynamics model file, overwriting the <code>model_specific_plotting()</code> function allows users to specify additional per-model figures to be generated every validation step. Restarting training from a checkpoint or loading in a model for testing is done currently by specifying the <code>ckpt_path</code> to the base experiment folder and the <code>checkpt</code> filename, as well as setting the <code>resume</code> flag to True.

<p> </p>
The baseline dynamics model provided is a simple Neural Ordinary Differential Equation,  e.g. 
<code>z<sup>'</sup><sub>t</sub> = f<sub>θ</sub>(z<sub>s</sub>)</code>, using the <a href="https://github.com/rtqichen/torchdiffeq">torchdiffeq</a> library for its <code>odeint</code> function.

### Configuration Files
Hyperparameters are saved and loaded in through JSON configuration files, which specify everything from the dynamics function used to which metrics are being tracked. These configuration files are stored in <code>configs/</code> and two example files are provided - one with documentation and the other specifically for the Neural ODE. A copy of this exact config is stored in every experiment folder created.

<p align='center'><img src="https://user-images.githubusercontent.com/32918812/233156082-53d133e7-fa11-4757-baed-ea5e36279be9.png" alt="config example" /></p>
<p align='center'>Fig 3. A portion of the example configuration file.</p>

<!-- METRICS -->
<a name="metrics"></a>
## Metrics
To specify additional metrics that are tracked automatically, one just needs to add another function in <code>metrics.py</code> with the given input parameter scheme as the others and have it return two values - normally the mean and std of the metric. We provide at base MSE and MAPE metric functions.

<p align='center'><img src="https://user-images.githubusercontent.com/32918812/233155855-8ca2b591-2f90-4558-905f-14fc83b4c09a.png" alt="metric example" /></p>
<p align='center'>Fig 4. Example function of a metric.</p>

<!-- CONTRIBUTIONS -->
<a name="contributions"></a>
## Contributions
Contributions are welcome and encouraged! If you have an implementation of a latent dynamics function you think 
would be relevant and add to the conversation, feel free to submit an Issue or PR and we can discuss its 
incorporation. Similarly, if you feel an area of the README is lacking or contains errors, please put up a 
README editing PR with your suggested updates. Even tackling items on the To-Do would be massively helpful!

<!-- REFERENCES  -->
<a name="references"></a>
## References
1. Maximilian Karl, Maximilian Soelch, Justin Bayer, and Patrick van der Smagt. Deep variational bayes filters: Unsupervised learning of state space models from raw data. In International Conference on Learning Representations, 2017.
2. Marco Fraccaro, Simon Kamronn, Ulrich Paquetz, and OleWinthery. A disentangled recognition and nonlinear dynamics model for unsupervised learning. In Advances in Neural Information Processing Systems, 2017.
3. Alexej Klushyn, Richard Kurle, Maximilian Soelch, Botond Cseke, and Patrick van der Smagt. Latent matters: Learning deep state-space models. Advances in Neural Information Processing Systems, 34, 2021.
4. Aleksandar Botev, Andrew Jaegle, Peter Wirnsberger, Daniel Hennes, and Irina Higgins. Which priors matter? benchmarking models for learning latent dynamics. In Advances in Neural Information Processing Systems, 2021.
5. C. Yildiz, M. Heinonen, and H. Lahdesmaki. ODE2VAE: Deep generative second order odes with bayesian neural networks. In Neural Information Processing Systems, 2020.
6. Batuhan Koyuncu. Analysis of ode2vae with examples. arXiv preprint arXiv:2108.04899, 2021.
7. Rahul G. Krishnan, Uri Shalit, and David Sontag. Structured inference networks for nonlinear state space models. In Association for the Advancement of Artificial Intelligence, 2017.
8. Daehoon Gwak, Gyuhyeon Sim, Michael Poli, Stefano Massaroli, Jaegul Choo, and Edward Choi. Neural ordinary differential equations for intervention modeling. arXiv preprint arXiv:2010.08304, 2020.
9. Junyoung Chung, Kyle Kastner, Laurent Dinh, Kratarth Goel, Aaron Courville, and Yoshua Bengio. A recurrent latent variable model for sequential data. In Advances in Neural Information Processing Systems, 2015.
10. Yulia Rubanova, Ricky T. Q. Chen, and David Duvenaud. Latent odes for irregularly-sampled time series. In Neural Information Processing Systems, 2019.
11. Tsuyoshi Ishizone, Tomoyuki Higuchi, and Kazuyuki Nakamura. Ensemble kalman variational objectives: Nonlinear latent trajectory inference with a hybrid of variational inference and ensemble kalman filter. arXiv preprint arXiv:2010.08729, 2020.
12. Justin Bayer, Maximilian Soelch, Atanas Mirchev, Baris Kayalibay, and Patrick van der Smagt. Mind the gap when conditioning amortised inference in sequential latent-variable models. arXiv preprint arXiv:2101.07046, 2021.
13. Ðor ̄de Miladinovi ́c, Muhammad Waleed Gondal, Bernhard Schölkopf, Joachim M Buhmann, and Stefan Bauer. Disentangled state space representations. arXiv preprint arXiv:1906.03255, 2019.
14. Zeshan Hussain, Rahul G. Krishnan, and David Sontag. Neural pharmacodynamic state space modeling, 2021.
15. Francesco Paolo Casale, Adrian Dalca, Luca Saglietti, Jennifer Listgarten, and Nicolo Fusi.Gaussian process prior variational autoencoders. Advances in neural information processing systems, 31, 2018.
16. Yingzhen Li and Stephan Mandt. Disentangled sequential autoencoder. arXiv preprint arXiv:1803.02991, 2018.
17. Patrick Kidger, James Morrill, James Foster, and Terry Lyons. Neural controlled differential equations for irregular time series. Advances in Neural Information Processing Systems, 33:6696-6707, 2020.
18. Edward De Brouwer, Jaak Simm, Adam Arany, and Yves Moreau. Gru-ode-bayes: Continuous modeling of sporadically-observed time series. Advances in neural information processing systems, 32, 2019.
19. Ruben Villegas, Jimei Yang, Yuliang Zou, Sungryull Sohn, Xunyu Lin, and Honglak Lee. Learning to generate long-term future via hierarchical prediction. In international conference on machine learning, pages 3560–3569. PMLR, 2017
20. Xiajun Jiang, Ryan Missel, Maryam Toloubidokhti, Zhiyuan Li, Omar Gharbia, John L Sapp, and Linwei Wang. Label-free physics-informed image sequence reconstruction with disentangled spatial-temporal modeling. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 361–371. Springer, 2021.
21. Ricky TQ Chen, Yulia Rubanova, Jesse Bettencourt, and David K Duvenaud. Neural ordinary differential equations. Advances in neural information processing systems, 31, 2018.
22. Junyoung Chung, Kyle Kastner, Laurent Dinh, Kratarth Goel, Aaron C Courville, and Yoshua Bengio. A recurrent latent variable model for sequential data. Advances in neural information processing systems, 28, 2015.
