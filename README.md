# MRI_Diffusion_Patterns

Code for reproducing the results in the paper,

Optimizing sampling patterns for compressed sensing MRI with diffusion generative models,  
Sriram Ravula, Brett Levac, Yamin Arefeen, Ajil Jalal, Alexandros G Dimakis, Jonathan I Tamir,  
Arxiv preprint: https://arxiv.org/abs/2306.03284


### Submodule

This project contains an EDM submodule. To properly initialize it, run the following command after you clone this project: `git submodule update --init --recursive`.

### Running

Check the scripts folder for examples of how to train sampling patterns. Make sure to set the correct paths and variables for your runs, either in the hydra config files in the configs folder, or on the command line by overriding the variable values.

