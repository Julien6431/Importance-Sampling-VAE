# Importance-Sampling-VAE

Software and data related to the paper "Variational autoencoder with weighted samples for high-dimensional non-parametric adaptive importance sampling".

## Repository structure

The folder "src" contains our suggested impletementation of the variational autoencoder with the proposed pre-training procedure.

The folder "cross-entropy" contains the implementation of the cross-entropy algorithm to estimate the failure probability for two parametric families of distributions: VAE and vMNFM.

The folder "numerical_tests" contains the implementation of the test cases from the article as well as the corresponding data. These files illustrate how to use the algorithms. 

Feel free to test these algorithms on your own test cases and/or to play with the proposed ones.

## Quick start
### Installation

In order to install the current project locally, please execute the following lines in a terminal:

```
git clone https://github.com/Julien6431/Importance-Sampling-VAE.git
cd Importance-Sampling-VAE
```


### Install requirements

In order to install the required modules, please run the following line in a terminal or in the console:

```
pip install -r requirements.txt
```

### Code execution

Remember to execute all files in a terminal or in a console from the main folder. 
