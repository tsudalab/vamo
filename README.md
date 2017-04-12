# vamo
Variational Autoencoder for Materials Optimization

# Required Packages ############################
  * python >= 3.4.x
  * numpy >= 1.11.x
  * scipy >= 0.18.x
  * pandas >= 1.11.x
  * tensorflow >= 1.x
  * keras >= 2.x
  * combo development version

## Install Guide ################################
  * Install COMBO with the develop version
    1. Download or clone the github repository, e.g.
      > git clone https://github.com/tsudalab/combo.git
      > git checkout develop

    2. Run setup.py install
      > cd combo
      > python setup.py install


  * Install TensorFlow
    * Install the tensorflow suitable for your machine according to
      the install guide shown in https://www.tensorflow.org/install/

  * Install Keras
    > pip3 install keras

## Usage ################################
  1. run encoder_Si-Ge.ipynb in order to learn the variational encoder
  2. run vamo_Si-Ge.ipynb for searching the structure with the maximum templature
