# JPMC ML

This repository contains code samples for the **2025 Machine Learning Center of Excellence Summer Associate** projects. Specifically, it replicates the results from the article *"Efficient Bayesian inference with latent Hamiltonian neural networks in No-U-Turn Sampling"* by **Somayajulu L.N. Dhulipala, Yifeng Che, and Michael D. Shields**.

---

## Environment

The code is written in **Python 3.8** and requires the following libraries:

- **TensorFlow 2.10**
- **TensorFlow Probability 0.18.0**

---

## Replication Materials

The main code files are stored in the **`codes/`** directory under the root directory. Below is a description of the key files in the repository:

### Core Files
- **`functions.py`**: Contains Hamiltonian functions used in the study.
- **`utils.py`**: Implements leapfrog algorithms and other frequently used utility functions.
- **`get_args.py`**: Stores the parameters used in the replication project.
- **`nn_models.py`**: Generates Multilayer Perceptrons (MLPs).
- **`hnn.py`**: Packages the MLPs from `nn_models.py` and generates Latent Hamiltonian Neural Networks (L-HNNs).

### Training and Sampling
- **`train_hnn.py`**: Script to train the models.
- **`hnn_hmc.py`**: Performs L-HNN Hamiltonian Monte Carlo (HMC).
- **`traditional_hmc.py`**: Implements traditional HMC.
- **`hnn_nuts_test.py`**: Performs both L-HNN No-U-Turn Sampling (NUTS) and traditional NUTS.

### Scripts for Replication
Run the following scripts to replicate the respective results presented in Dhulipala (2023):

- **`Figure2.py`**
- **`Figure3.py`**
- **`Figure4_5_6.py`**
- **`Table1.py`**
- **`Figure7_table2.py`**
- **`Figure8_table3.py`**
- **`Figure9_table3.py`**
- **`Figure10_table3.py`**
- **`Figure11_table4.py`**
- **`Figure12_table4.py`**

### Other Folders
- **`files/`**: Stores weights of networks and generated data.
- **`logs/`**: Contains logs generated during execution.
- **`figures/`**: Saves the generated figures.

---

## Testing

Unit tests are written in **`unittest`** and stored in the **`tests/`** folder under the root directory. Below is a detailed breakdown of the test files and their purposes:

- **`test_utils.py`**: Tests utility functions in `utils.py`.
- **`test_hnn_utils.py`**: Tests utility functions from `utils.py` and their integration with the Hamiltonian Neural Network (HNN) framework.
- **`test_data.py`**: Tests data generation functions from the data module, specifically `get_trajectory` and `get_dataset`.
- **`test_functions.py`**: Tests mathematical operations implemented in `functions.py`.
- **`test_traditional_hmc.py`**: Tests the implementation of the `TraditionalHMC` class for performing Hamiltonian Monte Carlo (HMC) sampling.
- **`test_hnn_hmc.py`**: Tests the implementation of the `HNNSampler` class, which combines Hamiltonian Neural Networks (HNN) with Hamiltonian Monte Carlo (HMC) sampling.
- **`test_hnn_nuts.py`**: Tests the No-U-Turn Sampler (NUTS) implementation integrated with Hamiltonian Neural Networks (HNN).

---

## How to Run

1. Set up the environment using Python 3.8 and install the required libraries.
2. Navigate to the **`codes/`** folder to access the core files and scripts.
3. Use the replication scripts to reproduce the results, or explore the provided modules for further experimentation.
4. Run the unit tests in the **`tests/`** folder to validate the functionality of the codebase.