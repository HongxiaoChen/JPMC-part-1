import tensorflow as tf
import numpy as np
import sys
from hnn_nuts_olm import nuts_hnn_sample, get_model
from traditional_hmc import TraditionalHMC
from hnn_hmc import HNNSampler
from get_args import get_args
import tensorflow_probability as tfp


class Logger:
    """
    Custom logger that writes outputs to both the terminal and a log file.
    """
    def __init__(self, filename="table1_results.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def fileno(self):
        return self.terminal.fileno()

def compute_ess(samples):
    """
    Compute the Effective Sample Size (ESS) for position coordinates.

    Args:
        samples (tf.Tensor): Samples tensor of shape [num_chains, num_samples, dim].

    Returns:
        tf.Tensor: The ESS of position coordinates.
    """
    # samples shape: [1, num_samples, 2]

    # Extract position coordinates (first dimension of the state)
    position_samples = samples[:, :, 0]  # Shape: [num_chains, num_samples]

    # Remove the chain dimension (only one chain is present in this case)
    position_samples = tf.squeeze(position_samples, axis=0)  # Shape: [num_samples]

    # Compute ESS
    ess = tfp.mcmc.effective_sample_size(position_samples)

    return ess

def run_experiment():
    """
    Perform the experiment to replicate Table 1 results from the paper.

    Returns:
        dict: A dictionary containing results for each method (samples, ESS, acceptance, gradients).
    """
    # Set experimental parameters
    args = get_args()
    args.dist_name = '1D_Gauss_mix'
    args.save_dir = 'files/1D_Gauss_mix'
    args.input_dim = 2
    args.latent_dim = 100
    args.total_samples = 50
    args.burn_in = 10
    args.nuts_step_size = 0.05
    args.hnn_error_threshold = 10.0
    args.num_chains = 1
    args.hmc_samples = 50
    args.num_burnin = 10
    args.hmc_step_size = 0.05
    args.trajectory_length = 5

    print("\n=== Replicating Table 1 Results ===")
    print("Distribution: 1D Gaussian Mixture")
    print(f"Total samples: {args.total_samples}")
    print(f"Burn-in samples: {args.burn_in}")

    results = {}

    # 1. Traditional NUTS
    print("\n1. Running Traditional NUTS...")
    model_trad = get_model(args)
    samples_trad, acceptance_trad, errors_trad, trad_nuts_grads = nuts_hnn_sample(
        model_trad, args,
        traditional_only=True,
        logger=sys.stdout
    )
    results['Traditional NUTS'] = {
        'samples': samples_trad,
        'ess': compute_ess(samples_trad),
        'acceptance': acceptance_trad,
        'grads': trad_nuts_grads
    }
    # 2. L-HNN NUTS
    print("\n2. Running L-HNN NUTS...")
    model_lhnn = get_model(args)
    samples_lhnn, acceptance_lhnn, errors_lhnn, lhnn_nuts_grads = nuts_hnn_sample(
        model_lhnn, args,
        traditional_only=False,
        logger=sys.stdout
    )
    results['L-HNN NUTS'] = {
        'samples': samples_lhnn,
        'ess': compute_ess(samples_lhnn),
        'acceptance': acceptance_lhnn,
        'grads': lhnn_nuts_grads
    }
    # 3. Traditional HMC
    print("\n3. Running Traditional HMC...")
    trad_hmc = TraditionalHMC(args)
    samples_trad_hmc, acceptance_trad_hmc = trad_hmc.sample()
    steps_per_sample = int(args.trajectory_length / args.hmc_step_size)
    trad_hmc_grads = steps_per_sample * 2 * args.hmc_samples  # Each leapfrog step requires 2 gradient evaluations
    results['Traditional HMC'] = {
        'samples': samples_trad_hmc,
        'ess': compute_ess(samples_trad_hmc),
        'acceptance': acceptance_trad_hmc,
        'grads': trad_hmc_grads
    }

    # 4. HNN HMC
    print("\n4. Running HNN HMC...")
    hnn_sampler = HNNSampler(args)
    samples_hnn_hmc, acceptance_hnn_hmc = hnn_sampler.sample()
    training_grads = 8000  # Number of gradient evaluations during HNN training
    results['HNN HMC'] = {
        'samples': samples_hnn_hmc,
        'ess': compute_ess(samples_hnn_hmc),
        'acceptance': acceptance_hnn_hmc,
        'grads': training_grads
    }

    # Print results table
    print("\n=== Results ===")
    print("Method            ESS     Gradients   ESS/grad")
    print("-" * 50)

    # Traditional HMC
    ess = results['Traditional HMC']['ess'].numpy()
    grads = results['Traditional HMC']['grads']
    print(f"Traditional HMC: {ess:.2f}  {grads}  {ess / grads:.5f}")

    # HNN HMC
    ess = results['HNN HMC']['ess'].numpy()
    grads = results['HNN HMC']['grads']
    print(f"HNN HMC:        {ess:.2f}  {grads}  {ess / grads:.5f}")

    # Traditional NUTS
    ess = results['Traditional NUTS']['ess'].numpy()
    grads = results['Traditional NUTS']['grads']
    print(f"Traditional NUTS:{ess:.2f}  {grads}  {ess / grads:.5f}")

    # L-HNN NUTS
    ess = results['L-HNN NUTS']['ess'].numpy()
    grads = training_grads + results['L-HNN NUTS']['grads']
    print(f"L-HNN NUTS:     {ess:.2f}  {grads}  {ess / grads:.5f}")

    return results


if __name__ == "__main__":

    sys.stdout = Logger()


    try:
        results = run_experiment()
    finally:
        sys.stdout = sys.stdout.terminal