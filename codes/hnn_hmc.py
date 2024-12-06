import tensorflow as tf
import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF
import tensorflow_probability as tfp
from nn_models import MLP
from hnn import HNN
from utils import leapfrog
from functions import functions
from get_args import get_args
import sys
import datetime
from pathlib import Path


class Logger:
    def __init__(self, log_dir="logs"):
        """
        Initialize the Logger instance.

        Args:
            log_dir (str): Directory to save log files. Defaults to "logs".
        """
        # Create the logging directory
        Path(log_dir).mkdir(parents=True, exist_ok=True)

        # Create a log file name using the current timestamp
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = Path(log_dir) / f"hnn_hmc_log_{current_time}.txt"

        # Open the log file for writing
        self.terminal = sys.stdout
        self.log = open(log_file, "w", buffering=1)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  # Ensure real-time writing

    def flush(self):
        """
        Flush the terminal and log file buffers.
        """
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


class HNNSampler:
    """
    A class for performing Hamiltonian Monte Carlo (HMC) sampling using a Hamiltonian Neural Network (HNN).
    """
    def __init__(self, args):
        """
        Initialize the HNNSampler.

        Args:
            args: Configuration arguments.

        Raises:
            ValueError: If any of the parameters are invalid.
        """
        if args.hmc_step_size <= 0:
            raise ValueError(f"Step size must be positive, got {args.hmc_step_size}")

        if args.trajectory_length <= 0:
            raise ValueError(f"Trajectory length must be positive, got {args.trajectory_length}")

        if args.input_dim <= 0:
            raise ValueError(f"Input dimension must be positive, got {args.input_dim}")

        if args.input_dim % 2 != 0:
            raise ValueError(f"Input dimension must be even (for position-momentum pairs), got {args.input_dim}")

        if args.latent_dim <= 0:
            raise ValueError(f"Latent dimension must be positive, got {args.latent_dim}")

        if args.hmc_samples <= 0:
            raise ValueError(f"Number of samples must be positive, got {args.hmc_samples}")

        if args.num_burnin < 0:
            raise ValueError(f"Number of burn-in samples cannot be negative, got {args.num_burnin}")

        if args.num_burnin >= args.hmc_samples:
            raise ValueError(f"Number of burn-in samples ({args.num_burnin}) must be less than "
                             f"total number of samples ({args.hmc_samples})")

        if args.num_chains <= 0:
            raise ValueError(f"Number of chains must be positive, got {args.num_chains}")
        # Sampling parameters
        self.num_chains = args.num_chains
        self.num_samples = args.hmc_samples
        self.trajectory_length = args.trajectory_length
        self.num_burnin = args.num_burnin
        self.step_size = args.hmc_step_size
        self.args = args
        self.trajectories = []  # List to store trajectories

        # Initialize model and state
        self.state_dim = args.input_dim // 2
        self.total_dim = args.input_dim
        self.latent_dim = args.latent_dim
        self.hidden_dim = args.hidden_dim
        self.hnn_model = self.build_model()

    def build_model(self):
        """
        Build and load the pretrained HNN model.

        Returns:
            HNN: The Hamiltonian Neural Network model.
        """
        nn_model = MLP(
            input_dim=self.total_dim,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            nonlinearity=self.args.nonlinearity
        )

        model = HNN(
            input_dim=self.args.input_dim,
            differentiable_model=nn_model
        )

        # load_weights
        # checkpoint_path = f"{self.args.save_dir}/{self.args.dist_name}"
        checkpoint_path = self.args.save_dir
        model.load_weights(checkpoint_path)
        return model

    def integrate_model(self, z0, t_span, n_steps):
        """
        Simulate Hamiltonian dynamics using the leapfrog integrator.

        Args:
            z0 (tf.Tensor): Initial state tensor with shape [batch_size, dim].
            t_span (list): Time range [t0, t1].
            n_steps (int): Number of integration steps.

        Returns:
            tuple: Time points and the trajectory array [n_steps+1, batch_size, dim].
        """

        t, z = leapfrog(self.hnn_model, z0, t_span, n_steps)
        return t, z

    def sample(self):
        """
        Perform HMC sampling.

        Returns:
            tuple: Samples tensor of shape [num_chains, num_samples, total_dim]
                   and acceptance tensor of shape [num_chains, num_samples].
        """
        # Initialize storage arrays
        samples = tf.zeros([self.num_chains, self.num_samples, self.total_dim])
        acceptance = tf.zeros([self.num_chains, self.num_samples])

        # Set integration parameters
        t_span = [0.0, self.trajectory_length]
        n_steps = self.trajectory_length * int(1 / self.step_size)

        # Perform sampling for each Markov chain
        for chain in range(self.num_chains):
            # Initialize the current state
            current_state = tf.zeros(self.total_dim, dtype=tf.float32)

            # Counters for burn-in and sampling acceptance
            burnin_accepted = 0
            sampling_accepted = 0

            # Perform sampling for each sample
            for sample in range(self.num_samples):
                # Determine the current phase (burn-in or sampling)
                phase = "Burn-in" if sample < self.num_burnin else "Sampling"

                # Sample momentum from a standard normal distribution
                momenta = tf.random.normal([self.state_dim], mean=0., stddev=1.)
                current_state = tf.concat([
                    current_state[:self.state_dim],  # Keep position coordinates
                    momenta  # New momentum
                ], axis=0)

                z0 = (tf.expand_dims(current_state, 0)
                      if len(current_state.shape) == 1 else current_state)

                # Integrate the trajectory using leapfrog
                _, trajectory = self.integrate_model(z0, t_span, n_steps)
                # Save trajectories for non-burn-in samples
                if sample >= self.num_burnin:
                    self.trajectories.append(trajectory.numpy())

                proposed_state = trajectory[-1, 0, :]

                # Compute the acceptance probability
                current_hamiltonian = functions(tf.expand_dims(current_state, 0), self.args)
                proposed_hamiltonian = functions(tf.expand_dims(proposed_state, 0), self.args)
                delta_H = current_hamiltonian - proposed_hamiltonian
                acceptance_prob = tf.minimum(1.0, tf.exp(tf.squeeze(delta_H)))

                # Perform Metropolis acceptance/rejection step
                is_accepted = tf.random.uniform([]) < acceptance_prob
                if is_accepted:
                    current_state = proposed_state
                    # Update acceptance counters
                    if sample < self.num_burnin:
                        burnin_accepted += 1
                    else:
                        sampling_accepted += 1

                    acceptance = tf.tensor_scatter_nd_update(
                        acceptance,
                        [[chain, sample]],
                        [1.0]
                    )


                    if sample < self.num_burnin:
                        current_accept_rate = burnin_accepted / (sample + 1)
                    else:
                        current_accept_rate = sampling_accepted / (sample - self.num_burnin + 1)

                    print(f"HNN: Chain {chain + 1}/{self.num_chains}, "
                          f"Sample {sample + 1}/{self.num_samples} ({phase}): "
                          f"ACCEPTED (prob: {acceptance_prob:.4f}, "
                          f"current rate: {current_accept_rate:.4f})")
                else:
                    print(f"Chain {chain + 1}/{self.num_chains}, "
                          f"Sample {sample + 1}/{self.num_samples} ({phase}): "
                          f"REJECTED (prob: {acceptance_prob:.4f})")

                print(f"    Current H: {tf.squeeze(current_hamiltonian):.4f}")
                print(f"    Proposed H: {tf.squeeze(proposed_hamiltonian):.4f}")
                print(f"    exp(Delta H): {tf.exp(tf.squeeze(delta_H)):.4f}")

                # Store the current state
                samples = tf.tensor_scatter_nd_update(
                    samples,
                    [[chain, sample, i] for i in range(self.total_dim)],
                    current_state
                )

                # Print acceptance statistics for the chain
            burnin_rate = burnin_accepted / self.num_burnin
            sampling_rate = sampling_accepted / (self.num_samples - self.num_burnin)
            print(f"\nChain {chain + 1} Statistics:")
            print(f"Burn-in phase acceptance rate: {burnin_rate:.4f}")
            print(f"Sampling phase acceptance rate: {sampling_rate:.4f}")

        return samples, acceptance[:, self.num_burnin:]  # Return non-burn-in samples' acceptance

    def compute_ess(self, samples):
        """
        Compute the Effective Sample Size (ESS) for position coordinates.

        Args:
            samples (tf.Tensor): Samples tensor of shape [num_chains, num_samples, total_dim].

        Returns:
            tf.Tensor: ESS tensor of shape [num_chains, state_dim].
        """
        ess = tf.zeros([self.num_chains, self.state_dim])

        for chain in range(self.num_chains):
            # Exclude burn-in samples and select position coordinates
            chain_samples = samples[chain, self.num_burnin:, :self.state_dim]
            # Compute ESS
            ess_values = tfp.mcmc.effective_sample_size(chain_samples)

            ess = tf.tensor_scatter_nd_update(
                ess,
                [[chain, i] for i in range(self.state_dim)],
                ess_values
            )

        return ess

    def get_trajectories(self):
        return np.array(self.trajectories)

def run_hnn_hmc(args):
    """
    Run HNN-HMC sampling.
    """
    sampler = HNNSampler(args)
    samples, acceptance = sampler.sample()
    ess = sampler.compute_ess(samples)


    print("\nOverall Sampling Statistics:")
    print(f"Average acceptance rate (excluding burn-in): {tf.reduce_mean(acceptance):.4f}")
    print(f"Average ESS: {tf.reduce_mean(ess):.4f}")

    return samples, acceptance, ess


if __name__ == "__main__":
    # Setup logger
    logger = Logger()
    sys.stdout = logger

    try:
        args = get_args()
        args.dist_name = "1D_Gauss_mix"
        args.input_dim = 2
        args.latent_dim = 100
        args.hmc_step_size = 0.05
        args.trajectory_length = 5
        args.hmc_samples = 100
        args.num_burnin = 10
        args.num_chains = 1

        samples, acceptance, ess = run_hnn_hmc(args)

        # save to csv
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        samples_path = f"hnn_hmc_samples_{timestamp}.csv"

        # Reshape and save samples
        samples_reshaped = samples.numpy().reshape(-1, samples.shape[-1])
        header = ','.join([f'q{i + 1},p{i + 1}' for i in range(samples_reshaped.shape[1] // 2)])
        np.savetxt(samples_path, samples_reshaped, delimiter=',',
                   header=header, comments='')

        print(f"\nSamples saved to: {samples_path}", file=logger)

    finally:
        # Ensure the log file is closed at the end
        logger.close()
        sys.stdout = logger.terminal  # Restore standard output