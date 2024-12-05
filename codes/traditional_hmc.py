import tensorflow as tf
import numpy as np
from utils import traditional_leapfrog
from functions import functions
from get_args import get_args
from logger_utils import Logger
import sys
import datetime


class TraditionalHMC:
    """
    Implements the Traditional Hamiltonian Monte Carlo (HMC) sampling method.
    """
    def __init__(self, args, logger=None):
        """
        Initialize the TraditionalHMC instance with the given parameters.

        Args:
            args: Configuration arguments containing HMC parameters.
            logger: Logger instance for logging outputs
        """
        self.logger = logger or sys.stdout
        self.num_chains = args.num_chains
        self.num_samples = args.hmc_samples
        self.num_burnin = args.num_burnin
        self.state_dim = args.input_dim // 2
        self.total_dim = args.input_dim
        self.step_size = args.hmc_step_size
        self.trajectory_length = args.trajectory_length
        self.trajectories = []
        self.args = args
        print(f"\nInitializing Traditional HMC with parameters:", file=self.logger)
        print(f"  Step size: {self.step_size}", file=self.logger)
        print(f"  Trajectory length: {self.trajectory_length}", file=self.logger)
        print(f"  Number of samples: {self.num_samples}", file=self.logger)
        print(f"  Burn-in period: {self.num_burnin}", file=self.logger)
        print(f"  Number of chains: {self.num_chains}", file=self.logger)
        print(f"  State dimension: {self.state_dim}", file=self.logger)

    def sample(self):
        """
        Perform HMC sampling.

        Returns:
            tuple: Samples tensor [num_chains, num_samples, total_dim]
                   and acceptance tensor [num_chains, num_samples].
        """
        # initialize storage
        samples = tf.zeros([self.num_chains, self.num_samples, self.total_dim])
        acceptance = tf.zeros([self.num_chains, self.num_samples])

        # Perform sampling for each chain
        for chain in range(self.num_chains):
            # Initialize the current state
            current_state = tf.zeros(self.total_dim, dtype=tf.float32)

            # Counters for burn-in and sampling acceptance
            burnin_accepted = 0
            sampling_accepted = 0

            # Perform sampling for each sample
            for sample in range(self.num_samples):
                # Determine the current phase
                phase = "Burn-in" if sample < self.num_burnin else "Sampling"

                # Sample momentum from a standard normal distribution
                momenta = tf.random.normal([self.state_dim], mean=0., stddev=1.)
                current_state = tf.concat([
                    current_state[:self.state_dim],  # 保持位置坐标
                    momenta  # 新的动量
                ], axis=0)

                # prepare data for leapfrog
                z0 = tf.expand_dims(current_state, 0)
                t_span = [0.0, self.trajectory_length]
                n_steps = int(self.trajectory_length / self.step_size)

                # leapfrog
                trajectory, _ = traditional_leapfrog(functions, z0, t_span, n_steps, self.args)
                if sample >= self.num_burnin:  # keep non-burnin sample
                    self.trajectories.append(trajectory.numpy())

                proposed_state = trajectory[-1, 0, :]

                # Compute acceptance probability
                current_H = functions(tf.expand_dims(current_state, 0), self.args)
                proposed_H = functions(tf.expand_dims(proposed_state, 0), self.args)
                delta_H = current_H - proposed_H
                acceptance_prob = tf.minimum(1.0, tf.exp(tf.squeeze(delta_H)))

                # Metropolis acceptance/rejection step
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

                    print(f"Traditional: Chain {chain + 1}/{self.num_chains}, "
                          f"Sample {sample + 1}/{self.num_samples} ({phase}): "
                          f"ACCEPTED (prob: {acceptance_prob:.4f}, "
                          f"current rate: {current_accept_rate:.4f})")
                else:
                    print(f"Chain {chain + 1}/{self.num_chains}, "
                          f"Sample {sample + 1}/{self.num_samples} ({phase}): "
                          f"REJECTED (prob: {acceptance_prob:.4f})")

                print(f"    Current H: {tf.squeeze(current_H):.4f}")
                print(f"    Proposed H: {tf.squeeze(proposed_H):.4f}")
                print(f"    exp(Delta H): {tf.exp(tf.squeeze(delta_H)):.4f}")

                samples = tf.tensor_scatter_nd_update(
                    samples,
                    [[chain, sample, i] for i in range(self.total_dim)],
                    current_state
                )

            burnin_rate = burnin_accepted / self.num_burnin
            sampling_rate = sampling_accepted / (self.num_samples - self.num_burnin)
            print(f"\nChain {chain + 1} Statistics:")
            print(f"Burn-in phase acceptance rate: {burnin_rate:.4f}")
            print(f"Sampling phase acceptance rate: {sampling_rate:.4f}")

        return samples, acceptance[:, self.num_burnin:]

    def get_trajectories(self):
        """
        Retrieve the saved trajectories.

        Returns:
            np.ndarray: Array of saved trajectories.
        """
        return np.array(self.trajectories)


if __name__ == "__main__":
    # logger
    logger = Logger()
    sys.stdout = logger

    try:
        # test
        args = get_args()
        args.dist_name = "1D_Gauss_mix"
        args.input_dim = 2
        args.hmc_step_size = 0.05
        args.trajectory_length = 5
        args.hmc_samples = 100
        args.num_burnin = 10
        args.num_chains = 1

        hmc = TraditionalHMC(args, logger)
        samples, acceptance = hmc.sample()

        print(f"\nTest completed", file=logger)
        print(f"Final acceptance rate: {tf.reduce_mean(acceptance):.4f}", file=logger)
        print(f"Samples shape: {samples.shape}", file=logger)

        # save to csv
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        samples_path = f"traditional_hmc_samples_{timestamp}.csv"

        # reshape and save
        samples_reshaped = samples.numpy().reshape(-1, samples.shape[-1])
        header = ','.join([f'q{i + 1},p{i + 1}' for i in range(samples_reshaped.shape[1] // 2)])
        np.savetxt(samples_path, samples_reshaped, delimiter=',',
                   header=header, comments='')

        print(f"\nSamples saved to: {samples_path}", file=logger)

    finally:
        logger.close()
        sys.stdout = logger.terminal
