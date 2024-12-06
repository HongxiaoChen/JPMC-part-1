import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from hnn_hmc import HNNSampler
from traditional_hmc import TraditionalHMC
from get_args import get_args
from logger_utils import Logger
import sys
import datetime
from utils import traditional_leapfrog
from functions import functions
import copy
from pathlib import Path


def plot_figure_2(args, logger):
    """
    Generate Figure 2 with three subplots: phase space, HNN-HMC position histogram,
    and Traditional HMC position histogram.

    Parameters
    ----------
    args : argparse.Namespace
        The command-line arguments containing model and sampling configurations.
    logger : Logger or file-like object
        Logger for printing and recording output messages.

    Returns
    -------
    tuple
        A tuple containing the matplotlib figure object and axes for the three subplots.
    """
    print("\nGenerating Figure 2...", file=logger)

    # Create a 1x3 figure layout
    fig = plt.figure(figsize=(15, 5))

    # ** Subplot 1: Phase Space (q-p plane) **
    print("\nPlotting phase space...", file=logger)
    ax1 = fig.add_subplot(131)

    # Modify parameters to generate longer trajectories
    args_traj = copy.deepcopy(args)
    args_traj.trajectory_length = 5  # Trajectory time
    args_traj.hmc_step_size = 0.05  # Step size
    args_traj.hmc_samples = 1  # Only one sample needed
    args_traj.num_burnin = 0  # No burn-in required
    args_traj.save_dir = 'files/1D_Gauss_mix'
    # Define several fixed initial momentum values
    initial_momenta = [
        tf.constant([1.0], dtype=tf.float32),
        tf.constant([1.5], dtype=tf.float32),
        tf.constant([2.0], dtype=tf.float32),
        tf.constant([2.5], dtype=tf.float32),
        tf.constant([3.0], dtype=tf.float32)
    ]

    # Generate and plot trajectories for each initial momentum
    for p0 in initial_momenta:
        # Traditional HMC trajectory
        q0 = tf.zeros([1], dtype=tf.float32)  # Fix initial position to 0
        initial_state = tf.concat([q0, p0], axis=0)
        z0 = tf.expand_dims(initial_state, 0)

        # Generate trajectory
        t_span = [0.0, args_traj.trajectory_length]
        n_steps = int(args_traj.trajectory_length / args_traj.hmc_step_size)

        # Traditional HMC trajectory
        trajectory, _ = traditional_leapfrog(functions, z0, t_span, n_steps, args)
        q_trad = trajectory[:, 0, 0]
        p_trad = trajectory[:, 0, 1]
        ax1.plot(q_trad, p_trad, 'b-', alpha=0.7, linewidth=1.0)

        # HNN-HMC trajectory
        hnn_sampler = HNNSampler(args_traj)
        _, trajectory = hnn_sampler.integrate_model(z0, t_span, n_steps)
        q_hnn = trajectory[:, 0, 0]
        p_hnn = trajectory[:, 0, 1]
        ax1.plot(q_hnn, p_hnn, 'r--', alpha=0.7, linewidth=1.0)

    # Add legend and labels to Subplot 1
    ax1.plot([], [], 'b-', label='Traditional HMC', alpha=0.7)
    ax1.plot([], [], 'r--', label='L-HNN in HMC', alpha=0.7)
    ax1.set_xlabel('Position (q)')
    ax1.set_ylabel('Momentum (p)')
    ax1.set_title('(a) Phase Space')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([-2.5, 2.5])
    ax1.set_ylim([-4, 4])

    # ** Run samplers to get histogram data **
    print("\nRunning samplers for histograms...", file=logger)
    args_sample = copy.deepcopy(args)
    args_sample.hmc_samples = 50  # Number of samples
    args_sample.num_burnin = 10  # Number of burn-in samples
    args_sample.trajectory_length = 5  # Trajectory length for sampling
    args_sample.hmc_step_size = 0.05  # Step size

    # Run samplers
    trad_hmc = TraditionalHMC(args_sample, logger)
    trad_samples, _ = trad_hmc.sample()
    hnn_sampler = HNNSampler(args_sample)
    hnn_samples, _ = hnn_sampler.sample()

    # Extract position coordinates (excluding burn-in)
    trad_q = trad_samples[0, args_sample.num_burnin:, 0]
    hnn_q = hnn_samples[0, args_sample.num_burnin:, 0]

    # ** Subplot 2: HNN-HMC Position Histogram **
    print("Plotting HNN-HMC position histogram...", file=logger)
    ax2 = fig.add_subplot(132)
    ax2.hist(hnn_q, bins=50, density=True, color='red', alpha=0.7,
             label='L-HNN in HMC')
    ax2.set_xlabel('Position (q)')
    ax2.set_ylabel('Density')
    ax2.set_title('(b) HNN-HMC Position Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([-2, 2])

    # ** Subplot 3: Traditional HMC Position Histogram **
    print("Plotting Traditional HMC position histogram...", file=logger)
    ax3 = fig.add_subplot(133)
    ax3.hist(trad_q, bins=50, density=True, color='blue', alpha=0.7,
             label='Traditional HMC')
    ax3.set_xlabel('Position (q)')
    ax3.set_ylabel('Density')
    ax3.set_title('(c) Traditional HMC Position Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([-2, 2])

    # Adjust layout
    plt.tight_layout()

    # ** Save the figure **
    # Check and create the figures directory
    figures_dir = Path("figures")
    if not figures_dir.exists():
        figures_dir.mkdir(parents=True)
        print(f"\nCreated directory: {figures_dir}", file=logger)

    # Save the figure with a timestamped filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = figures_dir / f'figure_2_reproduction_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved as {filename}", file=logger)

    return fig, (ax1, ax2, ax3)


if __name__ == "__main__":
    import os

    # Fix for possible library conflict
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    logger = Logger()
    sys.stdout = logger

    try:
        # Get default arguments
        args = get_args()

        # Configure necessary parameters
        args.dist_name = "1D_Gauss_mix"
        args.save_dir = 'files/1D_Gauss_mix'
        args.input_dim = 2  # 1D position + 1D momentum
        args.hidden_dim = 100
        args.latent_dim = 100
        args.nonlinearity = 'sine'

        # Generate the complete Figure 2
        fig, (ax1, ax2, ax3) = plot_figure_2(args, logger)

        # Display the plot
        plt.show()

    finally:
        # Close logger and restore standard output
        logger.close()
        sys.stdout = logger.terminal