from pathlib import Path
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


def plot_figure_3(args, logger):
    """
    Generate Figure 3 with two subplots: time reversibility and Hamiltonian conservation.

    This function visualizes:
    1. The forward and reverse time trajectories to check time reversibility.
    2. The Hamiltonian conservation over time for both L-HNN and traditional numerical gradients.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing model and sampling configurations.
    logger : Logger or file-like object
        Logger for printing and recording output messages.

    Returns
    -------
    tuple
        A tuple containing the matplotlib figure object and axes for the two subplots.
    """
    print("\nGenerating Figure 3...", file=logger)

    # Create a 1x2 figure layout
    fig = plt.figure(figsize=(12, 5))

    # ** Modify parameters for trajectory generation **
    args_traj = copy.deepcopy(args)
    args_traj.trajectory_length = 5  # Length of the trajectory
    args_traj.hmc_step_size = 0.05  # Step size
    args_traj.hmc_samples = 1  # Only one sample needed
    args_traj.num_burnin = 0  # No burn-in required

    # ** Subplot 1: Time Reversibility **
    print("\nPlotting time reversibility...", file=logger)
    ax1 = fig.add_subplot(121)

    # Set two different initial momentum values
    initial_momenta = [
        tf.constant([0.25], dtype=tf.float32),
        tf.constant([-0.5], dtype=tf.float32)
    ]

    # Define time span and compute time points
    t_span = [0.0, args_traj.trajectory_length]
    n_steps = int(args_traj.trajectory_length / args_traj.hmc_step_size)
    time_points = np.linspace(0, args_traj.trajectory_length, n_steps + 1)

    # Generate forward and reverse trajectories for each initial momentum
    for p0 in initial_momenta:
        # Traditional HMC trajectory
        q0 = tf.zeros([1], dtype=tf.float32)  # Fix initial position to 0
        initial_state = tf.concat([q0, p0], axis=0)
        z0 = tf.expand_dims(initial_state, 0)

        # Generate forward trajectory
        _, forward_traj = hnn_sampler.integrate_model(z0, t_span, n_steps)
        q_forward = forward_traj[:, 0, 0]

        # Use the endpoint of the forward trajectory as the starting point for the reverse trajectory
        q_end = q_forward[-1]
        p_end = forward_traj[-1, 0, 1]

        print(f"\nForward trajectory end state:")
        print(f"Position (q_end): {q_end.numpy()}")
        print(f"Momentum (p_end): {p_end.numpy()}")

        # Set the initial state for the reverse trajectory
        reverse_initial_state = tf.stack([q_end, -p_end])  # Reverse the momentum
        z0_reverse = tf.expand_dims(reverse_initial_state, 0)

        print(f"\nReverse trajectory initial state:")
        print(f"Position: {z0_reverse[0, 0].numpy()}")
        print(f"Momentum: {z0_reverse[0, 1].numpy()}")

        # Use the same forward time span
        t_span_forward = [0.0, args_traj.trajectory_length]  # Keep time moving forward

        # Generate reverse trajectory (using forward time span)
        _, reverse_traj = hnn_sampler.integrate_model(z0_reverse, t_span_forward, n_steps)
        q_reverse = reverse_traj[:, 0, 0]

        print(f"\nTrajectory shapes:")
        print(f"Forward trajectory: {q_forward.shape}")
        print(f"Reverse trajectory: {q_reverse.shape}")

        # Plot trajectories - reverse the time axis only when plotting
        ax1.plot(time_points, q_forward, 'b-', alpha=0.7)
        ax1.plot(time_points[::-1], q_reverse, 'r--', alpha=0.7)  # Reverse time on the plot

        # Add endpoint markers
        ax1.scatter(time_points[-1], q_forward[-1], color='blue', s=50, zorder=3)
        ax1.scatter(time_points[0], q_reverse[0], color='red', s=50, zorder=3)

    # Add legend and labels to Subplot 1
    ax1.plot([], [], 'b-', label='Forward time')
    ax1.plot([], [], 'r--', label='Reverse time')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Position (q)')
    ax1.set_title('(a) Time Reversibility')
    ax1.legend()
    ax1.grid(False)
    ax1.set_xlim([0, 5])

    # ** Subplot 2: Hamiltonian Conservation **
    print("\nPlotting Hamiltonian conservation...", file=logger)
    ax2 = fig.add_subplot(122)

    # Set different initial momentum values to obtain different energy levels
    initial_momenta = [
        tf.constant([0.25], dtype=tf.float32),
        tf.constant([0.5], dtype=tf.float32),
        tf.constant([1.0], dtype=tf.float32),
        tf.constant([1.25], dtype=tf.float32),
        tf.constant([1.5], dtype=tf.float32)
    ]

    # Compute Hamiltonian over time for each initial momentum
    for p0 in initial_momenta:
        # Set initial state
        q0 = tf.zeros([1], dtype=tf.float32)
        initial_state = tf.concat([q0, p0], axis=0)
        z0 = tf.expand_dims(initial_state, 0)

        # Generate trajectory using L-HNN
        _, l_hnn_traj = hnn_sampler.integrate_model(z0, t_span, n_steps)

        # Compute the Hamiltonian at each point of the trajectory (L-HNN)
        l_hnn_H = []
        for state in l_hnn_traj:
            H = functions(state, args)
            l_hnn_H.append(float(H))

        # Generate trajectory using traditional leapfrog and compute Hamiltonian
        trad_traj, _ = traditional_leapfrog(functions, z0, t_span, n_steps, args)
        trad_H = []
        for state in trad_traj:
            H = functions(state, args)
            trad_H.append(float(H))

        # Plot the Hamiltonian over time
        ax2.plot(time_points, l_hnn_H, 'r--', alpha=0.7)
        ax2.plot(time_points, trad_H, 'b-', alpha=0.7)

    # Add legend and labels to Subplot 2
    ax2.plot([], [], 'b-', label='Numerical gradients')
    ax2.plot([], [], 'r--', label='L-HNN')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Hamiltonian')
    ax2.set_title('(b) Hamiltonian Conservation')
    ax2.legend()
    ax2.grid(False)

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
    filename = figures_dir / f'figure_3_reproduction_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved as {filename}", file=logger)

    return fig, (ax1, ax2)


if __name__ == "__main__":
    """
    Main script to configure arguments, initialize the logger, and generate Figure 3.
    """
    import os
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).parent.parent
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    # Set up logging
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

        # Initialize the HNN sampler (created once in the main function)
        hnn_sampler = HNNSampler(args)

        # Generate Figure 3
        fig, (ax1, ax2) = plot_figure_3(args, logger)

        # Display the plot
        plt.show()

    finally:
        # Close logger and restore standard output
        logger.close()
        sys.stdout = logger.terminal