import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from hnn_nuts_olm import nuts_hnn_sample, get_model
from get_args import get_args
import os
from pathlib import Path
import datetime


def set_plot_style():
    """
    Set the basic style for matplotlib plots.

    This function updates `matplotlib` parameters to create consistent
    and clean visualizations with gridlines, white backgrounds, and
    other stylistic preferences.
    """
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.grid': True,
        'grid.color': '#CCCCCC',
        'grid.linestyle': '--',
        'grid.alpha': 0.3,
        'lines.linewidth': 1.0,
        'axes.spines.top': True,
        'axes.spines.right': True
    })


def plot_figure4(samples_lhnn_no_monitor):
    """
    Generate Figure 4: L-HNN sampling without error monitoring.

    This function visualizes the degradation in sampling quality
    when error monitoring is not applied to L-HNN.

    Parameters
    ----------
    samples_lhnn_no_monitor : np.ndarray
        The L-HNN samples generated without error monitoring.
    """
    plt.figure(figsize=(15, 4))

    # ** Subplot 1: q1 trace plot **
    plt.subplot(131)
    plt.plot(samples_lhnn_no_monitor[0, :, 0], 'k-', linewidth=1)
    plt.title('q1')
    plt.xlabel('Sample index')
    plt.ylabel('Value')
    plt.ylim(-3, 3)
    plt.grid(True, alpha=0.3)

    # ** Subplot 2: q2 trace plot **
    plt.subplot(132)
    plt.plot(samples_lhnn_no_monitor[0, :, 1], 'k-', linewidth=1)
    plt.title('q2')
    plt.xlabel('Sample index')
    plt.ylabel('Value')
    plt.ylim(0, 8)
    plt.grid(True, alpha=0.3)

    # ** Subplot 3: q3 trace plot **
    plt.subplot(133)
    plt.plot(samples_lhnn_no_monitor[0, :, 2], 'k-', linewidth=1)
    plt.title('q3')
    plt.xlabel('Sample index')
    plt.ylabel('Value')
    plt.ylim(0, 80)
    plt.grid(True, alpha=0.3)

    # Save and display the figure
    plt.tight_layout()
    figures_dir = Path("figures")
    if not figures_dir.exists():
        figures_dir.mkdir(parents=True)
        print(f"\nCreated directory: {figures_dir}")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = figures_dir / f'figure_4_reproduction_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()


def plot_comparison(samples_hnn, samples_lhnn, errors_hnn, errors_lhnn):
    """
    Generate Figures 5 and 6: Comparison of HNN and L-HNN sampling.

    This function creates two figures:
    1. Figure 5: q1 trace plots for HNN and L-HNN.
    2. Figure 6: Error plots for HNN and L-HNN.

    Parameters
    ----------
    samples_hnn : np.ndarray
        The samples generated using HNN.
    samples_lhnn : np.ndarray
        The samples generated using L-HNN.
    errors_hnn : np.ndarray
        The errors associated with HNN sampling.
    errors_lhnn : np.ndarray
        The errors associated with L-HNN sampling.
    """
    # ** Figure 5: q1 trace plots **
    plt.figure(figsize=(10, 4))

    # ** Subplot 1: HNN q1 trace plot **
    plt.subplot(121)
    plt.plot(samples_hnn[0, :, 0], 'b-', linewidth=1)
    plt.title('HNN')
    plt.xlabel('Sample index')
    plt.ylabel('q1')
    plt.grid(True, alpha=0.3)

    # ** Subplot 2: L-HNN q1 trace plot **
    plt.subplot(122)
    plt.plot(samples_lhnn[0, :, 0], 'r-', linewidth=1)
    plt.title('L-HNN')
    plt.xlabel('Sample index')
    plt.ylabel('q1')
    plt.grid(True, alpha=0.3)

    # Save and display Figure 5
    plt.tight_layout()
    figures_dir = Path("figures")
    if not figures_dir.exists():
        figures_dir.mkdir(parents=True)
        print(f"\nCreated directory: {figures_dir}")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = figures_dir / f'figure_5_reproduction_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

    # ** Figure 6: Error plots **
    plt.figure(figsize=(10, 4))

    # ** Subplot 1: HNN error plot **
    plt.subplot(121)
    plt.plot(errors_hnn[0], 'b-', linewidth=1)
    plt.axhline(y=10.0, color='r', linestyle='--', alpha=0.5)
    plt.title('HNN')
    plt.xlabel('Sample index')
    plt.ylabel('Error')
    plt.grid(True, alpha=0.3)

    # ** Subplot 2: L-HNN error plot **
    plt.subplot(122)
    plt.plot(errors_lhnn[0], 'r-', linewidth=1)
    plt.axhline(y=10.0, color='r', linestyle='--', alpha=0.5)
    plt.title('L-HNN')
    plt.xlabel('Sample index')
    plt.ylabel('Error')
    plt.grid(True, alpha=0.3)

    # Save and display Figure 6
    plt.tight_layout()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = figures_dir / f'figure_6_reproduction_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()


def run_comparison():
    """
    Run the comparison experiments and generate Figures 4, 5, and 6.

    This function:
    1. Configures the global plotting style.
    2. Runs L-HNN without error monitoring to generate Figure 4.
    3. Runs HNN and L-HNN with proper error monitoring to generate Figures 5 and 6.
    4. Prints statistics comparing HNN and L-HNN performance.
    """
    # Set global plot style
    set_plot_style()

    # ** Configure parameters for Figure 4 (L-HNN without error monitoring) **
    args = get_args()
    args.dist_name = 'nD_Rosenbrock'
    args.input_dim = 6  # 3D Rosenbrock
    args.total_samples = 5000  # Use fewer samples for Figure 4
    args.burn_in = 0  # No burn-in for Figure 4
    args.nuts_step_size = 0.025
    args.num_chains = 1
    args.n_cooldown = 20

    # Run L-HNN without error monitoring
    print("\n===== Testing L-HNN without error monitoring =====")
    args.latent_dim = 100
    args.hnn_error_threshold = 1e6  # Set a very high threshold to disable error monitoring
    model_lhnn = get_model(args)
    model_lhnn.load_weights(f"{args.save_dir}/lhnn_3d_rosenbrock")

    samples_lhnn_no_monitor, _, _, _ = nuts_hnn_sample(
        model_lhnn, args,
        traditional_only=False
    )

    # Generate Figure 4
    plot_figure4(samples_lhnn_no_monitor)

    # ** Configure parameters for Figures 5 and 6 (HNN and L-HNN comparison) **
    args = get_args()
    args.dist_name = 'nD_Rosenbrock'
    args.input_dim = 6  # 3D Rosenbrock
    args.total_samples = 20000  # Increase sample size to match the paper
    args.burn_in = 5000
    args.nuts_step_size = 0.025
    args.hnn_error_threshold = 10.0
    args.num_chains = 1
    args.n_cooldown = 20

    # Test HNN (latent_dim=1)
    print("\n===== Testing HNN (latent_dim=1) =====")
    args.latent_dim = 1
    model_hnn = get_model(args)
    model_hnn.load_weights(f"{args.save_dir}/hnn_3d_rosenbrock")

    samples_hnn, acceptance_hnn, errors_hnn, grads_hnn = nuts_hnn_sample(
        model_hnn, args,
        traditional_only=False
    )

    # Test L-HNN (latent_dim=100)
    print("\n===== Testing L-HNN (latent_dim=100) =====")
    args.latent_dim = 100
    model_lhnn = get_model(args)
    model_lhnn.load_weights(f"{args.save_dir}/lhnn_3d_rosenbrock")

    samples_lhnn, acceptance_lhnn, errors_lhnn, grads_lhnn = nuts_hnn_sample(
        model_lhnn, args,
        traditional_only=False
    )

    # Generate Figures 5 and 6
    plot_comparison(samples_hnn, samples_lhnn, errors_hnn, errors_lhnn)

    # Print comparison statistics
    print("\nComparison Statistics:")
    print(f"HNN total gradients: {grads_hnn}")
    print(f"L-HNN total gradients: {grads_lhnn}")
    print(f"HNN mean error: {np.mean(errors_hnn):.4f}")
    print(f"L-HNN mean error: {np.mean(errors_lhnn):.4f}")
    print(f"HNN acceptance rate: {np.mean(acceptance_hnn):.4f}")
    print(f"L-HNN acceptance rate: {np.mean(acceptance_lhnn):.4f}")


if __name__ == "__main__":
    """
    Main script to run the comparison experiments and generate the figures.
    """
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    run_comparison()
