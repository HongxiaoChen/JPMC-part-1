import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF
from hnn_nuts_olm import nuts_hnn_sample, get_model
from get_args import get_args
from pathlib import Path
from datetime import datetime
import time
import logging
from utils import setup_logger, compute_ess


def plot_figure9(samples_nuts, samples_lhnn, burn_in):
    """
    Generate Figure 9: Scatter plot matrix.

    This function creates a scatter plot matrix to visualize the sampling results
    from NUTS and LHNN-NUTS methods for a 5D ill-conditioned Gaussian distribution.

    Parameters
    ----------
    samples_nuts : np.ndarray
        Samples generated by NUTS.
    samples_lhnn : np.ndarray
        Samples generated by LHNN-NUTS.
    burn_in : int
        Number of initial samples to discard as burn-in.
    """
    logger = logging.getLogger('Gaussian_Comparison')
    logger.info("Creating scatter plot matrix...")

    # ** Remove burn-in samples **
    samples_nuts = samples_nuts[:, burn_in:, :]
    samples_lhnn = samples_lhnn[:, burn_in:, :]
    fig, axes = plt.subplots(5, 5, figsize=(15, 15))

    # ** Iterate over all dimension combinations **
    for i in range(5):
        for j in range(5):
            ax = axes[i, j]

            if i == j:
                # ** Diagonal: Kernel Density Estimation (KDE) **
                nuts_data = samples_nuts[0, :, i]
                lhnn_data = samples_lhnn[0, :, i]

                # Compute x-axis range
                min_val = min(np.min(nuts_data), np.min(lhnn_data))
                max_val = max(np.max(nuts_data), np.max(lhnn_data))
                x = np.linspace(min_val, max_val, 200)

                # KDE computation (replace `plt.mlab.GaussianKDE` with `scipy.stats.gaussian_kde`)
                from scipy.stats import gaussian_kde
                nuts_kde = gaussian_kde(nuts_data)
                lhnn_kde = gaussian_kde(lhnn_data)

                # Plot KDE curves
                ax.plot(x, nuts_kde(x), color='blue', alpha=0.8, label='NUTS')
                ax.plot(x, lhnn_kde(x), color='red', alpha=0.8, label='LHNN-NUTS')
            else:
                # ** Off-diagonal: Scatter plot **
                ax.scatter(samples_nuts[0, :, j], samples_nuts[0, :, i],
                           alpha=0.5, color='blue', s=1, label='NUTS')
                ax.scatter(samples_lhnn[0, :, j], samples_lhnn[0, :, i],
                           alpha=0.5, color='red', s=1, label='LHNN-NUTS')

            # ** Set axis labels **
            if i == 4:  # Bottom row
                ax.set_xlabel(f'q{j + 1}')
            if j == 0:  # Leftmost column
                ax.set_ylabel(f'q{i + 1}')

            # Remove tick labels to reduce clutter
            ax.set_xticks([])
            ax.set_yticks([])

    # ** Add legend (only in the first subplot) **
    axes[0, 0].legend()

    # Adjust subplot spacing
    plt.tight_layout()

    # ** Save and display the figure **
    figures_dir = Path("figures")
    if not figures_dir.exists():
        figures_dir.mkdir(parents=True)
        print(f"\nCreated directory: {figures_dir}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = figures_dir / f'figure_9_reproduction_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    logger.info(f"Plot saved to {filename}")


def compute_gaussian_metrics(samples_nuts, nuts_grads,
                             samples_lhnn, lhnn_monitoring_grads,
                             burn_in=5000):
    """
    Compute performance metrics for a 5D ill-conditioned Gaussian distribution.

    This function computes:
    1. Effective Sample Size (ESS) for NUTS and LHNN-NUTS.
    2. Number of gradients evaluated during sampling.
    3. ESS per gradient for both methods.

    Parameters
    ----------
    samples_nuts : np.ndarray
        Samples generated by NUTS.
    nuts_grads : int
        Total number of gradients evaluated by NUTS.
    samples_lhnn : np.ndarray
        Samples generated by LHNN-NUTS.
    lhnn_monitoring_grads : int
        Total number of gradients evaluated during error monitoring by LHNN-NUTS.
    burn_in : int, optional
        Number of initial samples to discard as burn-in. Defaults to 5000.

    Returns
    -------
    dict
        A dictionary containing ESS, total gradients, and ESS per gradient for
        NUTS and LHNN-NUTS.
    """
    logger = logging.getLogger('Gaussian_Comparison')
    logger.info("Computing ESS for NUTS samples...")

    # ** Create logs directory if it doesn't exist **
    logs_dir = Path("logs")
    if not logs_dir.exists():
        logs_dir.mkdir(parents=True)
        logger.info(f"\nCreated directory: {logs_dir}")

    # ** Create a timestamped log file **
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f'gaussian_metrics_{timestamp}.log'

    def log_print(message):
        """
        Log a message to both the console and the log file.
        """
        print(message)
        with open(log_file, 'a') as f:
            f.write(message + '\n')

            # ** Compute ESS for NUTS and LHNN-NUTS **

    ess_nuts = compute_ess(samples_nuts, burn_in)
    ess_lhnn = compute_ess(samples_lhnn, burn_in)

    # ** Compute training gradients for LHNN-NUTS **
    training_grads = 40 * 250 * 40  # Mt=40, T=250, 40 steps per unit

    # ** Log performance metrics **
    log_print("\n=== 5-D ill-conditioned Gaussian Performance Comparison ===")
    log_print("{:<15} {:<45} {:<20} {:<15}".format(
        "Method", "ESS", "# gradients", "ESS per grad"))
    log_print("-" * 95)

    # ** NUTS Metrics **
    total_grads_nuts = nuts_grads
    avg_ess_nuts = np.mean(ess_nuts)
    ess_per_grad_nuts = avg_ess_nuts / total_grads_nuts
    log_print("{:<15} ({:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f})".format(
        "NUTS", *ess_nuts))
    log_print("{:<60} {:<20} {:.6f}".format(
        "", total_grads_nuts, ess_per_grad_nuts))
    log_print("")

    # ** LHNN-NUTS Metrics **
    total_grads_lhnn = training_grads + lhnn_monitoring_grads
    avg_ess_lhnn = np.mean(ess_lhnn)
    ess_per_grad_lhnn = avg_ess_lhnn / total_grads_lhnn
    log_print("{:<15} ({:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f})".format(
        "LHNN-NUTS", *ess_lhnn))
    log_print("{:<60} Evaluation: {}".format("", lhnn_monitoring_grads))
    log_print("{:<60} Training: {}".format("", training_grads))
    log_print("{:<60} {:<20} {:.6f}".format(
        "", total_grads_lhnn, ess_per_grad_lhnn))

    return {
        'NUTS': {'ess': ess_nuts, 'grads': total_grads_nuts, 'ess_per_grad': ess_per_grad_nuts},
        'LHNN': {'ess': ess_lhnn, 'grads': total_grads_lhnn, 'ess_per_grad': ess_per_grad_lhnn}
    }


def run_gaussian_comparison():
    """
    Run the comparison experiments for a 5D ill-conditioned Gaussian distribution.

    This function:
    1. Configures the sampling parameters for NUTS and LHNN-NUTS.
    2. Runs the sampling for both methods.
    3. Computes performance metrics (ESS, gradients, ESS per gradient).
    4. Generates Figure 9 to visualize the comparison.
    """
    logger = setup_logger('Gaussian_Comparison')
    start_time = time.time()

    # ** Set up basic parameters **
    args = get_args()
    args.dist_name = '5D_illconditioned_Gaussian'
    args.input_dim = 10  # 5D Gaussian (position + momentum = 10 dimensions)
    args.latent_dim = 5
    args.total_samples = 20000  # Total samples (adjust as needed)
    args.burn_in = 5000  # Burn-in samples
    args.nuts_step_size = 0.025
    args.hnn_error_threshold = 10.0
    args.num_chains = 1
    args.n_cooldown = 20

    logger.info("Starting Gaussian comparison with parameters:")
    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")

    # ** Run traditional NUTS **
    logger.info("\n===== Running Traditional NUTS =====")
    model_trad = get_model(args)
    samples_nuts, accept_nuts, errors_nuts, nuts_grads = nuts_hnn_sample(
        model_trad, args,
        traditional_only=True
    )
    logger.info(f"NUTS sampling completed. Accept rate: {np.mean(accept_nuts.numpy()):.4f}")
    logger.info(f"Number of monitoring gradient evaluations: {nuts_grads}")

    # ** Run LHNN-NUTS **
    logger.info("\n===== Running LHNN-NUTS =====")
    model_lhnn = get_model(args)
    # Load pretrained model weights
    model_lhnn.load_weights(f"{args.save_dir}/5D_illconditioned_Gaussian250")
    logger.info(f"Loading pretrained model from {args.save_dir}/5D_illconditioned_Gaussian250")
    logger.info("Starting LHNN-NUTS sampling...")
    samples_lhnn, accept_lhnn, errors_lhnn, lhnn_monitoring_grads = nuts_hnn_sample(
        model_lhnn, args,
        traditional_only=False
    )
    logger.info(f"LHNN-NUTS sampling completed. Accept rate: {np.mean(accept_lhnn.numpy()):.4f}")
    logger.info(f"Number of monitoring gradient evaluations: {lhnn_monitoring_grads}")

    # ** Generate Figure 9 **
    logger.info("Generating plots...")
    plot_figure9(samples_nuts, samples_lhnn, burn_in=args.burn_in)
    logger.info("Plots generated and saved")

    # ** Compute performance metrics **
    logger.info("Computing performance metrics...")
    metrics = compute_gaussian_metrics(
        samples_nuts, nuts_grads,
        samples_lhnn, lhnn_monitoring_grads,
        burn_in=args.burn_in
    )

    elapsed_time = time.time() - start_time
    logger.info(f"Total execution time: {elapsed_time:.2f} seconds")

    return metrics


if __name__ == "__main__":
    """
    Main script to run the comparison experiments for a 5D ill-conditioned Gaussian distribution.
    """
    import os

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    run_gaussian_comparison()
