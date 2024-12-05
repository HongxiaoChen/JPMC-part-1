import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from hnn_nuts_olm import nuts_hnn_sample, get_model
from get_args import get_args
from pathlib import Path
from datetime import datetime
import time
import logging
from functions import f_obs, f_tf
from utils import setup_logger, compute_average_ess


def plot_figure12(samples_nuts, samples_lhnn, burn_in):
    """复制论文Figure 12：显示f(x,y)场并比较NUTS和LHNN-NUTS的结果"""
    logger = logging.getLogger('Elliptic_Comparison')
    logger.info("Creating f(x,y) field plots...")
    samples_nuts = samples_nuts[:, burn_in:, :]
    samples_lhnn = samples_lhnn[:, burn_in:, :] 
    # 获取观测数据点
    f_obs_values, x_samples, y_samples = f_obs()

    # 创建网格用于绘制完整的f(x,y)场
    x_grid = np.linspace(0, 3, 200)
    y_grid = np.linspace(0, 3, 200)
    X, Y = np.meshgrid(x_grid, y_grid)

    # (a) 计算原始f(x,y)场
    X_tf = tf.constant(X, dtype=tf.float32)
    Y_tf = tf.constant(Y, dtype=tf.float32)
    F = f_tf(X_tf, Y_tf)

    # 计算NUTS和LHNN的加权平均
    nuts_samples = samples_nuts[0, :, :50]  # [num_samples, 50]
    lhnn_samples = samples_lhnn[0, :, :50]  # [num_samples, 50]

    # 计算每个样本的f(q)，然后取平均
    nuts_f_mean = compute_f_mean(nuts_samples, f_obs_values)
    lhnn_f_mean = compute_f_mean(lhnn_samples, f_obs_values)

    # 使用插值重建完整场
    F_nuts = griddata((x_samples.numpy(), y_samples.numpy()),
                      nuts_f_mean.numpy(),
                      (X, Y), method='cubic')

    F_lhnn = griddata((x_samples.numpy(), y_samples.numpy()),
                      lhnn_f_mean.numpy(),
                      (X, Y), method='cubic')

    # 创建图形和子图
    fig = plt.figure(figsize=(15, 15))

    # 创建四个子图
    ax1 = plt.subplot(2, 2, 1)
    ax2 = plt.subplot(2, 2, 2)
    ax3 = plt.subplot(2, 2, 3)
    ax4 = plt.subplot(2, 2, 4)

    # (a) 原始f(x,y)场和观测点
    im1 = ax1.imshow(F.numpy(), extent=[0, 3, 0, 3], origin='lower',
                     aspect='equal', cmap='RdYlBu')
    ax1.scatter(x_samples, y_samples, c='black', s=20)
    ax1.set_title('(a) f(x,y) Field with Sensor Locations')
    plt.colorbar(im1, ax=ax1)

    # (b) NUTS重建的f(x,y)场
    im2 = ax2.imshow(F_nuts, extent=[0, 3, 0, 3], origin='lower',
                     aspect='equal', cmap='RdYlBu')
    ax2.set_title('(b) NUTS Reconstructed f(x,y)')
    plt.colorbar(im2, ax=ax2)

    # (c) LHNN-NUTS重建的f(x,y)场
    im3 = ax3.imshow(F_lhnn, extent=[0, 3, 0, 3], origin='lower',
                     aspect='equal', cmap='RdYlBu')
    ax3.set_title('(c) LHNN-NUTS Reconstructed f(x,y)')
    plt.colorbar(im3, ax=ax3)

    # (d) 均方误差场
    mse = np.square(F_nuts - F_lhnn)
    im4 = ax4.imshow(mse, extent=[0, 3, 0, 3], origin='lower',
                     aspect='equal', cmap='viridis')
    ax4.set_title('(d) Mean Squared Error')
    plt.colorbar(im4, ax=ax4)

    # 设置所有子图的标签
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()

    figures_dir = Path("figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = figures_dir / f'figure_12_reproduction_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    logger.info(f"Plot saved to {filename}")

    return fig


@tf.function(experimental_relax_shapes=True)
def compute_f_mean(samples, f_obs_values):
    """计算所有样本的f(q) = exp(-0.5 * Σ(f_obs - q)²)的平均值
    Args:
        samples: shape [num_samples, 50] - 每个样本的q值
        f_obs_values: shape [50] - 观测值
    Returns:
        mean of f(q): shape [50]
    """
    # 广播f_obs_values到samples的形状
    f_obs_expanded = tf.expand_dims(f_obs_values, 0)  # [1, 50]

    # 计算每个样本的f(q)
    diff = f_obs_expanded - samples  # [num_samples, 50]
    U = 0.5 * tf.square(diff)  # [num_samples, 50]
    f_values = tf.exp(-U)  # [num_samples, 50]

    # 简单算术平均
    return tf.reduce_mean(f_values, axis=0)  # [50]


def compute_elliptic_metrics(samples_nuts, nuts_grads,
                             samples_lhnn, lhnn_monitoring_grads,
                             burn_in=1000):
    """计算Elliptic方程的性能指标"""
    logger = logging.getLogger('Elliptic_Comparison')
    logger.info("Computing metrics...")


    # 计算平均ESS
    avg_ess_nuts = compute_average_ess(samples_nuts, burn_in)
    avg_ess_lhnn = compute_average_ess(samples_lhnn, burn_in)

    # 计算训练数据的梯度次数
    training_grads = 64000  # 根据论文中的数值

    # 计算总梯度评估次数
    total_grads_nuts = nuts_grads
    total_grads_lhnn = training_grads + lhnn_monitoring_grads

    # 计算每个梯度的ESS
    ess_per_grad_nuts = avg_ess_nuts / total_grads_nuts
    ess_per_grad_lhnn = avg_ess_lhnn / total_grads_lhnn

    # 打印结果
    logger.info("\n=== Elliptic Performance Comparison ===")
    logger.info(f"NUTS - Average ESS: {avg_ess_nuts:.2f}")
    logger.info(f"NUTS - Total gradients: {total_grads_nuts}")
    logger.info(f"NUTS - ESS per gradient: {ess_per_grad_nuts:.6f}")

    logger.info(f"\nLHNN-NUTS - Average ESS: {avg_ess_lhnn:.2f}")
    logger.info(f"LHNN-NUTS - Evaluation gradients: {lhnn_monitoring_grads}")
    logger.info(f"LHNN-NUTS - Training gradients: {training_grads}")
    logger.info(f"LHNN-NUTS - Total gradients: {total_grads_lhnn}")
    logger.info(f"LHNN-NUTS - ESS per gradient: {ess_per_grad_lhnn:.6f}")

    return {
        'NUTS': {'ess': avg_ess_nuts, 'grads': total_grads_nuts,
                 'ess_per_grad': ess_per_grad_nuts},
        'LHNN': {'ess': avg_ess_lhnn, 'grads': total_grads_lhnn,
                 'ess_per_grad': ess_per_grad_lhnn}
    }


def run_elliptic_comparison():
    logger = setup_logger('Elliptic_Comparison')
    start_time = time.time()

    # 设置参数
    args = get_args()

    args.latent_dim = 50
    args.total_samples = 10000
    args.burn_in = 0
    args.nuts_step_size = 0.025
    args.hnn_error_threshold = 10.0
    args.num_chains = 1
    args.n_cooldown = 20

    logger.info("Starting Elliptic comparison with parameters:")
    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")

    # 运行传统NUTS
    logger.info("\n===== Running Traditional NUTS =====")
    model_trad = get_model(args)
    samples_nuts, accept_nuts, errors_nuts, nuts_grads = nuts_hnn_sample(
        model_trad, args, traditional_only=True
    )
    logger.info(f"NUTS sampling completed. Accept rate: {np.mean(accept_nuts.numpy()):.4f}")

    # 运行LHNN-NUTS
    logger.info("\n===== Running LHNN-NUTS =====")
    model_lhnn = get_model(args)
    #model_lhnn.load_weights(f"{args.save_dir}/Elliptic40")
    samples_lhnn, accept_lhnn, errors_lhnn, lhnn_monitoring_grads = nuts_hnn_sample(
        model_lhnn, args, traditional_only=True
    )
    logger.info(f"LHNN-NUTS sampling completed. Accept rate: {np.mean(accept_lhnn.numpy()):.4f}")

    # 绘制图形
    plot_figure12(samples_nuts, samples_lhnn, burn_in=args.burn_in)

    # 计算和打印指标
    metrics = compute_elliptic_metrics(
        samples_nuts, nuts_grads,
        samples_lhnn, lhnn_monitoring_grads,
        burn_in=args.burn_in
    )

    elapsed_time = time.time() - start_time
    logger.info(f"Total execution time: {elapsed_time:.2f} seconds")

    return metrics


if __name__ == "__main__":
    import os
    import tensorflow_probability as tfp


    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    run_elliptic_comparison()