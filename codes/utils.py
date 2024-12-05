import tensorflow as tf
import numpy as np
import pickle
import logging
import tensorflow
from pathlib import Path
import tensorflow_probability as tfp
from datetime import datetime
import time


def dynamics_fn(function, z, args):
    """
    Compute the derivatives in a Hamiltonian system.

    Args:
        function: The Hamiltonian function.
        z (tf.Tensor): Current state [batch_size, dim].
        args: Configuration arguments.

    Returns:
        tf.Tensor: Derivatives [dq/dt, dp/dt] with shape [batch_size, dim].
    """
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(z)
        H = function(z, args)  # [batch_size, 1]

    grads = tape.gradient(H, z)  # [batch_size, dim]

    dim = z.shape[1] // 2
    dH_dq = grads[:, :dim]  # ∂H/∂q = -dp/dt
    dH_dp = grads[:, dim:]  # ∂H/∂p = dq/dt

    del tape
    return tf.concat([dH_dp, -dH_dq], axis=1)  # return [dq/dt, dp/dt]

def traditional_leapfrog(function: object, z0: object, t_span: object, n_steps: object, args: object) -> object:
    """
    Traditional leapfrog integrator.

    Args:
        function: The Hamiltonian function.
        z0 (tf.Tensor): Initial state [batch_size, dim].
        t_span (list): Time range [t0, t1].
        n_steps (int): Number of integration steps.
        args: Configuration arguments.

    Returns:
        tuple: Trajectories (z) and derivatives (dz) with shapes:
               z: [n_steps+1, batch_size, dim],
               dz: [n_steps+1, batch_size, dim].
    """
    if len(z0.shape) == 1:
        z0 = tf.expand_dims(z0, 0)
    dt = (t_span[1] - t_span[0]) / n_steps
    n_steps = tf.cast(n_steps, tf.int32)
    t_span = tf.cast(t_span, tf.float32)

    # initialize storage
    z = tf.TensorArray(tf.float32, size=n_steps + 1, clear_after_read=False)
    dz = tf.TensorArray(tf.float32, size=n_steps + 1, clear_after_read=False)

    # Store initial values
    z = z.write(0, z0)
    dz0 = dynamics_fn(function, z0, args)
    dz = dz.write(0, dz0)

    # Main loop
    for i in tf.range(n_steps):
        z_curr = z.read(i)
        dim = z_curr.shape[1] // 2
        q = z_curr[:, :dim]
        p = z_curr[:, dim:]

        # Compute current gradients
        dz_curr = dynamics_fn(function, z_curr, args) # z -> H, then return dH/dp = dq/dt, -dH/dq = dp/dt
        dH_dq = -dz_curr[:, dim:]  # dH/dq = -dp/dt

        # Update position
        q_next = q + dt * p - (dt ** 2) / 2 * dH_dq

        # Compute gradients at the new position
        z_temp = tf.concat([q_next, p], axis=1)
        dz_next = dynamics_fn(function, z_temp, args)
        dH_dq_next = -dz_next[:, dim:]

        # Update momentum
        p_next = p - dt / 2 * (dH_dq + dH_dq_next)

        # Store new state
        z_next = tf.concat([q_next, p_next], axis=1)
        z = z.write(i + 1, z_next)
        dz = dz.write(i + 1, dynamics_fn(function, z_next, args))

    return z.stack(), dz.stack()  # return a list of data and deriv #return [q,p], [dq/dt, dp/dt]

@tf.function(experimental_relax_shapes=True)
def leapfrog(model, z0, t_span, n_steps):
    """
    Synchronized leapfrog integrator based on Equations (7) and (8) in the paper.

    Args:
        model: HNN model instance.
        z0 (tf.Tensor): Initial state [batch_size, dim].
        t_span (list): Time range [t0, t1].
        n_steps (int): Number of integration steps.

    Returns:
        tuple: Times (t) and trajectories (z) with shapes:
               t: [n_steps+1],
               z: [n_steps+1, batch_size, dim].
    """
    if len(z0.shape) == 1:
        z0 = tf.expand_dims(z0, 0)  # Add batch dimension

    dt = (t_span[1] - t_span[0]) / n_steps
    dim = z0.shape[-1] // 2  # Half for position, half for momentum
    n_steps = tf.cast(n_steps, tf.int32)
    t_span = tf.cast(t_span, tf.float32)

    # Initialize
    t = tf.linspace(t_span[0], t_span[1], n_steps + 1)
    z = tf.TensorArray(z0.dtype, size=n_steps + 1, clear_after_read=False)
    z = z.write(0, z0)
    z_curr = z0

    for i in tf.range(n_steps):
        q, p = z_curr[:, :dim], z_curr[:, dim:]

        # ∂H/∂q(t)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(z_curr)
            H = model.compute_hamiltonian(z_curr)
        grads = tape.gradient(H, z_curr)
        dH_dq = grads[:, :dim]  # ∂H/∂q(t)

        # Update position
        q_next = q + dt / model.M * p - (dt ** 2) / (2 * model.M) * dH_dq

        # Compute ∂H/∂q(t+∆t)
        z_next_temp = tf.concat([q_next, p], axis=1)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(z_next_temp)
            H_next = model.compute_hamiltonian(z_next_temp)
        grads_next = tape.gradient(H_next, z_next_temp)
        dH_dq_next = grads_next[:, :dim]  # ∂H/∂q(t+∆t)

        # Update momentum
        p_next = p - dt / 2 * (dH_dq + dH_dq_next)

        # combine new states
        z_next = tf.concat([q_next, p_next], axis=1)
        z = z.write(i + 1, z_next)
        z_curr = z_next
        del tape
    return t, z.stack() # [ time_steps, batch_size=1, total_dim]

@tf.function(experimental_relax_shapes=True)
def L2_loss(u, v):
    """
    Compute the L2 loss.

    Args:
        u (tf.Tensor): Predicted values.
        v (tf.Tensor): Ground truth values.

    Returns:
        tf.Tensor: L2 loss.
    """
    return tf.reduce_mean(tf.square(u - v))


def to_pickle(thing, path):
    """
    Save an object to a pickle file.

    Args:
        obj: Object to save.
        path (str): File path to save the pickle.
    """
    with open(path, 'wb') as handle:
        pickle.dump(thing, handle, protocol=pickle.HIGHEST_PROTOCOL)


def from_pickle(path):  # load something
    """
    Load an object from a pickle file.

    Args:
        path (str): File path to load the pickle.

    Returns:
        Object: The loaded object.
    """
    with open(path, 'rb') as handle:
        thing = pickle.load(handle)
    return thing

@tf.function(experimental_relax_shapes=True)
def compute_gradients(model, x):
    """
    Compute gradients required for training.

    Args:
        model: HNN model.
        x (tf.Tensor): Input data.

    Returns:
        tuple: Gradients of momentum (∂H/∂p) and position (-∂H/∂q).
    """
    with tf.GradientTape() as tape:
        tape.watch(x)
        H = model.compute_hamiltonian(x)

    grads = tape.gradient(H, x)
    return grads[:, model.dim:], -grads[:, :model.dim]  # ∂H/∂p, -∂H/∂q

@tf.function(experimental_relax_shapes=True)
def compute_training_loss(model, batch_data, time_derivatives):
    """
    Compute the training loss based on Equation (13) in the paper.

    Args:
        model: HNN model.
        batch_data (tf.Tensor): Input data [batch_size, dim].
        time_derivatives (tuple): True gradients of position and momentum.

    Returns:
        tf.Tensor: Training loss.
    """
    dq_pred, dp_pred = compute_gradients(model, batch_data)
    dq_true, dp_true = time_derivatives

    loss = tf.reduce_mean(tf.square(dq_pred - dq_true)) + \
           tf.reduce_mean(tf.square(dp_pred - dp_true))
    return loss


def setup_logger(name):
    """
    Set up a logger that writes to both a file and the console.

    Args:
        name (str): Name of the logger.

    Returns:
        logging.Logger: Configured logger instance.
    """
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)


    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'{name}_{timestamp}.log'

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger

def compute_ess(samples, burn_in):
    """
    Compute the Effective Sample Size (ESS) for each dimension.

    Args:
        samples (tf.Tensor): Samples [num_chains, num_samples, dim].
        burn_in (int): Number of burn-in samples to exclude.

    Returns:
        list: ESS values for each dimension.
    """
    ess_values = []
    state_dim = samples.shape[-1] // 2
    for dim in range(state_dim):
        chain_samples = samples[0, burn_in:, dim]
        ess = tfp.mcmc.effective_sample_size(chain_samples)
        ess_values.append(float(ess.numpy()))
    return ess_values

def compute_average_ess(samples, burn_in):
    """
    Compute the average ESS over all dimensions.

    Args:
        samples (tf.Tensor): Samples [num_chains, num_samples, dim].
        burn_in (int): Number of burn-in samples to exclude.

    Returns:
        float: Average ESS across all dimensions.
    """
    ess_values = []
    state_dim = samples.shape[-1] // 2
    for dim in range(state_dim):
        chain_samples = samples[0, burn_in:, dim]
        ess = tfp.mcmc.effective_sample_size(chain_samples)
        ess_values.append(float(ess.numpy()))
    return np.mean(ess_values)



