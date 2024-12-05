import tensorflow as tf
import tensorflow_probability as tfp
from nn_models import MLP
from hnn import HNN
from utils import leapfrog, traditional_leapfrog
from functions import functions
from get_args import get_args
import sys




def get_model(args):
    """
    Build and load a pretrained HNN model.

    Args:
        args: Configuration containing model parameters.

    Returns:
        HNN: The pretrained HNN model.
    """
    nn_model = MLP(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        nonlinearity=args.nonlinearity
    )
    model = HNN(args.input_dim, differentiable_model=nn_model)
    return model


def build_tree(q, p, u, v, j, step_size, H0, hnn_model, state_dim,
               hnn_threshold, leapfrog_threshold, use_leapfrog, args, traditional_only = False):
    """
    Recursive function to construct a binary tree for NUTS sampling.

    Args:
        q (tf.Tensor): Position coordinates.
        p (tf.Tensor): Momentum coordinates.
        u (float): Slice sampling value.
        v (int): Direction of tree growth (-1 or 1).
        j (int): Depth of the tree.
        step_size (float): Leapfrog step size.
        H0 (float): Initial Hamiltonian value.
        hnn_model: The HNN model.
        state_dim (int): Dimensionality of the state space.
        hnn_threshold (float): HNN error threshold.
        leapfrog_threshold (float): Leapfrog error threshold.
        use_leapfrog (bool): Whether to use traditional leapfrog integration.
        args: Configuration parameters.
        traditional_only (bool): If True, only traditional leapfrog is used.

    Returns:
        Various outputs including updated positions, momenta, acceptance statistics,
        errors, and gradient counts.
    """
    grad_count = 0
    if j == 0:  # Base case: Perform a single leapfrog step
        z0 = tf.concat([q, p], axis=0)
        z0 = tf.expand_dims(z0, 0)
        step_size = tf.cast(step_size, tf.float32)
        v = tf.cast(v, tf.float32)  # Ensure v is of float type
        if traditional_only:   # Use traditional leapfrog directly (traditional NUTS)
            grad_count += 2
            traj, _ = traditional_leapfrog(functions, z0, [0, v * step_size], 1, args)
            z_new = traj[-1, 0, :]
            q_prime = z_new[:state_dim]
            p_prime = z_new[state_dim:]
            H_prime = functions(tf.expand_dims(z_new, 0), args)
            s_prime = (tf.math.log(u) + tf.squeeze(H_prime) <= leapfrog_threshold)
            n_prime = tf.cast(u <= tf.exp(-tf.squeeze(H_prime)), tf.int32)

            return (q_prime, p_prime, q_prime, p_prime, q_prime, p_prime,
                    n_prime, s_prime, tf.minimum(1.0, tf.exp(tf.squeeze(H0 - H_prime))), 1, 0.0, True, grad_count)

        # Use HNN-based leapfrog integration
        _, traj = leapfrog(hnn_model, z0, [0, v * step_size], 1)
        z_new = traj[-1, 0, :]
        q_prime = z_new[:state_dim]
        p_prime = z_new[state_dim:]
        H_prime = functions(tf.expand_dims(z_new, 0), args)


        # Check HNN error
        error = tf.math.log(u) + tf.squeeze(H_prime)
        use_leapfrog = tf.logical_or(tf.reduce_any(tf.cast(use_leapfrog, tf.bool)),
                                   error > hnn_threshold)
        s_prime = (error <= hnn_threshold)

        # Switch to traditional leapfrog if needed
        if tf.reduce_any(use_leapfrog):
            grad_count += 2
            traj, _ = traditional_leapfrog(functions, z0, [0, v * step_size], 1, args)
            z_new = traj[-1, 0, :]
            q_prime = z_new[:state_dim]
            p_prime = z_new[state_dim:]
            H_prime = functions(tf.expand_dims(z_new, 0), args)
            s_prime = (tf.math.log(u) + tf.squeeze(H_prime) <= leapfrog_threshold)

        # Update acceptance statistics
        n_prime = tf.cast(u <= tf.exp(-tf.squeeze(H_prime)), tf.int32)

        return (q_prime, p_prime, q_prime, p_prime, q_prime, p_prime,
                n_prime, s_prime, tf.minimum(1.0, tf.exp(tf.squeeze(H0 - H_prime))), 1, error, use_leapfrog, grad_count)

    else:
        # Recursive case: Build left and right subtrees
        (q_minus, p_minus, q_plus, p_plus, q_prime, p_prime,
         n_prime, s_prime, alpha, n_alpha, error, use_leapfrog, grad_count_1) = build_tree(
            q, p, u, v, j - 1, step_size, H0, hnn_model, state_dim,
            hnn_threshold, leapfrog_threshold, use_leapfrog, args, traditional_only)
        grad_count += grad_count_1
        if tf.reduce_all(tf.cast(s_prime, tf.bool)):
            if tf.equal(v, -1):
                # Extend tree to the left
                (q_minus_2, p_minus_2, _, _, q_prime_2, p_prime_2,
                 n_prime_2, s_prime_2, alpha_2, n_alpha_2, error_2,
                 use_leapfrog, grad_count_2) = build_tree(
                    q_minus, p_minus, u, v, j - 1, step_size, H0,
                    hnn_model, state_dim, hnn_threshold,
                    leapfrog_threshold, use_leapfrog, args, traditional_only)
                q_minus = q_minus_2
                p_minus = p_minus_2
                grad_count += grad_count_2
            elif tf.equal(v, 1):
                # Extend tree to the right
                (_, _, q_plus_2, p_plus_2, q_prime_2, p_prime_2,
                 n_prime_2, s_prime_2, alpha_2, n_alpha_2, error_2,
                 use_leapfrog, grad_count_2) = build_tree(
                    q_plus, p_plus, u, v, j - 1, step_size, H0,
                    hnn_model, state_dim, hnn_threshold,
                    leapfrog_threshold, use_leapfrog, args, traditional_only)
                q_plus = q_plus_2
                p_plus = p_plus_2
                grad_count += grad_count_2
            else:
                print('v not equal to 1 or -1')

            # Accept new samples with some probability
            accept_prob = tf.cast(n_prime_2, tf.float32) / tf.cast(
                n_prime + n_prime_2, tf.float32
            )
            random_accept = tf.less(tf.random.uniform([]), accept_prob)
            should_accept = tf.reduce_all(random_accept)

            if should_accept:
                q_prime = q_prime_2
                p_prime = p_prime_2

            # Update stop condition
            delta = q_plus - q_minus
            dot_minus = tf.tensordot(delta, p_minus, 1)
            dot_plus = tf.tensordot(delta, p_plus, 1)

            # make sure all conditions are of the same shape
            s_prime_2_scalar = tf.reduce_all(tf.cast(s_prime_2, tf.bool))
            dot_minus_cond = tf.greater_equal(dot_minus, 0.0)
            dot_plus_cond = tf.greater_equal(dot_plus, 0.0)

            # Combine all stopping conditions
            s_prime = tf.logical_and(s_prime_2_scalar,
                                     tf.logical_and(dot_minus_cond, dot_plus_cond))

            # Update statistics
            n_prime += n_prime_2
            alpha += alpha_2
            n_alpha += n_alpha_2
            error = tf.maximum(error, error_2)

        return (q_minus, p_minus, q_plus, p_plus, q_prime, p_prime,
                n_prime, s_prime, alpha, n_alpha, error, use_leapfrog, grad_count)


def nuts_hnn_sample(hnn_model, args, traditional_only=False, logger=None):
    """
    Perform NUTS-HNN sampling.

    Args:
        hnn_model: Pretrained HNN model.
        args: Configuration containing sampler parameters.
        traditional_only (bool): If True, use only traditional leapfrog integration.
        logger: Logger for output messages.

    Returns:
        samples: [num_chains, num_samples, dim]
        acceptance: [num_chains, num_samples]
    """
    logger = logger or sys.stdout
    state_dim = args.input_dim // 2
    total_dim = args.input_dim
    num_chains = args.num_chains
    num_samples = args.total_samples
    num_burnin = args.burn_in
    step_size = args.nuts_step_size
    hnn_threshold = args.hnn_error_threshold
    if traditional_only == True:
        hnn_threshold = -1e-6  # Disable HNN if only traditional leapfrog is used
    leapfrog_threshold = args.leapfrog_error_threshold
    n_cooldown = args.n_cooldown
    # 初始化存储数组
    samples = tf.zeros([num_chains, num_samples, total_dim])
    acceptance = tf.zeros([num_chains, num_samples])
    errors = tf.zeros([num_chains, num_samples])  # Store error for each sample
    total_grads = 0

    # Sampling for each Markov chain
    for chain in range(num_chains):
        current_state = tf.zeros(state_dim)
        use_leapfrog = False
        leapfrog_count = 0

        burnin_accepted = 0
        sampling_accepted = 0
        total_burnin_tried = 0
        total_sampling_tried = 0

        for sample in range(num_samples):
            phase = "Burn-in" if sample < num_burnin else "Sampling"

            # Sample momentum from a standard normal distribution
            p0 = tf.random.normal([state_dim])

            # Compute initial Hamiltonian
            z0 = tf.concat([current_state, p0], axis=0)  #current state is q0
            H0 = functions(tf.expand_dims(z0, 0), args)

            # Sample slice value u
            u = tf.random.uniform([]) * tf.exp(-tf.squeeze(H0))
            u = tf.cast(u, tf.float32)

            # Initialize tree
            q_minus = current_state
            q_plus = current_state
            p_minus = p0
            p_plus = p0
            j = 0
            n_prime = 1
            s = 1

            # Switch back to HNN after cooldown
            if tf.reduce_any(use_leapfrog):
                leapfrog_count += 1
            if leapfrog_count == n_cooldown:
                use_leapfrog = False
                leapfrog_count = 0

            # Build tree
            max_error = tf.constant(-float('inf'))  # 初始化当前sample的最大error
            sample_accepted = False
            while tf.reduce_all(tf.cast(s, tf.bool)):
                # randomly select direction
                v = tf.where(tf.random.uniform([]) < 0.5, tf.constant(1), tf.constant(-1))

                if tf.equal(v, -1):
                    # left
                    (q_minus, p_minus, _, _, q_prime, p_prime,
                     n_prime_2, s_prime, alpha, n_alpha, error,
                     use_leapfrog, grad_count) = build_tree(
                        q_minus, p_minus, u, v, j, step_size,
                        H0, hnn_model, state_dim, hnn_threshold,
                        leapfrog_threshold, use_leapfrog, args, traditional_only = traditional_only)
                else:
                    # right
                    (_, _, q_plus, p_plus, q_prime, p_prime,
                     n_prime_2, s_prime, alpha, n_alpha, error,
                     use_leapfrog, grad_count) = build_tree(
                        q_plus, p_plus, u, v, j, step_size,
                        H0, hnn_model, state_dim, hnn_threshold,
                        leapfrog_threshold, use_leapfrog, args, traditional_only = traditional_only)
                total_grads += grad_count
                max_error = tf.maximum(max_error, error)

                # Metropolis acceptance/rejection
                if tf.reduce_all(tf.cast(s_prime, tf.bool)):
                    ratio = tf.cast(n_prime_2, tf.float32) / tf.cast(n_prime, tf.float32)
                    accept_prob = tf.minimum(tf.constant(1.0), ratio)
                    random_accept = tf.less(tf.random.uniform([]), accept_prob)
                    should_accept = tf.reduce_all(random_accept)

                    if should_accept:
                        current_state = q_prime
                        current_momentum = p_prime
                        z0 = tf.concat([current_state, current_momentum], axis=0)
                        sample_accepted = True

                n_prime += n_prime_2

                # Check stop condition
                delta = q_plus - q_minus
                dot_minus = tf.tensordot(delta, p_minus, 1)
                dot_plus = tf.tensordot(delta, p_plus, 1)

                s_prime_scalar = tf.reduce_all(tf.cast(s_prime, tf.bool))
                dot_minus_cond = tf.greater_equal(dot_minus, 0.0)
                dot_plus_cond = tf.greater_equal(dot_plus, 0.0)

                s = tf.logical_and(s_prime_scalar,
                                   tf.logical_and(dot_minus_cond, dot_plus_cond))

                j += 1

            if sample_accepted:
                if sample < num_burnin:
                    burnin_accepted += 1
                    total_burnin_tried += 1
                else:
                    sampling_accepted += 1
                    total_sampling_tried += 1

                acceptance = tf.tensor_scatter_nd_update(
                    acceptance,
                    [[chain, sample]],
                    [1.0]
                )

                if sample < num_burnin:
                    current_rate = burnin_accepted / total_burnin_tried
                else:
                    current_rate = sampling_accepted / total_sampling_tried

                print(f"Chain {chain + 1}/{num_chains}, "
                      f"Sample {sample + 1}/{num_samples} ({phase}): "
                      f"ACCEPTED (current rate: {current_rate:.4f})",
                      file=logger)

            else:
                if sample < num_burnin:
                    total_burnin_tried += 1
                else:
                    total_sampling_tried += 1



            # after while loop record the largest error
            errors = tf.tensor_scatter_nd_update(
                errors,
                [[chain, sample]],
                [max_error]
            )
            # save sample
            samples = tf.tensor_scatter_nd_update(
                samples,
                [[chain, sample]],
                [z0]
            )

        # print acceptance rate
        burnin_rate = burnin_accepted / total_burnin_tried if total_burnin_tried > 0 else 0.0
        sampling_rate = sampling_accepted / total_sampling_tried if total_sampling_tried > 0 else 0.0
        print(f"\nChain {chain + 1} Statistics:", file=logger)
        print(f"Burn-in phase acceptance rate: {burnin_rate:.4f} ({burnin_accepted}/{total_burnin_tried})",
              file=logger)
        print(f"Sampling phase acceptance rate: {sampling_rate:.4f} ({sampling_accepted}/{total_sampling_tried})",
              file=logger)

    return samples, acceptance[:, num_burnin:], errors, total_grads

