import unittest
import sys
import os
import tensorflow as tf
import numpy as np
from pathlib import Path
import tempfile
import logging

from codes.utils import (dynamics_fn, traditional_leapfrog, L2_loss,
                         to_pickle, from_pickle, setup_logger,
                         compute_ess, compute_average_ess)
from codes.functions import functions
from codes.get_args import get_args


class TestUtils(unittest.TestCase):
    def setUp(self):
        self.args = get_args()
        self.args.dist_name = '1D_Gauss_mix'
        self.args.input_dim = 2

    def test_dynamics_fn(self):
        test_cases = [
            ('1D_Gauss_mix', 2),
            ('2D_Neal_funnel', 4),
            ('5D_illconditioned_Gaussian', 10),
            ('nD_Rosenbrock', 6),
            ('Allen_Cahn', 50),
            ('Elliptic', 100)
        ]

        for dist_name, input_dim in test_cases:
            with self.subTest(dist_name=dist_name):
                self.args.dist_name = dist_name
                self.args.input_dim = input_dim

                # test data
                z = tf.constant([[1.0] * input_dim], dtype=tf.float32)

                # calculate derivative
                derivatives = dynamics_fn(lambda x, args: functions(x, args), z, self.args)

                # check output shape
                self.assertEqual(derivatives.shape, z.shape)
                # if output range is reasonable
                self.assertTrue(tf.reduce_all(tf.math.is_finite(derivatives)))

    def test_traditional_leapfrog(self):
        # Basic test
        z0 = tf.constant([[1.0, 0.5]], dtype=tf.float32)
        t_span = [0.0, 1.0]
        n_steps = 10

        traj, derivatives = traditional_leapfrog(
            lambda x, args: functions(x, args),
            z0, t_span, n_steps, self.args
        )

        # output shape
        self.assertEqual(traj.shape, (n_steps + 1, z0.shape[0], z0.shape[1]))
        self.assertEqual(derivatives.shape, (n_steps + 1, z0.shape[0], z0.shape[1]))

        # Continuity of the trajcetory
        diff = tf.reduce_max(tf.abs(traj[1:] - traj[:-1]))
        self.assertLess(diff, 1.0)

        # different density and their threshold
        diff_thresholds = {
            '1D_Gauss_mix': 1.0,
            '2D_Neal_funnel': 2.0,
            '5D_illconditioned_Gaussian': 10.0,
            'nD_Rosenbrock': 5.0
        }

        # different density and # of inputs
        test_cases = [
            ('1D_Gauss_mix', 2),
            ('2D_Neal_funnel', 4),
            ('5D_illconditioned_Gaussian', 10),
            ('nD_Rosenbrock', 6)
        ]

        for dist_name, input_dim in test_cases:
            with self.subTest(dist_name=dist_name):
                self.args.dist_name = dist_name
                self.args.input_dim = input_dim

                # different shapes of input
                input_shapes = [
                    (tf.constant([[1.0] * input_dim], dtype=tf.float32), "batch_shape"),
                    (tf.constant([1.0] * input_dim, dtype=tf.float32), "single_shape")
                ]

                for z0_test, shape_type in input_shapes:
                    t_span = [0.0, 1.0]
                    n_steps = 10

                    # test int n_steps
                    traj, derivatives = traditional_leapfrog(
                        lambda x, args: functions(x, args),
                        z0_test, t_span, n_steps, self.args
                    )

                    expected_shape = (n_steps + 1, 1, input_dim) if shape_type == "batch_shape" \
                        else (n_steps + 1, 1, input_dim)
                    self.assertEqual(traj.shape, expected_shape)

                    # Continuity of the trajcetory
                    diff = tf.reduce_max(tf.abs(traj[1:] - traj[:-1]))
                    self.assertLess(diff, diff_thresholds[dist_name])

                    # Reasonable range of derivative
                    self.assertTrue(tf.reduce_all(tf.math.is_finite(derivatives)))

                    # Conservation of Hamilton
                    if dist_name in ['1D_Gauss_mix', '2D_Neal_funnel']:
                        initial_energy = functions(traj[0], self.args)
                        final_energy = functions(traj[-1], self.args)
                        energy_diff = tf.abs(final_energy - initial_energy)
                        self.assertLess(float(energy_diff), 0.1)

        # if n_steps is float
        n_steps_float = 10.0
        try:
            traj, derivatives = traditional_leapfrog(
                lambda x, args: functions(x, args),
                z0, t_span, n_steps_float, self.args
            )
            
            self.assertEqual(traj.shape, (int(n_steps_float) + 1, z0.shape[0], z0.shape[1]))
        except Exception as e:
            self.fail(f"Float n_steps raised unexpected error: {e}")

        # range of time span
        t_span_cases = [
            ([0.0, 1.0], None),  # normal timespan
            ([0.0, 1e3], None)  # large timespan
        ]

        for t_span_test, expected_error in t_span_cases:
            if expected_error is None:
                try:
                    traj, _ = traditional_leapfrog(
                        lambda x, args: functions(x, args),
                        z0, t_span_test, n_steps, self.args
                    )
                except Exception as e:
                    self.fail(f"t_span={t_span_test} raised unexpected error: {e}")
            else:
                with self.assertRaises(expected_error):
                    traditional_leapfrog(
                        lambda x, args: functions(x, args),
                        z0, t_span_test, n_steps, self.args
                    )

    def test_traditional_leapfrog_invalid_inputs(self):
        """leapfrog invalid input"""
        z0 = tf.constant([[1.0, 0.5]], dtype=tf.float32)

        # negative steps
        n_steps_negative = -1
        with self.assertRaises(tf.errors.InvalidArgumentError):
            traditional_leapfrog(
                lambda x, args: functions(x, args),
                z0, [0.0, 1.0], n_steps_negative, self.args
            )

        # zero steps
        n_steps_zero = 0
        with self.assertRaises(tf.errors.InvalidArgumentError):
            traditional_leapfrog(
                lambda x, args: functions(x, args),
                z0, [0.0, 1.0], n_steps_zero, self.args
            )



    def test_L2_loss(self):
        # test L2 correct
        u = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        v = tf.constant([[1.1, 2.1], [3.1, 4.1]])

        loss = L2_loss(u, v)

        # test if the loss is a scaler
        self.assertEqual(loss.shape, ())
        # test if the loss is > 0
        self.assertGreater(loss, 0)
        # expected loss
        expected_loss = ((1-1.1)**2 + (2-2.1)**2 + (3-3.1)**2 + (4-4.1)**2) / 4
        self.assertAlmostEqual(float(loss), expected_loss, places=6)

    def test_pickle_operations(self):
        # test pickle save and load
        test_data = {"test": "data"}

        # create temp file
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            # save the temp
            to_pickle(test_data, tmp.name)
            # load the temp
            loaded_data = from_pickle(tmp.name)

        self.assertEqual(test_data, loaded_data)

        os.unlink(tmp.name)

    def test_setup_logger(self):
        logger = setup_logger("test_logger")

        # check log type
        self.assertIsInstance(logger, logging.Logger)
        # check number of handlers = 2
        self.assertEqual(len(logger.handlers), 2)

        # close all handlers
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)

        # clear logs
        log_dir = Path('logs')
        if log_dir.exists():
            for log_file in log_dir.glob('test_logger_*.log'):
                log_file.unlink()
            log_dir.rmdir()

    def test_ess_computations(self):
        # different dimensions of input
        test_dims = [2, 4, 6, 8]

        for dim in test_dims:
            samples = tf.random.normal([1, 1000, dim])
            burn_in = 100

            # test ESS of each dimension
            ess_values = compute_ess(samples, burn_in)
            self.assertEqual(len(ess_values), dim // 2)

            # test average ESS
            avg_ess = compute_average_ess(samples, burn_in)
            self.assertIsInstance(avg_ess, float)
            self.assertGreater(avg_ess, 0)

        # test if input dim is odd
        odd_dim = 5
        samples_odd = tf.random.normal([1, 1000, odd_dim])

        with self.assertRaises(ValueError) as context:
            ess_values = compute_ess(samples_odd, burn_in)

        self.assertTrue("Input dimension must be even" in str(context.exception))

    def tearDown(self):
        tf.keras.backend.clear_session()


if __name__ == '__main__':
    unittest.main()
