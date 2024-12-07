import unittest
import tensorflow as tf
import numpy as np
import os
import sys
from pathlib import Path
from codes.functions import functions

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent


class TestTraditionalHMC(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up environment before all tests"""
        # Change to codes directory
        os.chdir(os.path.join(PROJECT_ROOT, 'codes'))
        # Add codes directory to Python path
        if str(PROJECT_ROOT / 'codes') not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT / 'codes'))

        # Import required modules after changing working directory
        global TraditionalHMC, get_args
        from codes.traditional_hmc import TraditionalHMC
        from codes.get_args import get_args

    def setUp(self):
        """Set up test environment"""
        # Configure test cases
        self.test_configs = [
            {
                'name': 'nD_Rosenbrock',
                'input_dim': 6,
                'settings': [
                    {'step_size': 0.01, 'trajectory_length': 1},
                    {'step_size': 0.05, 'trajectory_length': 5},
                ]
            },
            {
                'name': '2D_Neal_funnel',
                'input_dim': 4,
                'settings': [
                    {'step_size': 0.01, 'trajectory_length': 1},
                    {'step_size': 0.05, 'trajectory_length': 5},
                ]
            },
            {
                'name': '5D_illconditioned_Gaussian',
                'input_dim': 10,
                'settings': [
                    {'step_size': 0.01, 'trajectory_length': 1},
                    {'step_size': 0.05, 'trajectory_length': 5},
                ]
            },
            {
                'name': 'Allen_Cahn',
                'input_dim': 50,
                'settings': [
                    {'step_size': 0.01, 'trajectory_length': 1},
                    {'step_size': 0.05, 'trajectory_length': 5},
                ]
            }
        ]

    def get_modified_args(self, config, setting):
        """Get modified arguments

        Args:
            config: Dictionary containing model configuration
            setting: Dictionary containing HMC settings

        Returns:
            Modified args object
        """
        args = get_args()
        args.dist_name = config['name']
        args.input_dim = config['input_dim']
        args.hmc_step_size = setting['step_size']
        args.trajectory_length = setting['trajectory_length']
        # Use smaller sample size to speed up testing
        args.hmc_samples = 100
        args.num_burnin = 50
        args.num_chains = 2
        return args

    def test_hmc_initialization(self):
        """Test HMC initialization"""
        for config in self.test_configs:
            with self.subTest(distribution=config['name']):
                args = self.get_modified_args(config, config['settings'][0])
                hmc = TraditionalHMC(args)

                # Check initialized parameters
                self.assertEqual(hmc.state_dim, args.input_dim // 2)
                self.assertEqual(hmc.total_dim, args.input_dim)
                self.assertEqual(hmc.step_size, args.hmc_step_size)
                self.assertEqual(hmc.trajectory_length, args.trajectory_length)
                self.assertEqual(len(hmc.trajectories), 0)

    def test_hmc_sampling(self):
        """Test HMC sampling"""
        for config in self.test_configs:
            for setting in config['settings']:
                with self.subTest(distribution=config['name'],
                                  step_size=setting['step_size'],
                                  trajectory_length=setting['trajectory_length']):
                    args = self.get_modified_args(config, setting)
                    hmc = TraditionalHMC(args)

                    # Perform sampling
                    samples, acceptance = hmc.sample()

                    # Check output dimensions
                    expected_shape = (args.num_chains, args.hmc_samples, args.input_dim)
                    self.assertEqual(samples.shape, expected_shape)

                    # Check acceptance dimensions
                    expected_acceptance_shape = (args.num_chains,
                                                 args.hmc_samples - args.num_burnin)
                    self.assertEqual(acceptance.shape, expected_acceptance_shape)

                    # Check sample finiteness
                    self.assertTrue(tf.reduce_all(tf.math.is_finite(samples)))

                    # Check acceptance rates within valid range (0 to 1)
                    self.assertTrue(tf.reduce_all(acceptance >= 0))
                    self.assertTrue(tf.reduce_all(acceptance <= 1))

                    # Check mean acceptance rate within reasonable range (typically 0.2 to 0.9)
                    mean_acceptance = tf.reduce_mean(acceptance)
                    self.assertGreater(float(mean_acceptance), 0.1)

    def test_trajectory_storage(self):
        """Test trajectory storage functionality"""
        for config in self.test_configs:
            with self.subTest(distribution=config['name']):
                args = self.get_modified_args(config, config['settings'][0])
                hmc = TraditionalHMC(args)

                # Perform sampling
                samples, _ = hmc.sample()

                # Get stored trajectories
                trajectories = hmc.get_trajectories()

                # Check trajectory count (should equal non-burn-in sample count)
                expected_trajectory_count = args.num_chains * (args.hmc_samples - args.num_burnin)
                self.assertEqual(len(trajectories), expected_trajectory_count)

                # Check trajectory dimensions
                n_steps = int(args.trajectory_length / args.hmc_step_size)
                expected_trajectory_shape = (n_steps + 1, 1, args.input_dim)
                self.assertEqual(trajectories[0].shape, expected_trajectory_shape)

    def test_energy_conservation(self):
        """Test energy conservation"""
        for config in self.test_configs:
            with self.subTest(distribution=config['name']):
                args = self.get_modified_args(config, config['settings'][0])
                hmc = TraditionalHMC(args)

                # Perform sampling
                hmc.sample()

                # Get trajectories
                trajectories = hmc.get_trajectories()

                # Check energy change for each trajectory
                for traj in trajectories:
                    initial_state = tf.convert_to_tensor(traj[0], dtype=tf.float32)
                    final_state = tf.convert_to_tensor(traj[-1], dtype=tf.float32)

                    initial_energy = float(functions(initial_state, args))
                    final_energy = float(functions(final_state, args))

                    # Energy change should be relatively small
                    energy_diff = abs(final_energy - initial_energy)
                    self.assertLess(energy_diff, 1.0)

    def test_error_handling(self):
        """Test error handling"""
        config = self.test_configs[0]
        args = self.get_modified_args(config, config['settings'][0])

        # Test invalid step size
        args.hmc_step_size = -0.1
        with self.assertRaises(ValueError):
            TraditionalHMC(args)

        # Test invalid trajectory length
        args.hmc_step_size = 0.01
        args.trajectory_length = -1
        with self.assertRaises(ValueError):
            TraditionalHMC(args)

        # Test invalid dimension
        args.trajectory_length = 5
        args.input_dim = -1
        with self.assertRaises(ValueError):
            TraditionalHMC(args)

        # Test invalid input dimension (odd dimension)
        args.input_dim = 3  # Odd dimension will cause state_dim calculation error
        with self.assertRaises(ValueError):
            hmc = TraditionalHMC(args)
            hmc.sample()

        # Test extreme step size (causes numerical instability)
        args.input_dim = 6  # Restore valid dimension
        args.hmc_step_size = 1e10
        with self.assertRaises(tf.errors.InvalidArgumentError):
            hmc = TraditionalHMC(args)
            hmc.sample()

        # Test invalid sample count
        args.hmc_step_size = 0.01  # Restore normal step size
        args.hmc_samples = 0
        with self.assertRaises(ValueError):
            hmc = TraditionalHMC(args)
            hmc.sample()

        # Test burn-in greater than sample count
        args.hmc_samples = 10
        args.num_burnin = 20
        with self.assertRaises(ValueError):
            hmc = TraditionalHMC(args)
            hmc.sample()

    def tearDown(self):
        """Clean up test environment"""
        tf.keras.backend.clear_session()


if __name__ == '__main__':
    unittest.main()