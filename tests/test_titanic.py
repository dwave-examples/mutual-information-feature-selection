# Copyright 2019 D-Wave Systems Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import subprocess
import sys
import unittest

import numpy as np

from titanic import (prob, shannon_entropy, conditional_shannon_entropy,
                     mutual_information, conditional_mutual_information)


class TestTwoDimensionalCalcs(unittest.TestCase):
    """Verify entropy and mutual information calculations for two random variables."""
    @classmethod
    def setUpClass(cls):
        # Probability table for use in verification:
        cls.prob = np.array([[0.5, 0], [0.3, 0.2]])

        # Directly calculated values, for verification:
        cls.hxy = -0.5*np.log2(0.5) - 0.3*np.log2(0.3) -0.2*np.log2(0.2)
        cls.hx = -0.5*np.log2(0.5) - 0.5*np.log2(0.5)
        cls.hy = -0.8*np.log2(0.8) - 0.2*np.log2(0.2)

    def test_joint_shannon_entropy(self):
        result = shannon_entropy(self.prob)

        self.assertAlmostEqual(result, self.hxy)

    def test_marginal_shannon_entropy(self):
        self.assertAlmostEqual(shannon_entropy(np.sum(self.prob, axis=1)), self.hx)

        self.assertAlmostEqual(shannon_entropy(np.sum(self.prob, axis=0)), self.hy)

    def test_conditional_shannon_entropy(self):
        # H(X|Y) = H(X,Y) - H(Y)
        self.assertAlmostEqual(conditional_shannon_entropy(self.prob, 1), self.hxy - self.hy)
        # H(Y|X) = H(X,Y) - H(X)
        self.assertAlmostEqual(conditional_shannon_entropy(self.prob, 0), self.hxy - self.hx)

    def test_mutual_information(self):
        # I(X;Y) = H(X) - H(X|Y)
        # H(X|Y) = H(X,Y) - H(Y)
        expected = self.hx - (self.hxy - self.hy)
        # Note: I(X;Y) = I(Y;X)
        self.assertAlmostEqual(mutual_information(self.prob, 0), expected)
        self.assertAlmostEqual(mutual_information(self.prob, 1), expected)


class TestThreeDimensionalCalcs(unittest.TestCase):
    """Verify calculations with three random variables."""
    @classmethod
    def setUpClass(cls):
        cls.p = np.array([[[0.2, 0.0],
                       [0.1, 0.1]],
                      [[0.0, 0.3],
                       [0.25, 0.05]]])

    def test_conditional_shannon_entropy(self):
        p = self.p

        p_x0 = sum(sum(p[0, :, :]))  # p(x=0)
        p_x1 = sum(sum(p[1, :, :]))  # p(x=1)

        # Calculating conditional shannon entropy using the definition
        # sum [p(x,y,z) * log2 (p(x)/p(x,y,z))]
        expected = 0
        for y in range(2):
            for z in range(2):
                p_xyz0 = p[0, y, z]     # p(x=0, y, z)
                p_xyz1 = p[1, y, z]     # p(x=1, y, z)

                if p_xyz0 != 0:
                    expected += (p_xyz0 * np.log2(p_x0/p_xyz0))
                if p_xyz1 != 0:
                    expected += (p_xyz1 * np.log2(p_x1/p_xyz1))

        result = conditional_shannon_entropy(p, 0)
        self.assertAlmostEqual(result, expected)

    def test_conditional_mutual_information(self):
        # I(X;Y|Z) = H(X|Z) - H(X|Y,Z)

        # H(X|Z) = H(X,Z) - H(Z)
        pxz = np.sum(self.p, axis=1)
        pz = np.sum(pxz, axis=0)
        hx_given_z = shannon_entropy(pxz) - shannon_entropy(pz)

        # H(X|Y,Z) = H(X,Y,Z) - H(Y,Z)
        pyz = np.sum(self.p, axis=0)
        hx_given_yz = shannon_entropy(self.p) - shannon_entropy(pyz)

        cmi = hx_given_z - hx_given_yz
        self.assertAlmostEqual(conditional_mutual_information(self.p, 1, 2), cmi)
        # I(X;Y|Z) = I(Y;X|Z)
        self.assertAlmostEqual(conditional_mutual_information(self.p, 0, 2), cmi)

    def test_prob(self):
        data = np.array([[True, 0, 4],
                         [True, 2, 3],
                         [True, 1, 2],
                         [False, 0, 1],
                         [False, 0, 1]])
        multidim_prob = prob(data)    # multidimensional probabilities

        # Expected values
        expected_bins = (2, 3, 4)    # number of unique values per data column

        # Check shape
        self.assertEqual(multidim_prob.shape, expected_bins)

        # Check values
        # Note: two items in data are identical and should fall in the same bin,
        #   while the other three items are unique. Hence, we should expect
        #   a probability of 0.4 and three of 0.2.
        # Note2: we're doing an indirect check on the values because I'm not
        #   sure if the histogram will always produce the same binning order per
        #   axis.
        flat_prob = multidim_prob.ravel()
        self.assertEqual(np.sum(flat_prob), 1)   # probabilities sum to 1
        self.assertEqual(np.sum(flat_prob==0.4), 1)
        self.assertEqual(np.sum(flat_prob==0.2), 3)


class TestIntegration(unittest.TestCase):
    @unittest.skipIf(os.getenv('SKIP_INT_TESTS'), "Skipping integration test.")
    def test_integration(self):
        """Test integration of demo script."""
        # /path/to/demos/mutual-information-feature-selection/
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        demo_file = os.path.join(project_dir, 'titanic.py')

        output = subprocess.check_output([sys.executable, demo_file])
        output = output.decode('utf-8') # Bytes to str
        output = output.lower()

        self.assertIn('your plots are saved', output)
        self.assertNotIn('error', output)
        self.assertNotIn('warning', output)
