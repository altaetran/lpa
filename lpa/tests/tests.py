from unittest import TestCase
import lpa
import numpy as np
import sys

class TestLPA(TestCase):
    def test_numpy_seed(self):
        """
        If the seed reproduction is not possible, then tests based on randomized conditions cannot be
        verified.
        """
        m = 2
        n = 10
        np.random.seed(0)
        X = np.tile(np.arange(n), [m,1]) + np.random.randn(m,n)
        X_true = np.array(
            [[ 1.76405235,  1.40015721,  2.97873798,  5.24089320,  5.86755799, 4.02272212,
               6.95008842,  6.84864279,  7.89678115,  9.41059850, ],
             [ 0.14404357,  2.45427351,  2.76103773,  3.12167502,  4.44386323, 5.33367433,
               7.49407907,  6.79484174,  8.31306770,  8.14590426,]]
            )
        assert np.linalg.norm(X-X_true)<1e-4
        
    def test_simple_dataset(self):
        np.random.seed(0)

        m = 2
        n = 10
        X = np.tile(np.arange(n), [m,1]) + np.random.randn(m,n)

        n_components = 1
        model = lpa.LPA(n_components, verbose=True)
        model.fit(X)

        W = model.get_components()
        W_true = np.array([[-0.71685396],
                           [-0.69722335]])
        
        assert np.linalg.norm(W-W_true) < 1e-4
