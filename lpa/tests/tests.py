from unittest import TestCase
import lpa
import numpy as np
import sys

class TestLPA(TestCase):
    def test_simple_dataset(self):
        m = 2
        n = 10
        X = np.zeros([m,n])

        
        print X
        
        model = lpa.lpa.LPA(1)