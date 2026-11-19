"""Strict unary local minima use spin-flip adjacency, not raw state count."""
import unittest
import numpy as np
from word_energy.wikitext_four_spin_diagnostics import STATES,PRODUCTS,strict_local_maxima
class MinimaTest(unittest.TestCase):
 def test_flat_and_ferromagnet(self):
  logits=np.stack([np.zeros(16),PRODUCTS.sum(axis=1)])
  m=strict_local_maxima(logits);self.assertEqual(int(m[0].sum()),0);self.assertEqual(np.flatnonzero(m[1]).tolist(),[0,15])
if __name__=='__main__':unittest.main()
