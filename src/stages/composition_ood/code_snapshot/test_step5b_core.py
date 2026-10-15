import unittest
import numpy as np
import pandas as pd
from step5b_core import validate_matrix, apply_bins, feature_shift, composition_errors

class Guards(unittest.TestCase):
    def test_order_duplicates_and_nonfinite(self):
        d=pd.DataFrame({'a':[1.], 'b':[2.]})
        np.testing.assert_array_equal(validate_matrix(d,['a','b']),[[1,2]])
        for bad in [pd.DataFrame([[1,2]],columns=['a','a']), pd.DataFrame({'a':[np.inf],'b':[2]})]:
            with self.assertRaises((AssertionError,KeyError)): validate_matrix(bad,['a','b'])
        with self.assertRaises(AssertionError): validate_matrix(d,['a','a'])
    def test_quartile_ties_retained(self):
        s=pd.Series([0,0,1,2,3,4])
        b=apply_bins(s,[0,1,2,4])
        self.assertFalse(b.isna().any())
        self.assertEqual(str(b.iloc[0]),'Q1')
    def test_shift_known(self):
        x=pd.DataFrame({'v':[0.,2.]}); y=pd.DataFrame({'v':[2.,4.]})
        r=feature_shift(x,y,['v']).iloc[0]
        self.assertEqual(r.standardized_mean_shift,2.)
        self.assertEqual(r.fraction_outside_train_range,.5)

if __name__=='__main__': unittest.main()
