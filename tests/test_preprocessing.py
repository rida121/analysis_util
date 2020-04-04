import unittest
from analysis_util import preprocessing
from unittest import TestCase
from collections import namedtuple
import pandas as pd


class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_target_encoding(self):
        """ test method of s"""


        #'a' x:1, y:2
        #'b' x:2, y:1
        df = pd.DataFrame({
            'a':['x','y','x','y','x'],
            'b':['y','x','y','x','y'],
            'target':[1,2,1,2,1]
            })
        C = namedtuple("C", "msg method expected")

        candidates = [
            C(msg="mean test", method="mean", expected=
            pd.DataFrame({
                'a':['y','x'],
                'b':['x','y'],
                'enc_mean_a':[2,1],
                'enc_mean_b':[2,1],
                'target':[2,1]
                }, index=[3,4])),
            C(msg="median test", method="median", expected=
            pd.DataFrame({
                'a':['y','x'],
                'b':['x','y'],
                'enc_median_a':[2,1],
                'enc_median_b':[2,1],
                'target':[2,1]
                }, index=[3,4]))
        ]
        for c in candidates:
            with self.subTest(msg=c.msg):
                actual = preprocessing.target_encoding(
                    train_df=df.loc[0:2], 
                    test_df=df.loc[3:],
                    target_key='target', 
                    encoding_keys=['a', 'b'],
                    method=c.method)
                pd.testing.assert_frame_equal(actual,c.expected)

if __name__ == '__main__':
    unittest.main()