import json
import math
import unittest
from pathlib import Path

from headroom import division_score, filtering_requirement, repaired_edge_jaccard, generate

A = 0.9024998206702552
J = 0.9000721819574569
S = 0.9117740142186423


class HeadroomTests(unittest.TestCase):
    def test_baseline(self):
        self.assertAlmostEqual(division_score(A,151,23,97)['score'],S)

    def test_division_repair(self):
        self.assertAlmostEqual(division_score(A,151,75,50)['score'],0.9398132535060761)

    def test_false_forks_only_ceiling(self):
        self.assertAlmostEqual(division_score(A,151,23,0)['score']-S,.005957594531083132)

    def test_no_divisions(self):
        r=division_score(A,0,0,0)
        self.assertIsNone(r['division_jaccard'])
        self.assertEqual(r['score'],A)

    def test_false_forks_without_gt(self):
        self.assertEqual(division_score(A,0,0,3)['division_jaccard'],0)

    def test_invalid_divisions(self):
        for args in [(A,151,152,0),(A,151,23,-1),(float('nan'),151,23,97),(-1,151,23,97),(A,151,True,0)]:
            with self.subTest(args=args),self.assertRaises(ValueError):
                division_score(*args)

    def test_filter_identity(self):
        r=filtering_requirement(J,A,1)
        self.assertAlmostEqual(r['required_uniform_tp_retention'],1)
        self.assertEqual(r['count_only_max_gain_under_assumptions'],0)

    def test_ten_percent_not_enough(self):
        r=filtering_requirement(J,A,.9,.024)
        self.assertFalse(r['feasible_by_retention_alone'])
        self.assertAlmostEqual(r['count_only_max_gain_under_assumptions'],.008757957948294735)

    def test_half_retention(self):
        r=filtering_requirement(J,A,.5,.024)
        self.assertAlmostEqual(r['required_uniform_tp_retention'],.9790869628877537)
        self.assertAlmostEqual(r['independent_endpoint_recall_approximation'],.9894882328192456)

    def test_exact_weighted_formula(self):
        # Distinct sample Jaccards and node ratios; unchanged sample denominators.
        samples=[(100,.8,.5),(300,.9,1.7)]
        w=sum(x[0] for x in samples)
        raw=sum(weight*j for weight,j,c in samples)/w
        adj=sum(weight*j*(1.1-.1*c) for weight,j,c in samples)/w
        r,u=.4,.98
        predicted=u*(adj+(1-r)*(1.1*raw-adj))
        direct=sum(weight*u*j*(1.1-.1*r*c) for weight,j,c in samples)/w
        self.assertAlmostEqual(predicted,direct)

    def test_invalid_filtering(self):
        for args in [(0,A,.5),(J,A,1.1),(J,A,-.1),(J,1.5,.5),(J,A,.5,-.01),(J,float('nan'),.5)]:
            with self.subTest(args=args),self.assertRaises(ValueError):
                filtering_requirement(*args)

    def test_association_repair(self):
        r=repaired_edge_jaccard(122201,6885,6682,1500)
        self.assertAlmostEqual(r['raw_edge_jaccard'],.9212991926594573)
        self.assertEqual(r['edge_tp']+r['edge_fn'],128883)

    def test_association_identity(self):
        self.assertAlmostEqual(repaired_edge_jaccard(122201,6885,6682,0)['raw_edge_jaccard'],J)

    def test_invalid_association(self):
        for args in [(2,1,1,2),(0,0,0,0),(2,1,1,.5),(2,-1,1,0)]:
            with self.subTest(args=args),self.assertRaises(ValueError):
                repaired_edge_jaccard(*args)

    def test_bundle_data_consistency(self):
        b=json.loads((Path(__file__).parent/'measured_baseline.json').read_text())
        result=generate(b)
        self.assertAlmostEqual(result['baseline_score'],S)
        json.dumps(result,allow_nan=False)

    def test_reject_changed_source_counts(self):
        b=json.loads((Path(__file__).parent/'measured_baseline.json').read_text())
        b['public_pooled']['edge_tp']+=1000
        with self.assertRaises(ValueError):generate(b)


if __name__=='__main__':
    unittest.main()
