"""Meaningful invariants for causal inference and prospective labels."""
import unittest,json,tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from fusion import feature_cube,normalized,Streaming,metrics
from simulation import generate,perturb,ROOT
from eventstudy import instances,split_inner

class ResearchInvariants(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config=json.loads((ROOT/"input/config.json").read_text())
    def test_frozen_partition_disjointness(self):
        groups=[set(self.config[k]) for k in ["train_seeds","calibration_seeds","threshold_seeds","evaluation_seeds"]]
        for i,a in enumerate(groups):
            for b in groups[i+1:]: self.assertFalse(a & b)
    def test_generator_repeatability(self):
        a=generate(6001,self.config);b=generate(6001,self.config)
        for k in a: np.testing.assert_array_equal(a[k],b[k])
    def test_feature_causality(self):
        d=generate(8001,self.config);x=d["x"].copy()
        f=feature_cube(x,d["u"],d["a"])
        x[:,20:]=1-x[:,20:]
        g=feature_cube(x,d["u"],d["a"])
        np.testing.assert_array_equal(f[:,:20],g[:,:20])
    def test_independent_sequence_reset(self):
        d=generate(8001,self.config)
        f=feature_cube(d["x"],d["u"],d["a"])
        one=feature_cube(d["x"][3:4],d["u"][3:4],d["a"][3:4])
        np.testing.assert_array_equal(f[3],one[0])
    def test_missing_values_are_normalized(self):
        x,u,a=normalized([1,1,1],[.1,.1,.1],[0,1,0])
        np.testing.assert_array_equal(x,[0,1,0]);np.testing.assert_array_equal(u,[1,.1,1])
    def test_future_target_never_enters_features(self):
        d=generate(8001,self.config)
        f=feature_cube(d["x"][:,:40],d["u"][:,:40],d["a"][:,:40])
        d["y"][:]=1-d["y"]
        g=feature_cube(d["x"][:,:40],d["u"][:,:40],d["a"][:,:40])
        np.testing.assert_array_equal(f,g)
    def test_paired_stress_preserves_labels(self):
        d=generate(8001,self.config)
        for scenario in self.config["scenarios"]:
            q=perturb(d,8001,scenario);np.testing.assert_array_equal(d["y"],q["y"])
    def test_empirical_features_exclude_future(self):
        d=pd.DataFrame({"onset":[0.,5.,6.,7.,20.],
             "value":["3210001","2310001","4111201","4311001","2131001"]})
        rows,_=instances(d,1,"SCMM",8.)
        self.assertEqual(rows[0]["y"],1)
        self.assertEqual(rows[0]["recent_incorrect_20s"],0)
        self.assertEqual(rows[0]["prior_collisions_60s"],0)
        self.assertEqual(rows[0]["lead_seconds"],2.)
    def test_empirical_right_censoring(self):
        d=pd.DataFrame({"onset":[0.,18.,20.],"value":["3210001","2310001","2131001"]})
        rows,censored=instances(d,1,"SCMM",8.)
        self.assertEqual(len(rows),0);self.assertEqual(censored,1)
    def test_two_hazards_can_share_future_outcome(self):
        d=pd.DataFrame({"onset":[0.,5.,7.,10.,25.],
             "value":["3210001","2310001","2320001","4311001","2131001"]})
        rows,_=instances(d,1,"SCMM",8.)
        self.assertEqual([r["y"] for r in rows],[1,1])
    def test_zero_delay_collision_not_a_future_target(self):
        d=pd.DataFrame({"onset":[0.,5.,5.,25.],"value":["3210001","2310001","4311001","2131001"]})
        rows,_=instances(d,1,"SCMM",8.);self.assertEqual(rows[0]["y"],0)
    def test_input_validation(self):
        with self.assertRaises(ValueError):normalized([np.nan,0,0],[0,0,0],[1,1,1])
        with self.assertRaises(ValueError):normalized([0,0,0],[0,0,0],[2,1,1])
    def test_metric_boundary_probabilities(self):
        r=metrics([0,1],[0,1],.5);self.assertEqual(r["AUROC"],1);self.assertEqual(r["ECE"],0)
    def test_portable_matches_fitted_batch(self):
        path=ROOT/"output/synthetic/uatif_h0.json"
        if not path.exists(): self.skipTest("requires completed simulation")
        model=json.loads(path.read_text());d=generate(8001,self.config)
        cube=feature_cube(d["x"][:1],d["u"][:1],d["a"][:1])
        from scipy.special import expit
        z=cube[0];s=((z-np.array(model["mean"]))/np.array(model["scale"]))@np.array(model["coefficient"])+model["intercept"]
        expected=expit(model["calibration"][0]*s+model["calibration"][1])
        predictor=Streaming(model)
        actual=[predictor.step(d["x"][0,t],d["u"][0,t],d["a"][0,t])["probability"] for t in range(len(z))]
        np.testing.assert_allclose(expected,actual,rtol=1e-12,atol=1e-12)
        self.assertEqual(predictor.step([0,0,0],[1,1,1],[0,0,0])["status"],"unavailable")
    def test_empirical_partitions_are_disjoint(self):
        p=ROOT/"output/events/participant_splits.json"
        if not p.exists():self.skipTest("requires completed event study")
        for r in json.loads(p.read_text()):
            groups=[set(r[k]) for k in ["fit_subjects","calibration_subjects","threshold_subjects","test_subjects"]]
            for i,a in enumerate(groups):
                for b in groups[i+1:]:self.assertFalse(a & b)
if __name__=="__main__":unittest.main()
