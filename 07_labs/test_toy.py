import unittest
from toy_world_model import collect_data, LinearWorldModel, State, rollout_errors
from toy_cem_planner import cem_plan, run_closed_loop

class TestTinyWorldModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = LinearWorldModel().fit(collect_data())

    def test_action_effect(self):
        s=State(0,0)
        self.assertGreater(self.model.predict(s,1).x, self.model.predict(s,-1).x)

    def test_learned_dynamics(self):
        self.assertAlmostEqual(self.model.coef_v[1], 0.92, delta=0.04)
        self.assertAlmostEqual(self.model.coef_v[2], 0.35, delta=0.04)

    def test_rollout_error(self):
        errs = rollout_errors(self.model, episodes=10)
        self.assertLess(errs[1], 0.2)

    def test_planner_improves_from_zero_control(self):
        final, _ = run_closed_loop(self.model, steps=22)
        self.assertLess(abs(final.x-3.0), 2.0)

if __name__ == '__main__':
    unittest.main()
