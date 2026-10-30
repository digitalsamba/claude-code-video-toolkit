import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from production import BuildError, load_state, plan_build, run_build, select_profile

ROOT = Path(__file__).resolve().parents[1]


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.project = Path(self.temp.name)
        self.plan = {'version': 1, 'tasks': [{'id': 'local', 'command': ['{python}', '-c',
            "from pathlib import Path; Path('result').write_text('done')"], 'outputs': ['result'], 'estimatedCostUsd': 2}]}

    def test_cap_blocks_before_any_state_or_output(self):
        with self.assertRaisesRegex(BuildError, 'Budget exceeded'):
            run_build(self.project, self.plan, ROOT, budget_usd=1)
        self.assertEqual(list(self.project.iterdir()), [])

    def test_unknown_costs_are_explicit_and_block_a_cap(self):
        del self.plan['tasks'][0]['estimatedCostUsd']
        preview = plan_build(self.project, self.plan, ROOT, budget_usd=5)
        self.assertEqual(preview['budget']['unknownTasks'], ['local'])
        self.assertFalse(preview['budget']['allowed'])
        with self.assertRaisesRegex(BuildError, 'unknown costs'):
            run_build(self.project, self.plan, ROOT, budget_usd=5)

    def test_prior_attempts_count_and_resuming_costs_zero(self):
        result = run_build(self.project, self.plan, ROOT, budget_usd=2)
        self.assertTrue(result['success'])
        preview = plan_build(self.project, self.plan, ROOT, budget_usd=2)
        self.assertEqual(preview['budget']['plannedEstimateUsd'], 0)
        self.assertTrue(preview['budget']['allowed'])
        with self.assertRaisesRegex(BuildError, 'Budget exceeded'):
            run_build(self.project, self.plan, ROOT, budget_usd=3, restart=True)
        attempt = load_state(self.project)['costHistory'][0]['id']
        cli = subprocess.run([sys.executable, str(ROOT / 'tools/budget.py'), '--project', str(self.project),
            '--attempt', attempt, '--actual-usd', '0.25', '--note', 'provider invoice'], capture_output=True, text=True)
        self.assertEqual(cli.returncode, 0, cli.stderr)
        self.assertEqual(json.loads(cli.stdout)['summary']['previousReservedUsd'], .25)
        self.assertTrue(run_build(self.project, self.plan, ROOT, budget_usd=3, restart=True)['success'])

    def test_failure_keeps_reservation(self):
        self.plan['tasks'][0]['command'] = ['{python}', '-c', 'raise SystemExit(1)']
        result = run_build(self.project, self.plan, ROOT, budget_usd=2)
        self.assertFalse(result['success'])
        self.assertEqual(load_state(self.project)['costHistory'][0]['status'], 'failed')
        with self.assertRaisesRegex(BuildError, 'Budget exceeded'):
            run_build(self.project, self.plan, ROOT, budget_usd=2)

    def test_invalid_amount_and_manifest_limit(self):
        self.plan['budgetUsd'] = 1
        self.assertFalse(plan_build(self.project, self.plan, ROOT)['budget']['allowed'])
        for amount in [-1, float('nan'), True]:
            with self.assertRaises(BuildError):
                plan_build(self.project, self.plan, ROOT, budget_usd=amount)

    def test_draft_final_profiles_change_only_selected_task_settings(self):
        self.plan['profiles'] = {'draft': {'tasks': {'local': {'estimatedCostUsd': .1}}}}
        draft = select_profile(self.plan, 'draft')
        self.assertEqual(draft['tasks'][0]['estimatedCostUsd'], .1)
        self.assertEqual(self.plan['tasks'][0]['estimatedCostUsd'], 2)
        self.assertTrue(plan_build(self.project, draft, ROOT, budget_usd=.1)['budget']['allowed'])
        with self.assertRaisesRegex(BuildError, 'Unknown production profile'):
            select_profile(self.plan, 'missing')
        self.plan['profiles']['draft']['tasks']['other'] = {}
        with self.assertRaisesRegex(BuildError, 'unknown task'):
            select_profile(self.plan, 'draft')


if __name__ == '__main__':
    unittest.main()
