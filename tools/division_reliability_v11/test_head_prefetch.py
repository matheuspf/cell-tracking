"""Ownership and failure boundaries for overlapping a later registered fit."""
from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from . import head_prefetch,source_prefetch,controller,readiness,report
from .common import Blocked,read,write


class HeadPrefetchTests(unittest.TestCase):
    def setUp(self):
        self.context=ExitStack();self.addCleanup(self.context.close)
        self.root=Path(self.context.enter_context(tempfile.TemporaryDirectory()))
        for module in (head_prefetch,source_prefetch):self.context.enter_context(patch.object(module,'WORK',self.root))
        self.context.enter_context(patch.object(head_prefetch,'RESULTS',self.root/'results'))
        self.context.enter_context(patch.object(readiness,'require_production'))
        self.context.enter_context(patch.object(report,'alive',side_effect=lambda pid:pid==99))
        self.worker=self.context.enter_context(patch.object(controller,'worker'))
        self.cell=dict(source='44b6',seed=314159)
        write(self.root/'results/execution_lock.json',dict(event_updates=4000,schedule=[dict(source='44b6',seed=20260918),self.cell]))
        write(self.root/'controller/pipeline.json',dict(controller_pid=99,cell=dict(source='44b6',seed=20260918)))
        self.prepared=self.root/'controller/source-prefetch/44b6/314159/progress.json'
        write(self.prepared,dict(status='complete'))
        self.census=self.root/'source_diagnostics/44b6/314159/summary.json'
        write(self.census,dict(census=dict(positive=1,negative=100,identity_groups=1)))

    def test_incomplete_source_never_starts_fit(self):
        write(self.prepared,dict(status='running'))
        with self.assertRaisesRegex(Blocked,'Complete source preparation'):head_prefetch.run(**self.cell)
        self.worker.assert_not_called()

    def test_main_queue_owns_reached_cell(self):
        write(self.root/'controller/pipeline.json',dict(controller_pid=99,cell=self.cell))
        with self.assertRaisesRegex(Blocked,'main queue has reached'):head_prefetch.run(**self.cell)
        self.worker.assert_not_called()

    def test_live_cell_worker_is_not_duplicated(self):
        write(self.root/'controller/jobs/other.json',dict(**self.cell,status='running',pid=99))
        with self.assertRaisesRegex(Blocked,'already active'):head_prefetch.run(**self.cell)
        self.worker.assert_not_called()

    def test_missing_identity_support_never_spends_on_heads(self):
        write(self.census,dict(census=dict(positive=1,negative=100,identity_groups=0)))
        with self.assertRaisesRegex(Blocked,'does not support'):head_prefetch.run(**self.cell)
        self.worker.assert_not_called()

    def test_registered_prefix_holds_barrier_and_releases(self):
        def execute(stage,source,seed):
            with self.assertRaises(Blocked):
                with source_prefetch.ownership(source,seed,exclusive=True):pass
            if stage=='train-compact':
                write(self.root/'fits/44b6/314159/compact/progress.json',dict(status='waiting_for_mining',step=2000))
        self.worker.side_effect=execute
        result=head_prefetch.run(**self.cell)
        self.assertEqual(result['compact_step'],2000)
        self.assertEqual([c.args[0] for c in self.worker.call_args_list],['fit-linear','train-compact'])
        with source_prefetch.ownership(**self.cell,exclusive=True):pass

    def test_fit_failure_releases_without_starting_compact(self):
        self.worker.side_effect=Blocked('injected linear failure')
        with self.assertRaisesRegex(Blocked,'injected linear'):head_prefetch.run(**self.cell)
        self.assertEqual(self.worker.call_count,1)
        self.assertEqual(read(self.root/'controller/head-prefetch/44b6/314159/progress.json')['status'],'failed')
        with source_prefetch.ownership(**self.cell,exclusive=True):pass


if __name__=='__main__':unittest.main()
