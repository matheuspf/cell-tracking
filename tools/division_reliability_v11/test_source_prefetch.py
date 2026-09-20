"""Process-level ownership checks for optional source preparation overlap."""
from pathlib import Path
import json
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from . import source_prefetch as prefetch
from .common import Blocked


class OwnershipTests(unittest.TestCase):
    def test_c00_package_waits_and_releases(self):
        code='''
from pathlib import Path
import sys
from division_reliability_v11 import source_prefetch,packaging
source_prefetch.WORK=Path(sys.argv[1])
def build(*args):
    Path(sys.argv[1],'entered').write_text('built')
    return {'status':'built'}
packaging._build=build
print('ready',flush=True)
print(packaging.run('44b6',314159),flush=True)
'''
        with tempfile.TemporaryDirectory() as name,patch.object(prefetch,'WORK',Path(name)):
            child=None
            try:
                with prefetch.ownership('44b6',314159,exclusive=True):
                    child=subprocess.Popen([sys.executable,'-u','-c',code,name],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
                    self.assertEqual(child.stdout.readline().strip(),'ready')
                    with self.assertRaises(subprocess.TimeoutExpired):child.wait(timeout=.2)
                    self.assertFalse((Path(name)/'entered').exists())
                stdout,stderr=child.communicate(timeout=10)
                self.assertEqual(child.returncode,0,stderr)
                self.assertEqual((Path(name)/'entered').read_text(),'built')
                self.assertIn('built',stdout)
            finally:
                if child and child.poll() is None:child.kill();child.communicate()

    def test_second_owner_rejected_and_exception_releases(self):
        with tempfile.TemporaryDirectory() as name,patch.object(prefetch,'WORK',Path(name)):
            with self.assertRaisesRegex(RuntimeError,'injected'):
                with prefetch.ownership('44b6',314159,exclusive=True):
                    with self.assertRaises(Blocked):
                        with prefetch.ownership('44b6',314159,exclusive=True):pass
                    raise RuntimeError('injected')
            with prefetch.ownership('44b6',314159,exclusive=True):pass

    def test_head_package_is_independent_of_source_barrier(self):
        from . import packaging
        with tempfile.TemporaryDirectory() as name,patch.object(prefetch,'WORK',Path(name)):
            with prefetch.ownership('44b6',314159,exclusive=True):
                with patch.object(packaging,'_build',return_value={'identity':'existing-head'}) as build:
                    self.assertEqual(packaging.run('44b6',314159,'C01')['identity'],'existing-head')
                    build.assert_called_once_with('44b6',314159,'C01')

    def test_orphan_source_worker_finishes_before_package_entry(self):
        from . import packaging
        with tempfile.TemporaryDirectory() as name,patch.object(prefetch,'WORK',Path(name)):
            jobs=Path(name)/'controller/jobs';jobs.mkdir(parents=True)
            child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(.4)','division_reliability_v11-orphan-proof'])
            try:
                (jobs/'source.json').write_text(json.dumps(dict(source='44b6',seed=314159,status='running',stage='prepare-actions',pid=child.pid)))
                with patch.object(packaging,'_build',return_value={}) as build:
                    tick=time.monotonic();packaging.run('44b6',314159)
                    self.assertGreaterEqual(time.monotonic()-tick,.3)
                    self.assertIsNotNone(child.poll());build.assert_called_once()
            finally:
                if child.poll() is None:child.kill()
                child.wait()


if __name__=='__main__':unittest.main()
