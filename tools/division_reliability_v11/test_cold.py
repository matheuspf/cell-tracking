"""Negative controls for the post-freeze cold comparison contract."""
from pathlib import Path
import tempfile
import unittest
from .cold import compare
from .common import read,write,sha


class ColdComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.original=self.root/'original';self.cold=self.root/'cold'
        for path,name in ((self.original,'source_clip'),(self.cold,'renamed_clip')):
            path.mkdir()
            (path/'submission.csv').write_bytes(b'id,dataset,row_type\r\n0,'+name.encode()+b',node\r\n')
            (path/'graph.npz').write_bytes(b'graph-byte-control')
            write(path/'receipt.json',dict(dataset=name,package_identity='same-explicit-package',
                frames=100,frame_hashes={str(t):str(t) for t in range(100)},graph_hash='same-graph',
                cold=path==self.cold,shared_C00=False,
                guard=dict(installed_before_numerical=True,denied=0,network_denied=0,subprocess_denied=0),
                csv_sha256=sha(path/'submission.csv'),graph_sha256=sha(path/'graph.npz'),
                gpu_lease_seconds=1.,wall_seconds=2.,inference_implementation_sha256='code-control'))

    def result(self):return compare(self.original,self.cold,'source_clip','renamed_clip')

    def change(self,key,value):
        r=read(self.cold/'receipt.json');r[key]=value;write(self.cold/'receipt.json',r)

    def test_complete_renamed_contract_passes(self):
        r=self.result();self.assertEqual(r['status'],'passed');self.assertEqual(r['compared_csv_rows'],1)

    def test_identical_graph_and_csv_do_not_hide_changed_images(self):
        hashes=read(self.cold/'receipt.json')['frame_hashes'];hashes['49']='different';self.change('frame_hashes',hashes)
        r=self.result();self.assertTrue(r['graph_exact']);self.assertTrue(r['csv_bytes_differ_only_by_dataset_name'])
        self.assertEqual(r['failed_checks'],['raw_image_frames_exact'])

    def test_incomplete_frames_fail(self):
        self.change('frames',99);self.assertIn('complete_frame_population',self.result()['failed_checks'])

    def test_baseline_cache_is_not_cold(self):
        self.change('shared_C00',True);self.assertIn('image_only_cold_contract',self.result()['failed_checks'])

    def test_denied_worker_access_fails(self):
        guard=read(self.cold/'receipt.json')['guard'];guard['denied']=1;self.change('guard',guard)
        self.assertIn('worker_guard_passed',self.result()['failed_checks'])

    def test_equal_parsed_csv_with_different_bytes_fails(self):
        path=self.cold/'submission.csv';path.write_bytes(path.read_bytes().replace(b'\r\n',b'\n'))
        self.change('csv_sha256',sha(path));r=self.result()
        self.assertTrue(r['csv_exact_after_dataset_rename']);self.assertFalse(r['csv_bytes_differ_only_by_dataset_name'])
        self.assertEqual(r['status'],'mismatch')

    def test_wrong_package_fails_even_for_equal_graphs(self):
        self.change('package_identity','other-package');self.assertIn('explicit_package_exact',self.result()['failed_checks'])

    def test_changed_graph_artifact_fails(self):
        (self.cold/'graph.npz').write_bytes(b'changed');self.assertIn('cold_artifact_hashes_exact',self.result()['failed_checks'])


if __name__=='__main__':unittest.main()
