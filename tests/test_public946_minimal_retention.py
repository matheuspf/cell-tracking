import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from public946_minimal.common import read_json,save_arrays,sha,write_json
from public946_minimal.retention import compact
from public946_minimal.measurements import graph_edits


class RetentionTests(unittest.TestCase):
    def make_clip(self,root,arm='E05'):
        arrays=dict(coords=np.arange(12,dtype=np.int16).reshape(3,4),
            candidates=np.asarray([[0,1,.7,1]],dtype=np.float64),
            offsets=np.asarray([[.1,.2,.3]],dtype=np.float64),
            p_0=np.asarray([[.2,.7],[.8,.3]],dtype=np.float32),
            s_0=np.asarray([0,1],dtype=np.int64),t_0=np.asarray([2,3],dtype=np.int64))
        save_arrays(root/'evidence.npz',**arrays)
        write_json(root/'neural.json',dict(evidence_sha256=sha(root/'evidence.npz'),provenance='unchanged-model-source-input'))
        write_json(root/'complete.json',dict(arm=arm))
        return arrays

    def test_b0_native_probabilities_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.make_clip(root,'B0');before=sha(root/'evidence.npz')
            self.assertIsNone(compact(root))
            self.assertEqual(sha(root/'evidence.npz'),before)

    def test_exact_retention_and_interrupted_receipt_recovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);arrays=self.make_clip(root)
            def interrupted(path,value,**kwargs):
                if Path(path).name=='neural.json':
                    raise OSError('simulated interruption after atomic evidence replacement')
                return write_json(path,value,**kwargs)
            with patch('public946_minimal.retention.write_json',side_effect=interrupted):
                with self.assertRaises(OSError):
                    compact(root)
            compact(root)
            with np.load(root/'evidence.npz',allow_pickle=False) as result:
                self.assertNotIn('p_0',result.files)
                for key,value in arrays.items():
                    if key!='p_0':
                        np.testing.assert_array_equal(result[key],value,strict=True)
            receipt=read_json(root/'neural.json')
            self.assertEqual(receipt['evidence_sha256'],sha(root/'evidence.npz'))
            self.assertEqual(receipt['provenance'],'unchanged-model-source-input')
            before=sha(root/'evidence.npz');compact(root)
            self.assertEqual(sha(root/'evidence.npz'),before)

    def test_graph_edits_ignore_renumbering_and_retain_multiplicity(self):
        a=dict(nodes=np.asarray([[0,0,1,2,3],[1,1,2,3,4]]),edges=np.asarray([[0,1]]))
        b=dict(nodes=np.asarray([[8,1,2,3,4],[7,0,1,2,3]]),edges=np.asarray([[7,8]]))
        self.assertFalse(any(graph_edits(a,b).values()))
        b['nodes']=np.concatenate([b['nodes'],[[9,0,1,2,3]]])
        self.assertEqual(graph_edits(a,b)['nodes_added_exact'],1)


if __name__=='__main__':
    unittest.main()
