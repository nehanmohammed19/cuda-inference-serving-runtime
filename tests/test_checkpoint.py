import json
import tempfile
from pathlib import Path
from dataclasses import asdict
import unittest
import numpy as np
from runtime.model import Config,toy_weights
from runtime.checkpoint import load_checkpoint


class CheckpointTests(unittest.TestCase):
    def test_local_roundtrip(self):
        from safetensors.numpy import save_file
        c=Config(); w=toy_weights(c)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp); (p/'config.json').write_text(json.dumps({'model_type':'llama',**asdict(c)}))
            save_file(w,str(p/'model.safetensors'))
            loaded,actual,_=load_checkpoint(p)
            self.assertEqual(loaded,c)
            for key in w: np.testing.assert_array_equal(w[key],actual[key])
            raw={'model_type':'llama',**asdict(c),'rope_scaling':{'factor':2}}
            (p/'config.json').write_text(json.dumps(raw))
            with self.assertRaises(ValueError): load_checkpoint(p)
