"""Probe contracts only. No training."""
import tempfile
import unittest
from pathlib import Path
import probe_modulation_amplitude as probe

class ProbeContracts(unittest.TestCase):
    def test_split_and_exclusions(self):
        self.assertEqual(len(probe.SELECTION),48);self.assertEqual(len(probe.TEST),48)
        self.assertFalse(set(probe.SELECTION)&set(probe.TEST))
        self.assertFalse(set(probe.SELECTION+probe.TEST)&probe.EXCLUDED)
    def test_selection_is_reference_on_tie(self):
        tables={v:[dict(psnr=30.,ssim=.8,raw_rgb_mse=.01)] for v in probe.GRID}
        self.assertEqual(probe.select_lambda(tables),1.)
        tables[.5][0]['psnr']=30.01
        self.assertEqual(probe.select_lambda(tables),.5)
    def test_pairing_and_decision(self):
        a=[dict(filename=str(i),psnr=30.,ssim=.8,raw_rgb_mse=.01) for i in range(48)]
        b=[dict(r,psnr=30.01) for r in a]
        stats=probe.paired(a,b)
        self.assertEqual(probe.decision(.5,stats),'PROBE_SUPPORTS_FURTHER_INVESTIGATION')
        self.assertEqual(probe.decision(1.,stats),'PROBE_INCONCLUSIVE')
        b[1]['filename']='0'
        with self.assertRaises(ValueError):probe.paired(a,b)
    def test_scale_hooks_identity_and_restore_cpu(self):
        import torch
        from model.lfmnf3 import Net
        torch.set_num_threads(4)
        net=Net(scale=4).eval()
        with torch.no_grad():
            for module in net.sfmls:
                module.current_correction[-1].weight.normal_(0,.05)
                module.current_correction[-1].bias.fill_(.01)
        before=probe.verify_hooks(net)
        x=torch.ones(1,3,32,32)
        with self.assertRaises(RuntimeError),probe.scaled_correction(net,.5):raise RuntimeError('injected')
        self.assertEqual(before,probe.state_digest(net))
        self.assertTrue(all(not m.current_correction._forward_hooks for m in net.sfmls))

if __name__=='__main__':unittest.main()
