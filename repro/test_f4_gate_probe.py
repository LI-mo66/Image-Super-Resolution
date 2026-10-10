"""Gate identity contracts; no training."""
from pathlib import Path
import unittest
import json
import tempfile
from unittest.mock import patch,Mock
import torch
import probe_f4_gates as probe

class Contracts(unittest.TestCase):
    def test_shutdown_guards_mocked_only(self):
        import shutdown_f4_gate_probe as stop
        probe.ROOT.joinpath('experiment').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=probe.ROOT/'experiment') as directory:
            root=Path(directory)
            report=dict(status='COMPLETE_F4_GATE_DIAGNOSTIC',ids=probe.IDS,modes=probe.MODES,
                        formal_forwards=240,optimizer_steps=0,state_unchanged=True,
                        checkpoint_unchanged=True,flags_restored=True,checkpoint_sha256=probe.SHA150)
            (root/'gate_report.json').write_text(json.dumps(report),encoding='utf-8')
            (root/'config.json').write_text(json.dumps(dict(status='completed',checkpoint='mock')),encoding='utf-8')
            with patch.object(stop,'sha',return_value=probe.SHA150),patch.object(stop.sys,'platform','linux'),patch.object(stop.os,'sync',create=True) as sync,patch.object(stop.subprocess,'run') as run:
                run.return_value=Mock(stdout='1234\n')
                with self.assertRaises(RuntimeError):stop.shutdown(root,True)
                self.assertEqual(run.call_count,1);sync.assert_not_called()
                run.reset_mock();run.return_value=Mock(stdout='')
                stop.shutdown(root,True)
                self.assertEqual(run.call_count,2);sync.assert_called_once()
                run.assert_called_with(['bash','-c','shutdown -h now'],check=True)
                report['formal_forwards']=239
                (root/'gate_report.json').write_text(json.dumps(report),encoding='utf-8')
                run.reset_mock()
                with self.assertRaises(ValueError):stop.shutdown(root,True)
                run.assert_not_called()
    def test_plan_fixed_and_disjoint_descriptive_splits(self):
        self.assertEqual(len(probe.IDS),24);self.assertEqual(len(set(probe.IDS)),24)
        self.assertEqual(probe.IDS[0],801);self.assertEqual(probe.IDS[-1],900)
        self.assertEqual(len(probe.MODES),10)
        self.assertFalse(set(probe.IDS[::2]) & set(probe.IDS[1::2]))
    def test_identity_preserves_residual_and_flag_exception_cleanup(self):
        from model.lfmnf4 import Net
        torch.set_num_threads(4);net=Net(scale=4).eval()
        gate=net.residual_gates[3]
        with torch.no_grad():gate.projection[-1].bias.fill_(.5)
        x=torch.ones(1,48,5,7);r=torch.ones_like(x)*2
        with probe.gate_control(net,'stage4_identity'):
            self.assertIs(gate(x,r),r)
            self.assertFalse(gate.enabled)
            self.assertTrue(all(g.enabled for i,g in enumerate(net.residual_gates) if i!=3))
        self.assertTrue(gate.enabled)
        with self.assertRaises(RuntimeError),probe.gate_control(net,'all_identity'):
            raise RuntimeError('controlled failure')
        self.assertTrue(all(g.enabled and not g._forward_hooks for g in net.residual_gates))
    def test_real_model_actual_OFF_and_state_contract(self):
        from model.lfmnf4 import Net
        torch.set_num_threads(4);net=Net(scale=4).eval()
        with torch.no_grad():
            for g in net.residual_gates:g.projection[-1].bias.fill_(.2)
        m,calls=probe.wrapper(net,'cpu');before=probe.verify(net,m,calls)
        self.assertEqual(before,probe.state_digest(net))
    def test_metric_consistency_and_paired_orientation(self):
        hr=torch.ones(1,3,32,32)*100;sr=hr+1
        score=probe.metrics(sr,hr)
        self.assertAlmostEqual(score['psnr_rgb'],-10*__import__('math').log10(score['quant_rgb_mse']))
        self.assertAlmostEqual(score['psnr_y'],-10*__import__('math').log10(score['quant_y_mse']))
        a=[dict(filename=str(i),**score) for i in range(4)]
        b=[dict(r,psnr_rgb=r['psnr_rgb']+.01,raw_rgb_mse=r['raw_rgb_mse']-.000001) for r in a]
        report=probe.paired(a,b)
        self.assertAlmostEqual(report['psnr_rgb']['mean_delta'],.01)
        self.assertEqual(report['psnr_rgb']['positive'],4)
        self.assertLess(report['raw_rgb_mse']['mean_delta'],0)
        b[1]['filename']='0'
        with self.assertRaises(ValueError):probe.paired(a,b)

if __name__=='__main__':unittest.main()
