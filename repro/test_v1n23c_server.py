"""CPU-only controller/download/protocol tests; never invokes training or shutdown."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'repro'))
sys.path.insert(0,str(ROOT/'LFMN'))
import v1n23c_protocol as protocol
import run_v1n23c_screen_server as server
import prepare_v1n23c_teacher as teacher


class FakeResponse(io.BytesIO):
    def __init__(self, data, status, headers):
        super().__init__(data)
        self.status = status
        self.headers = headers


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.temp_root = ROOT/'experiment/unit_tests'
        self.temp_root.mkdir(parents=True,exist_ok=True)
    def test_command_fixed_protocol_and_no_v1_optimizer(self):
        cmd = protocol.command('n23c',Path('data'),Path('repo'),Path('weight'),Path('out/run'))
        tokens = cmd[3:]
        def value(name):
            return tokens[tokens.index(name)+1]
        self.assertEqual(value('--model'),'lfmn_v1n23c')
        for name, val in {'--epochs':'20','--scheduler_t_max':'150','--eta_min':'1e-6',
                          '--batch_size':'4','--n_threads':'8','--resume':'0',
                          '--rgcrd_lambda_output':'0.1'}.items():
            self.assertEqual(value(name),val)
        with self.assertRaises(ValueError):
            protocol.command('v1',Path('data'),Path('repo'),Path('weight'),Path('out'))

    def test_reference_config_numeric_formatting_not_protocol_change(self):
        with tempfile.TemporaryDirectory(dir=self.temp_root) as temp:
            path = Path(temp)/'config.txt'
            values = dict(protocol.EXPECTED,weight_decay='0.0',gclip='0.0',epochs='20',resume='0',load='')
            path.write_text('\n'.join(f'{k}: {v}' for k,v in values.items()),encoding='utf-8')
            protocol.validate_v1_config(path)
            values['scheduler_t_max'] = '20'
            path.write_text('\n'.join(f'{k}: {v}' for k,v in values.items()),encoding='utf-8')
            with self.assertRaises(ValueError):
                protocol.validate_v1_config(path)

    def test_shutdown_wrapper_detection_without_execution(self):
        with tempfile.TemporaryDirectory(dir=self.temp_root) as temp:
            path = Path(temp)/'shutdown'
            for content in (b'\x7fELFfake',b'#!/bin/bash\necho fake\n'):
                path.write_bytes(content)
                self.assertEqual(server.shutdown_command(path),[str(path)])
            path.write_bytes(b'echo fake\n')
            with patch.object(server.subprocess,'run') as run:
                self.assertEqual(server.shutdown_command(path),['/bin/bash',str(path)])
                run.assert_called_once_with(['/bin/bash','-n',str(path)],check=True)

    def test_resume_keeps_existing_bytes(self):
        with tempfile.TemporaryDirectory(dir=self.temp_root) as temp:
            part = Path(temp)/'asset.partial'
            part.write_bytes(b'ab')
            def opener(request,timeout):
                self.assertEqual(request.get_header('Range'),'bytes=2-')
                return FakeResponse(b'cd',206,{'Content-Range':'bytes 2-3/4','Content-Length':'2'})
            with contextlib.redirect_stdout(io.StringIO()):
                teacher.download_once('https://example.test/asset',part,total=4,opener=opener)
            self.assertEqual(part.read_bytes(),b'abcd')

    def test_non_resuming_server_preserves_partial(self):
        with tempfile.TemporaryDirectory(dir=self.temp_root) as temp:
            part = Path(temp)/'asset.partial'
            part.write_bytes(b'ab')
            with self.assertRaises(ValueError):
                teacher.download_once('https://example.test/asset',part,total=4,
                    opener=lambda *a,**k:FakeResponse(b'abcd',200,{'Content-Length':'4'}))
            self.assertEqual(part.read_bytes(),b'ab')

    def test_bad_range_preserves_partial(self):
        with tempfile.TemporaryDirectory(dir=self.temp_root) as temp:
            part = Path(temp)/'asset.partial'
            part.write_bytes(b'ab')
            with self.assertRaises(ValueError):
                teacher.download_once('https://example.test/asset',part,total=4,
                    opener=lambda *a,**k:FakeResponse(b'cd',206,{'Content-Range':'bytes 1-3/4'}))
            self.assertEqual(part.read_bytes(),b'ab')

    def test_partial_eof_is_resumable(self):
        with tempfile.TemporaryDirectory(dir=self.temp_root) as temp:
            part = Path(temp)/'asset.partial'
            with contextlib.redirect_stdout(io.StringIO()):
                teacher.download_once('https://example.test/asset',part,total=4,
                    opener=lambda *a,**k:FakeResponse(b'ab',200,{'Content-Length':'4'}))
            self.assertEqual(part.read_bytes(),b'ab')

    def test_zero_gradient_is_not_excused_as_rounding(self):
        import torch
        from v1n23c_startup import audit_updates
        net = torch.nn.Sequential(torch.nn.LayerNorm(3),torch.nn.Linear(3,1))
        weight = net[0].weight
        other = net[1].weight
        before = {'0.weight':weight.detach().clone(),'1.weight':other.detach().clone()}
        with torch.no_grad():
            other.add_(.01)
        weight.grad = torch.full_like(weight,3e-13)
        fake = SimpleNamespace(param_groups=[dict(params=[weight,other],lr=2e-4,betas=(.9,.999),eps=1e-8)],
            state={weight:dict(step=torch.tensor(2.),exp_avg=torch.full_like(weight,3e-14),
                               exp_avg_sq=torch.full_like(weight,9e-29))})
        changed,rounded = audit_updates(net,{'0.weight':weight,'1.weight':other},before,fake)
        self.assertEqual(changed,['1.weight'])
        self.assertIn('0.weight',rounded)
        weight.grad.zero_()
        with self.assertRaises(RuntimeError):
            audit_updates(net,{'0.weight':weight,'1.weight':other},before,fake)

    def test_dirty_checkout_stops_before_cuda_or_optimizer(self):
        output = ROOT/'experiment/all_runs/unit_dirty_never_created'
        argv = ['server','--collect-only','--data-root','missing','--output',str(output),'--run']
        with patch.object(sys,'argv',argv),patch.object(server,'checked',return_value=' M file'), \
             patch.object(server,'gpu_environment') as gpu,patch.object(server,'run_child') as child:
            with self.assertRaisesRegex(RuntimeError,'Dirty checkout'):
                server.main()
            gpu.assert_not_called()
            child.assert_not_called()

    def test_shutdown_requires_dedicated_instance(self):
        output = ROOT/'experiment/all_runs/unit_power_never_created'
        argv = ['server','--collect-only','--data-root','missing','--output',str(output),
                '--shutdown-on-success','--run']
        with patch.object(sys,'argv',argv),patch.object(server,'checked',return_value=''), \
             patch.object(server,'shutdown_command') as power:
            with self.assertRaisesRegex(ValueError,'dedicated'):
                server.main()
            power.assert_not_called()

    def test_audit_only_starts_no_child_and_creates_no_result(self):
        output = ROOT/'experiment/all_runs/unit_audit_never_created'
        argv = ['server','--collect-only','--data-root','missing','--output',str(output)]
        def checked(cmd,cwd=None):
            return '' if 'status' in cmd else protocol.TEACHER_COMMIT
        with patch.object(sys,'argv',argv),patch.object(server,'checked',side_effect=checked), \
             patch.object(server,'sha256',return_value=protocol.TEACHER_SHA), \
             patch.object(server,'gpu_environment',return_value={}), \
             patch.object(server,'data_fingerprint',return_value={}), \
             patch.object(server,'source_fingerprint',return_value={}), \
             patch.object(server,'teacher_fingerprint',return_value={}), \
             patch.object(server.shutil,'disk_usage',return_value=SimpleNamespace(free=10*1024**3)), \
             patch.object(server,'run_child') as child,contextlib.redirect_stdout(io.StringIO()):
            server.main()
            child.assert_not_called()
            self.assertFalse(output.exists())

    def test_summary_is_pending_and_checks_all_image_rows(self):
        import torch
        from summarize_v1n23c import summarize,metrics
        with tempfile.TemporaryDirectory(dir=self.temp_root) as temp:
            output = Path(temp)
            directory = output/'n23c'
            per = directory/'per_image_metrics'
            per.mkdir(parents=True)
            psnr,ssim = [],[]
            for epoch in range(1,21):
                rows = [dict(filename=f'{i:04d}',dataset='DIV2K',scale=4,
                             psnr=28+epoch*.01,ssim=.8) for i in range(801,901)]
                acc = []
                for metric in ('psnr','ssim'):
                    total = torch.tensor(0.,dtype=torch.float32)
                    for row in rows:
                        total += row[metric]
                    acc.append(float(total/100))
                psnr.append(acc[0]); ssim.append(acc[1])
                torch.save(rows,per/f'epoch_{epoch:04d}.pt')
            torch.save(torch.tensor(psnr).reshape(20,1,1),directory/'psnr_log.pt')
            torch.save(torch.tensor(ssim).reshape(20,1,1),directory/'ssim_log.pt')
            with contextlib.redirect_stdout(io.StringIO()):
                report = summarize(output,['n23c'])
            self.assertEqual(report['results']['n23c']['decision'],'PENDING_MATCHED_BASELINE')
            self.assertFalse((output/'summary.json').exists())
            rows[0]['filename'] = '0900'
            torch.save(rows,per/'epoch_0020.pt')
            with self.assertRaises(ValueError):
                metrics(directory)

    def test_scc_linear_reference_and_ffn_order(self):
        import torch
        from model.v1n23c_scc import SCC
        from model.lfmn import LRSA,patch_divide,patch_reverse
        from model.lfmn_v1n23c import CalibratedLRSA
        torch.set_num_threads(2)
        torch.manual_seed(17)
        for window in (8,16):
            scc = SCC(window)
            # Match production NEW Linear initialization, not uncalibrated defaults.
            for layer in scc.modules():
                if isinstance(layer,torch.nn.Linear):
                    torch.nn.init.trunc_normal_(layer.weight,std=.02)
                    if layer.bias is not None:
                        torch.nn.init.zeros_(layer.bias)
            self.assertTrue(torch.allclose(scc.position_bias(),scc.reference_bias(),atol=2e-6,rtol=2e-5))
        original = LRSA(48,36,96,4)
        module = CalibratedLRSA(original,32).eval()
        image = torch.randn(1,48,32,32)
        seen = []
        hook = module.layer[1].register_forward_pre_hook(lambda m,args:seen.append(args[0].detach().clone()))
        with torch.no_grad():
            module.correlation_gain.fill_(.1)
            module(image,16)
            patches,_,_ = patch_divide(image,14,16)
            b,n,c,h,w = patches.shape
            tokens = patches.permute(0,1,3,4,2).reshape(b*n,h*w,c)
            tokens = module.layer[0](tokens)+tokens
            patches = tokens.reshape(b,n,h,w,c).permute(0,1,4,2,3)
            local = patch_reverse(patches,image,14,16,normalize_overlap=False)
            raw, normalized = module.correction(image)
            expected = (local+.1*torch.tanh(module.correlation_gain)*normalized).flatten(2).transpose(1,2)
            self.assertTrue(torch.equal(seen[0],expected))
            self.assertTrue(torch.allclose(normalized.permute(0,2,3,1),
                module.context_norm_out(raw.permute(0,2,3,1))))
        hook.remove()


if __name__ == '__main__':
    unittest.main()
