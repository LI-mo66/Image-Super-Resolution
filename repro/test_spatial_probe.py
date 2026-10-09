"""Spatial-control contracts only; no optimizer."""
import unittest
import probe_modulation_spatial as probe

class SpatialContracts(unittest.TestCase):
    def test_mean_and_original(self):
        import torch
        x=torch.arange(2*12*5*7,dtype=torch.float32).reshape(2,12,5,7)
        self.assertIs(probe.transform(x,'original',801,0),x)
        self.assertTrue(torch.equal(probe.transform(x,'spatial_mean',801,0),
                                    x.mean((-2,-1),keepdim=True).expand_as(x)))
    def test_shuffle_vectors_determinism_and_global_rng(self):
        import torch
        x=torch.arange(12*5*7,dtype=torch.float32).reshape(1,12,5,7)
        state=torch.get_rng_state().clone()
        a=probe.transform(x,'shuffle17',801,0);b=probe.transform(x,'shuffle17',801,0)
        self.assertTrue(torch.equal(a,b));self.assertTrue(torch.equal(state,torch.get_rng_state()))
        self.assertTrue(torch.equal(x.flatten(2).sort(-1).values,a.flatten(2).sort(-1).values))
        self.assertTrue(torch.equal(a[:,1]-a[:,0],x[:,1]-x[:,0]))
        self.assertFalse(torch.equal(a,x));self.assertFalse(torch.equal(a,probe.transform(x,'shuffle29',801,0)))
    def test_model_identity_and_exception_cleanup(self):
        import torch
        from model.lfmnf3 import Net
        torch.set_num_threads(4);net=Net(scale=4).eval()
        with torch.no_grad():
            for module in net.sfmls:module.current_correction[-1].weight.normal_(0,.05)
        before=probe.verify(net)
        with self.assertRaises(RuntimeError),probe.spatial_control(net,'shuffle17',801):
            raise RuntimeError('injected')
        self.assertEqual(before,probe.state_digest(net))
        self.assertTrue(all(not m.current_correction._forward_hooks for m in net.sfmls))
    def test_statistics_pairing(self):
        a=[dict(filename=str(i),psnr=30.,ssim=.8,raw_rgb_mse=.01) for i in range(48)]
        b=[dict(r,psnr=29.99) for r in a]
        self.assertAlmostEqual(probe.paired(a,b)['mean_delta'],-.01)

if __name__=='__main__':unittest.main()
