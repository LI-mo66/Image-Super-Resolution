"""CPU engineering check: observing a batch must not change a model step."""
import copy
import json
from pathlib import Path
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'LFMN'))
import torch
from model.lfmn import Net
from f2_150_train_entry import FirstBatchAudit


def main():
    torch.set_num_threads(1)
    torch.manual_seed(1)
    base=Net(scale=4);observed=copy.deepcopy(base)
    generator=torch.Generator().manual_seed(123)
    x=torch.rand(1,3,32,32,generator=generator)*255
    target=torch.nn.functional.interpolate(x,scale_factor=4,mode='bilinear',align_corners=False)
    class OneBatch:
        def __len__(self):return 1
        def __iter__(self):yield x,target,torch.tensor([0])
    outputs=[];losses=[]
    folder=ROOT/'experiment';folder.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='F2_CPU_OBSERVATION_',dir=folder) as temporary:
        wrapper=FirstBatchAudit(OneBatch(),Path(temporary));wrapper.epoch=1
        for model,loader in [(base,OneBatch()),(observed,wrapper)]:
            optimizer=torch.optim.Adam(model.parameters(),lr=2e-4,betas=(0.9,0.999),eps=1e-8)
            for lr,hr,_ in loader:
                optimizer.zero_grad();out=model(lr);loss=torch.nn.functional.l1_loss(out,hr)
                outputs.append(out.detach());losses.append(float(loss.detach()));loss.backward();optimizer.step()
        if not torch.equal(outputs[0],outputs[1]) or losses[0]!=losses[1]:
            raise AssertionError('Observation changed CPU step output or loss')
        if not all(torch.equal(v,observed.state_dict()[k]) for k,v in base.state_dict().items()):
            raise AssertionError('Observation changed parameter or buffer update')
        first=(Path(temporary)/'first_batch_hashes.csv').read_bytes()
        list(wrapper)  # Restart same epoch must preserve the original hash row.
        if (Path(temporary)/'first_batch_hashes.csv').read_bytes()!=first:
            raise AssertionError('Same first batch overwrote history')
    print(json.dumps(dict(passed=True,scope='synthetic CPU engineering step only; no quality result',
          outputs_equal=True,losses_equal=True,updates_equal=True,restart_hash_row_unchanged=True)))


if __name__=='__main__':main()
