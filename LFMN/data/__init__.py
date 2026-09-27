from importlib import import_module
#from dataloader import MSDataLoader
from torch.utils.data import dataloader
from torch.utils.data import ConcatDataset
import torch

# This is a simple wrapper function for ConcatDataset
class MyConcatDataset(ConcatDataset):
    def __init__(self, datasets):
        super(MyConcatDataset, self).__init__(datasets)
        self.train = datasets[0].train

    def set_scale(self, idx_scale):
        for d in self.datasets:
            if hasattr(d, 'set_scale'): d.set_scale(idx_scale)

class Data:
    def __init__(self, args):
        self.loader_train = None
        if not args.test_only:
            datasets = []
            for d in args.data_train:
                module_name = d if d.find('DIV2K-Q') < 0 else 'DIV2KJPEG'
                m = import_module('data.' + module_name.lower())
                datasets.append(getattr(m, module_name)(args, name=d))

            # Decouple crop/augmentation seeds from candidate-specific model
            # initialization, which consumes a different amount of RNG state.
            train_generator = torch.Generator()
            train_generator.manual_seed(args.seed)
            replay_epochs = int(getattr(args, 'resume_data_epochs', 0))
            if replay_epochs < 0:
                raise ValueError('resume_data_epochs must be non-negative')
            if replay_epochs:
                if not args.load or int(args.resume) != replay_epochs:
                    raise ValueError(
                        'resume_data_epochs requires matching --load and --resume'
                    )
                # Each DataLoader iterator consumes the same generator draws
                # for its worker base seed and shuffled sampler regardless of
                # worker count. Replay those draws without decoding images so
                # the resumed iterator starts at the exact next epoch stream.
                replay_loader = dataloader.DataLoader(
                    range(len(MyConcatDataset(datasets))),
                    batch_size=args.batch_size,
                    shuffle=True,
                    num_workers=0,
                    generator=train_generator,
                )
                for _ in range(replay_epochs):
                    for _ in replay_loader:
                        pass
            self.loader_train = dataloader.DataLoader(
                MyConcatDataset(datasets),
                batch_size=args.batch_size,
                shuffle=True,
                pin_memory=not args.cpu,
                num_workers=args.n_threads,
                generator=train_generator,
            )

        self.loader_test = []
        for d in args.data_test:
            if d in ['Set5', 'Set14', 'B100', 'Urban100', 'Manga109', 'manga109','Fmaps']:
                m = import_module('data.benchmark')
                testset = getattr(m, 'Benchmark')(args, train=False, name=d)
            else:
                module_name = d if d.find('DIV2K-Q') < 0 else 'DIV2KJPEG'
                m = import_module('data.' + module_name.lower())
                testset = getattr(m, module_name)(args, train=False, name=d)

            self.loader_test.append(
                dataloader.DataLoader(
                    testset,
                    batch_size=1,
                    shuffle=False,
                    pin_memory=not args.cpu,
                    num_workers=args.n_threads,
                )
            )
