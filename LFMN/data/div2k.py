import os
from data import srdata

class DIV2K(srdata.SRData):
    def __init__(self, args, name='DIV2K', train=True, benchmark=False):
        data_range = [r.split('-') for r in args.data_range.split('/')]
        if train:
            data_range = data_range[0]
        else:
            if args.test_only and len(data_range) == 1:
                data_range = data_range[0]
            else:
                data_range = data_range[1]

        self.begin, self.end = list(map(lambda x: int(x), data_range))
        super(DIV2K, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )

    def _scan(self):
        names_hr, names_lr = super(DIV2K, self)._scan()
        # Validation images keep their official IDs (0801-0900), while the
        # scanned list is zero-based and contains only those 100 files.
        offset = 800 if not self.train and self.begin >= 801 else 0
        begin = self.begin - 1 - offset
        end = self.end - offset
        names_hr = names_hr[begin:end]
        names_lr = [n[begin:end] for n in names_lr]

        return names_hr, names_lr

    def _set_filesystem(self, dir_data):
        super(DIV2K, self)._set_filesystem(dir_data)
        split = 'train' if self.train else 'valid'
        self.dir_hr = os.path.join(self.apath, f'DIV2K_{split}_HR')
        self.dir_lr = os.path.join(self.apath, f'DIV2K_{split}_LR_bicubic')
        if self.input_large: self.dir_lr += 'L'

