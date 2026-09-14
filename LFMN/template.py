def set_template(args):
    # Set the templates here
    if args.template.find('jpeg') >= 0:
        args.data_train = 'DIV2K_jpeg'
        args.data_test = 'DIV2K_jpeg'
        args.epochs = 20
        args.decay = '100'

    if args.template.find('LFMN_paper') >= 0:
        args.model = 'LFMN'
        args.n_resblocks = 32
        args.n_feats = 256
        args.res_scale = 0.1

    if args.template.find('GAN') >= 0:
        args.epochs = 20
        args.lr = 5e-5
        args.decay = '150'
