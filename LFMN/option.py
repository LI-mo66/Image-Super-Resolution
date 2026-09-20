import argparse
import os
import template

parser = argparse.ArgumentParser(description='LFMN')

parser.add_argument('--debug', action='store_true',
                    help='Enables debug mode')
parser.add_argument('--template', default='.',
                    help='You can set various templates in option.py')

# Hardware specifications
parser.add_argument('--n_threads', type=int, default=8,
                    help='number of threads for data loading')
parser.add_argument('--cpu', action='store_true',
                    help='use cpu only')
parser.add_argument('--n_GPUs', type=int, default=1,
                    help='number of GPUs')                  
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')

# Data specifications
parser.add_argument('--dir_data', type=str, default="/home/huanzhujie/datasets/",
                    help='dataset directory')
parser.add_argument('--dir_demo', type=str, default='../test',
                    help='demo image directory')
parser.add_argument('--data_train', type=str, default='DIV2K',
                    help='train dataset name')
parser.add_argument('--data_test', type=str, default='Set5',
                    help='test dataset name')
parser.add_argument('--data_range', type=str, default='1-800/801-810',
                    help='train/test data range')
parser.add_argument('--ext', type=str, default='sep',
                    help='dataset file extension')
parser.add_argument('--scale', type=str, default='2',
                    help='super resolution scale')
parser.add_argument('--patch_size', type=int, default=192,
                    help='output patch size')
parser.add_argument('--rgb_range', type=int, default=255,
                    help='maximum value of RGB')
parser.add_argument('--n_colors', type=int, default=3,
                    help='number of color channels to use')
parser.add_argument('--chop', action='store_true',
                    help='enable memory-efficient forward')
parser.add_argument('--no_augment', action='store_true',
                    help='do not use data augmentation')

# Model specifications
parser.add_argument('--model', default='LFMN',
                    help='model name')
parser.add_argument('--feedback_stages', type=str, default='3-5-7',
                    help='1-based stages using feedback correction (for LFMNFeedback)')
parser.add_argument('--feedback_mid', type=int, default=8,
                    help='hidden channels in each feedback correction branch')
parser.add_argument('--feedback_scale', type=float, default=0.1,
                    help='maximum magnitude of bounded feedback corrections')
parser.add_argument('--feedback_lr_mult', type=float, default=1.0,
                    help='learning-rate multiplier for feedback parameters')
parser.add_argument('--freq_scale', type=float, default=0.1,
                    help='maximum frequency-prior correction magnitude')
parser.add_argument('--freq_lr_mult', type=float, default=1.0,
                    help='learning-rate multiplier for frequency-prior parameters')
parser.add_argument('--rdsm_mid', type=int, default=4,
                    help='hidden channels in the shared RDSM demand router')
parser.add_argument('--rdsm_kernel', type=int, default=3,
                    help='odd spatial kernel size in the shared RDSM router')
parser.add_argument('--rdsm_scale', type=float, default=0.5,
                    help='maximum relative change to original SFML strength')
parser.add_argument('--rdsm_direct_scale', type=float, default=0.1,
                    help='fixed maximum modulation magnitude for direct RDSM-v2')
parser.add_argument('--rdsm_lr_mult', type=float, default=1.0,
                    help='learning-rate multiplier for RDSM-only parameters')
parser.add_argument('--stage_diff_lr_mult', type=float, default=1.0,
                    help='learning-rate multiplier for stage-difference gates')
parser.add_argument('--cross_window_lr_mult', type=float, default=1.0,
                    help='learning-rate multiplier for cross-window parameters')
parser.add_argument('--token_refine_iters', type=int, default=3,
                    help='test-time TAB prototype refinement iterations')

# Training-only reliability-gated cross-stage relational distillation (RGCRD).
parser.add_argument('--rgcrd_mode', type=str, default='off',
                    choices=('off', 'output', 'relation', 'full'),
                    help='off/B0, output/C1, ungated relation/M0, reliable relation/M1')
parser.add_argument('--rgcrd_teacher_repo', type=str, default='',
                    help='path to an external official SwinIR checkout')
parser.add_argument('--rgcrd_teacher_checkpoint', type=str, default='',
                    help='official SwinIR-M x4 DIV2K checkpoint')
parser.add_argument('--rgcrd_teacher_amp', action='store_true',
                    help='experimental teacher autocast; FP32 is the validated default')
parser.add_argument('--rgcrd_teacher_microbatch', type=int, default=0,
                    help='frozen-teacher microbatch size; 0 uses the student batch')
parser.add_argument('--rgcrd_lambda_output', type=float, default=0.1,
                    help='C1 output-distillation weight')
parser.add_argument('--rgcrd_lambda_rel', type=float, default=0.25,
                    help='M0/M1 stage-relation loss weight')
parser.add_argument('--rgcrd_lambda_evo', type=float, default=0.125,
                    help='M0/M1 cross-stage evolution loss weight')
parser.add_argument('--rgcrd_local_windows', type=str, default='4+8',
                    help='parameter-free local relation window sizes')
parser.add_argument('--rgcrd_global_grid', type=int, default=8,
                    help='pooled grid side for global relations')
parser.add_argument('--rgcrd_reliability_pixel', type=float, default=0.5,
                    help='pixel-fidelity share in the reliability error')
parser.add_argument('--rgcrd_reliability_low', type=float, default=0.25,
                    help='low-frequency share in the reliability error')
parser.add_argument('--rgcrd_reliability_grad', type=float, default=0.25,
                    help='gradient-direction share in the reliability error')
parser.add_argument('--rgcrd_reliability_temperature', type=float, default=0.25,
                    help='temperature of the normalized teacher-better gate')
parser.add_argument('--rgcrd_reliability_margin', type=float, default=0.0,
                    help='teacher advantage required before the gate exceeds 0.5')
parser.add_argument('--rgcrd_reliability_smooth', type=int, default=3,
                    help='odd LR-space averaging kernel for the reliability map')
parser.add_argument('--rgcrd_grad_diag_every', type=int, default=1,
                    help='epochs between first-batch gradient diagnostics; 0 disables')

parser.add_argument('--act', type=str, default='relu',
                    help='activation function')
parser.add_argument('--prior_update_mode', type=str, default='state',
                    choices=('shallow', 'state', 'change'),
                    help='source used for the mid-network shared-prior update')

parser.add_argument('--prior_proxy_mode', type=str, default='a1',
                    choices=('a1',),
                    help='anchored prior-evolution proxy variant')

parser.add_argument('--pre_train', type=str, default='',
                    help='pre-trained model directory')
parser.add_argument('--extend', type=str, default='.',
                    help='pre-trained model directory')
parser.add_argument('--n_resblocks', type=int, default=3,
                    help='number of residual blocks')
parser.add_argument('--n_feats', type=int, default=64,
                    help='number of feature maps')
parser.add_argument('--res_scale', type=float, default=1,
                    help='residual scaling')
parser.add_argument('--shift_mean', default=True,
                    help='subtract pixel mean from the input')
parser.add_argument('--dilation', action='store_true',
                    help='use dilated convolution')
parser.add_argument('--precision', type=str, default='single',
                    choices=('single', 'half'),
                    help='FP precision for test (single | half)')

# Training specifications
parser.add_argument('--reset', action='store_true',
                    help='reset the training')
parser.add_argument('--test_every', type=int, default=1000,
                    help='do test per every N batches')
parser.add_argument('--epochs', type=int, default=20,
                    help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=8,
                    help='input batch size for training')
parser.add_argument('--split_batch', type=int, default=1,
                    help='split the batch into smaller chunks')
parser.add_argument('--self_ensemble', action='store_true',
                    help='use self-ensemble method for test')
parser.add_argument('--test_only', action='store_true',
                    help='set this option to test the model')
parser.add_argument('--gan_k', type=int, default=1,
                    help='k value for adversarial loss')

# Optimization specifications
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('--decay', type=str, default='100',
                    help='learning rate decay type')
parser.add_argument('--scheduler', type=str, default='multistep',
                    choices=('multistep', 'cosine'),
                    help='learning-rate scheduler')
parser.add_argument('--eta_min', type=float, default=0.0,
                    help='minimum learning rate for cosine scheduler')
parser.add_argument('--scheduler_t_max', type=int, default=0,
                    help='cosine horizon; 0 uses --epochs')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='learning rate decay factor for step decay')
parser.add_argument('--optimizer', default='ADAM',
                    choices=('SGD', 'ADAM', 'RMSprop'),
                    help='optimizer to use (SGD | ADAM | RMSprop)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--betas', type=tuple, default=(0.9, 0.999),
                    help='ADAM beta')
parser.add_argument('--epsilon', type=float, default=1e-8,
                    help='ADAM epsilon for numerical stability')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight decay')
parser.add_argument('--gclip', type=float, default=0,
                    help='gradient clipping threshold (0 = no clipping)')

# Loss specifications
parser.add_argument('--loss', type=str, default='1*L1',
                    help='loss function configuration')
parser.add_argument('--skip_threshold', type=float, default='1e8',
                    help='skipping batch that has large error')


# Log specifications
parser.add_argument('--save', type=str, default='test',
                    help='file name to save')
parser.add_argument('--load', type=str, default='',
                    help='file name to load')
parser.add_argument('--experiment_root', type=str, default='../experiment/all_runs',
                    help='root directory for all experiment outputs')
parser.add_argument('--resume', type=int, default=0,
                    help='resume from specific checkpoint')
parser.add_argument('--save_models', action='store_true',
                    help='save all intermediate models')
parser.add_argument('--print_every', type=int, default=100,
                    help='how many batches to wait before logging training status')
parser.add_argument('--max_train_batches', type=int, default=0,
                    help='debug limit per epoch (0 runs the complete epoch)')
parser.add_argument('--save_results', action='store_true',
                    help='save output results')
parser.add_argument('--save_gt', action='store_true',
                    help='save low-resolution and high-resolution images together')
parser.add_argument('--save_per_image_metrics', action='store_true',
                    help='save full-precision per-image PSNR/SSIM for paired analysis')

args = parser.parse_args()
template.set_template(args)

args.scale = list(map(lambda x: int(x), args.scale.split('+')))
args.data_train = args.data_train.split('+')
args.data_test = args.data_test.split('+')

if args.epochs == 0:
    args.epochs = 1e8

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False

max_data_workers = min(16, max(1, (os.cpu_count() or 1) // 2))
if args.n_threads > max_data_workers:
    print(
        'Reducing n_threads from {} to {} to avoid excessive DataLoader workers.'.format(
            args.n_threads, max_data_workers
        )
    )
    args.n_threads = max_data_workers
