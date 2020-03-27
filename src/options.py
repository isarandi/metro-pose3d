import argparse


def get_parser():
    parser = argparse.ArgumentParser(description='MeTRo-Pose3D', allow_abbrev=False)
    # Essentials
    parser.add_argument('--file', type=open, action=ParseFromFileAction)
    parser.add_argument('--logdir', type=str, default='default_logdir',
                        help='Directory for saving data about the experiment, including '
                             'checkpoints, logs, results, tensorboard event files etc. Can be a '
                             'relative path, which is then appended to a default location.')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers to run.')

    # Task options (what to do)
    parser.add_argument('--train', action=YesNoAction, help='Train the model.')
    parser.add_argument('--test', action=YesNoAction, help='Test the model.')
    parser.add_argument('--export-file', type=str, help='Export filename.')

    # Monitoring options
    parser.add_argument('--gui', action=YesNoAction,
                        help='Create graphical user interface for visualization.')
    parser.add_argument('--hook-seconds', type=float, default=15,
                        help='How often to call log, imshow and summary hooks.')
    parser.add_argument('--print-log', action=YesNoAction,
                        help='Print the log to the standard output (besides saving to file).')
    parser.add_argument('--tensorboard', action=YesNoAction, default=True,
                        help='Apply augmentations to test images.')

    # Loading and input processing options
    parser.add_argument('--load-path', type=str, default=None,
                        help='Path of model checkpoint to load in the beginning.')
    parser.add_argument('--checkpoint-dir', type=str, default=None,
                        help='Directory path of model checkpoints.')
    parser.add_argument('--init', type=str, default='pretrained',
                        help='How to initialize the weights: "scratch" or "pretrained".')
    parser.add_argument('--init-path', type=str, default=None,
                        help="""Path of the pretrained checkpoint to initialize from once
                        at the very start of training (i.e. not when resuming!).
                        To restore for resuming an existing training use the --load-path option.""")
    parser.add_argument('--proc-side', type=int, default=256,
                        help='Side length of image as processed by network.')
    parser.add_argument('--geom-aug', action=YesNoAction, default=True,
                        help='Training data augmentations such as rotation, scaling, translation '
                             'etc.')
    parser.add_argument('--test-aug', action=YesNoAction,
                        help='Apply augmentations to test images.')
    parser.add_argument('--rot-aug', type=float,
                        help='Rotation augmentation in degrees.', default=20)
    parser.add_argument('--scale-aug-up', type=float,
                        help='Scale augmentation in percent.', default=25)
    parser.add_argument('--scale-aug-down', type=float,
                        help='Scale augmentation in percent.', default=25)
    parser.add_argument('--shift-aug', type=float,
                        help='Shift augmentation in percent.', default=10)
    parser.add_argument('--test-subjects', type=str, default=None,
                        help='Test subjects.')
    parser.add_argument('--valid-subjects', type=str, default=None,
                        help='Validation subjects.')
    parser.add_argument('--train-subjects', type=str, default=None,
                        help='Training subjects.')

    parser.add_argument('--train-on', type=str, default='train',
                        help='Training part.')
    parser.add_argument('--validate-on', type=str, default='valid',
                        help='Validation part.')
    parser.add_argument('--test-on', type=str, default='test',
                        help='Test part.')

    # Training options
    parser.add_argument('--epochs', type=float, default=0,
                        help='Number of training epochs, 0 means unlimited.')
    parser.add_argument('--dtype', type=str, default='float16',
                        help='The floating point type to use for computations.')
    parser.add_argument('--validate-period', type=float, default=None,
                        help='Periodically validate during training, every this many epochs.')

    # Optimizer options
    parser.add_argument('--optimizer', type=str, default='adamw')
    parser.add_argument('--weight-decay', type=float, default=3e-3)
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='Learning rate of the optimizer.')
    parser.add_argument('--stretch-schedule', type=float, default=1)
    parser.add_argument('--stretch-schedule2', type=float, default=1)
    parser.add_argument('--epsilon', type=float, default=1e-8,
                        help='Epsilon for the Adam optimizer (called epsilon-hat in the paper).')

    # Test options
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--batch-size-test', type=int, default=110)
    parser.add_argument('--multiepoch-test', action=YesNoAction)
    parser.add_argument('--data-format', type=str, default='NCHW',
                        help='Data format used internally. NCHW is faster than NHWC.')

    parser.add_argument('--stride-train', type=int, default=32)
    parser.add_argument('--stride-test', type=int, default=4)

    parser.add_argument('--max-unconsumed', type=int, default=64)
    parser.add_argument('--seed', type=int, default=1)

    parser.add_argument('--port', type=int, default=6023)
    parser.add_argument('--comment', type=str, default=None)

    parser.add_argument('--dataset2d', type=str, default='mpii',
                        action=HyphenToUnderscoreAction)

    parser.add_argument('--dataset', type=str, default='h36m',
                        action=HyphenToUnderscoreAction)
    parser.add_argument('--architecture', type=str, default='resnet_v2_50',
                        action=HyphenToUnderscoreAction,
                        help='Architecture of the predictor network.')
    parser.add_argument('--result-path', type=str, default=None)
    parser.add_argument('--depth', type=int, default=8,
                        help='Number of voxels along the z axis for volumetric prediction')
    parser.add_argument('--train-mixed', action=YesNoAction, default=True)
    parser.add_argument('--batch-size-2d', type=int, default=32)

    parser.add_argument('--centered-stride', action=YesNoAction, default=True)
    parser.add_argument('--box-size-mm', type=float, default=2200)
    parser.add_argument('--universal-skeleton', action=YesNoAction)
    parser.add_argument('--shift-aug-by-rot', action=YesNoAction, default=True,
                        help='Apply the translation augmentation by out-of-plane camera rotation.')

    parser.add_argument('--partial-visibility', action=YesNoAction)
    parser.add_argument('--init-logits-random', action=YesNoAction, default=True)

    parser.add_argument('--loss2d-factor', type=float, default=0.1)
    parser.add_argument('--tdhp-to-mpii-shift-factor', type=float, default=0.2)
    parser.add_argument('--scale-recovery', type=str, default='metro')
    parser.add_argument('--bone-length-dataset', type=str)

    parser.add_argument('--occlude-aug-prob', type=float, default=0.7)
    parser.add_argument('--background-aug-prob', type=float, default=0)
    parser.add_argument('--color-aug', action=YesNoAction, default=True)
    return parser


class ParseFromFileAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        with values as f:
            lines = f.read().splitlines()
            args = [f'--{line}' for line in lines if line and not line.startswith('#')]
            parser.parse_args(args, namespace)


class HyphenToUnderscoreAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values.replace('-', '_'))


class YesNoAction(argparse.Action):
    def __init__(self, option_strings, dest, default=False, required=False, help=None):
        positive_opts = option_strings
        if not all(opt.startswith('--') for opt in positive_opts):
            raise ValueError('Yes/No arguments must be prefixed with --')
        if any(opt.startswith('--no-') for opt in positive_opts):
            raise ValueError(
                'Yes/No arguments cannot start with --no-, the --no- version will be '
                'auto-generated')

        negative_opts = ['--no-' + opt[2:] for opt in positive_opts]
        opts = [*positive_opts, *negative_opts]
        super().__init__(
            opts, dest, nargs=0, const=None, default=default, required=required, help=help)

    def __call__(self, parser, namespace, values, option_string=None):
        if option_string.startswith('--no-'):
            setattr(namespace, self.dest, False)
        else:
            setattr(namespace, self.dest, True)
