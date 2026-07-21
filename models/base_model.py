import os
import torch
from collections import OrderedDict
from abc import ABC, abstractmethod
from . import networks


class BaseModel(ABC):
    """This class is an abstract base class (ABC) for models.
    To create a subclass, you need to implement the following five functions:
        -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, opt).
        -- <set_input>:                     unpack data from dataset and apply preprocessing.
        -- <forward>:                       produce intermediate results.
        -- <optimize_parameters>:           calculate losses, gradients, and update network weights.
        -- <modify_commandline_options>:    (optionally) add model-specific options and set default options.
    """

    def __init__(self, opt):
        """Initialize the BaseModel class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions

        When creating your custom class, you need to implement your own initialization.
        In this fucntion, you should first call <BaseModel.__init__(self, opt)>
        Then, you need to define four lists:
            -- self.loss_names (str list):          specify the training losses that you want to plot and save.
            -- self.model_names (str list):         specify the images that you want to display and save.
            -- self.visual_names (str list):        define networks used in our training.
            -- self.optimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.
        """
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)  # save all the checkpoints to save_dir
        if opt.preprocess != 'scale_width':  # with [scale_width], input images might have different sizes, which hurts the performance of cudnn.benchmark.
            torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')  # allow tf32 matmuls, as the GAN project does
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.optimizers = []
        self.image_paths = []
        self.metric = 0  # used for learning rate policy 'plateau'

    @staticmethod
    def dict_grad_hook_factory(add_func=lambda x: x):
        saved_dict = dict()

        def hook_gen(name):
            def grad_hook(grad):
                saved_vals = add_func(grad)
                saved_dict[name] = saved_vals
            return grad_hook
        return hook_gen, saved_dict

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new model-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    @abstractmethod
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): includes the data itself and its metadata information.
        """
        pass

    @abstractmethod
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        pass

    @abstractmethod
    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        pass

    def setup(self, opt):
        """Load and print networks; create schedulers

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        if self.isTrain:
            self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
        if not self.isTrain or opt.continue_train:
            load_suffix = opt.epoch
            self.load_networks(load_suffix)
        if self.isTrain and opt.continue_train:
            self._restore_training_state()

        self.print_networks(opt.verbose)

    def parallelize(self):
        if len(self.opt.gpu_ids) == 0:  # CPU mode: nothing to parallelize
            return
        if self.opt.compile:
            # Conv2dWeightModulate.forward is one code object shared by every
            # modulated conv, so it legitimately specialises once per (channel
            # width x batch size) combination; the default limit of 8 trips
            # mid-run and silently drops the function back to eager. The
            # accumulated limit is the same budget summed over all functions
            # (cache_size_limit / accumulated_cache_size_limit are the
            # pre-2.6 names for these knobs)
            torch._dynamo.config.recompile_limit = 64
            torch._dynamo.config.accumulated_recompile_limit = 512
            # never fall back to eager silently: uncompiled stretches have hung
            # the GPU under WSL2 (Windows TDR), so fail loudly instead
            torch._dynamo.config.fail_on_recompile_limit_hit = True
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                if len(self.opt.gpu_ids) > 1:  # a single GPU runs the bare module
                    net = torch.nn.DataParallel(net, self.opt.gpu_ids)
                # netF builds its MLPs lazily and draws random patch indices per
                # call, which defeats graph capture; every other net is static
                if self.opt.compile and name != 'F':
                    # fullgraph: error on any graph break rather than splicing
                    # eager regions into the compiled forward
                    net = torch.compile(net, fullgraph=True)
                setattr(self, 'net' + name, net)

    @staticmethod
    def _unwrap(net):
        """Undo the DataParallel / torch.compile wrappers, so saved state_dicts
        keep plain keys (no 'module.' or '_orig_mod.' prefixes) and load into
        wrapped and unwrapped networks alike."""
        if isinstance(net, torch.nn.DataParallel):
            net = net.module
        return getattr(net, '_orig_mod', net)

    def autocast(self):
        """torch.amp context for forward passes: bfloat16 when --mixed_precision
        is on and a GPU is in use, otherwise a no-op. bf16 shares fp32's exponent
        range, so no GradScaler is needed (fp16 would need one per optimizer and
        is notoriously unstable for GAN losses)."""
        enabled = self.opt.mixed_precision and self.device.type == 'cuda'
        return torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=enabled)

    def data_dependent_initialize(self, data):
        pass

    def eval(self):
        """Make models eval mode during test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()

    def test(self):
        """Forward function used in test time.

        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """
        with torch.no_grad(), self.autocast():
            self.forward()
            self.compute_visuals()

    def compute_visuals(self):
        """Calculate additional output images for visdom and HTML visualization"""
        pass

    def get_image_paths(self):
        """ Return image paths that are used to load current data"""
        return self.image_paths

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        for scheduler in self.schedulers:
            if self.opt.lr_policy == 'plateau':
                scheduler.step(self.metric)
            else:
                scheduler.step()

        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def get_current_visuals(self):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                loss = getattr(self, 'loss_' + name)
                errors_ret[name] = loss.detach().item() if isinstance(loss, torch.Tensor) else float(loss)
        return errors_ret

    def save_networks(self, epoch):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)
                unwrapped = self._unwrap(net)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(unwrapped.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(unwrapped.cpu().state_dict(), save_path)

    def save_training_state(self, next_epoch, total_iters, suffix='latest'):
        """Save the training position and optimizer/scheduler state alongside the
        network checkpoints, so --continue_train can resume where the run left off.

        Parameters:
            next_epoch (int)  -- the epoch a resumed run should start at
            total_iters (int) -- the iteration counter a resumed run should start from
            suffix (str/int)  -- file name prefix matching save_networks: '<suffix>_state.pth'
        """
        state = {
            'next_epoch': next_epoch,
            'total_iters': total_iters,
            'optimizers': [optimizer.state_dict() for optimizer in self.optimizers],
            'schedulers': [scheduler.state_dict() for scheduler in self.schedulers],
        }
        torch.save(state, os.path.join(self.save_dir, '%s_state.pth' % suffix))

    def load_training_state(self, suffix='latest'):
        """Read the training-state file saved alongside a checkpoint, if present.

        Returns (next_epoch, total_iters), or None when no state file exists (e.g.
        checkpoints predating state saving). The optimizer/scheduler states are kept
        aside and applied by setup(), once all optimizers and schedulers exist.
        """
        state_path = os.path.join(self.save_dir, '%s_state.pth' % suffix)
        if not os.path.exists(state_path):
            return None
        self._resume_state = torch.load(state_path, map_location='cpu', weights_only=True)
        return self._resume_state['next_epoch'], self._resume_state['total_iters']

    def _restore_training_state(self):
        """Apply the optimizer/scheduler states stashed by load_training_state()."""
        state = getattr(self, '_resume_state', None)
        if state is None:
            return
        if (len(state['optimizers']) != len(self.optimizers)
                or len(state['schedulers']) != len(self.schedulers)):
            print('warning: saved training state does not match the current optimizers; '
                  'resuming with fresh optimizer state')
        else:
            for optimizer, opt_state in zip(self.optimizers, state['optimizers']):
                optimizer.load_state_dict(opt_state)
            for scheduler, sched_state in zip(self.schedulers, state['schedulers']):
                scheduler.load_state_dict(sched_state)
                # re-derive the group LRs from the restored epoch position rather than
                # keeping the saved values, so a changed schedule (e.g. extending
                # n_epochs_decay to train a finished run further) takes effect immediately
                if hasattr(scheduler, 'lr_lambdas'):
                    for group, lr_lambda in zip(scheduler.optimizer.param_groups, scheduler.lr_lambdas):
                        group['lr'] = group['initial_lr'] * lr_lambda(scheduler.last_epoch)
        self._resume_state = None

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
               (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    def load_networks(self, epoch):
        """Load all the networks from the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                if self.opt.isTrain and self.opt.pretrained_name is not None:
                    load_dir = os.path.join(self.opt.checkpoints_dir, self.opt.pretrained_name)
                else:
                    load_dir = self.save_dir

                load_path = os.path.join(load_dir, load_filename)
                net = self._unwrap(getattr(self, 'net' + name))
                print('loading the model from %s' % load_path)
                state_dict = torch.load(load_path, map_location=self.device, weights_only=True)
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

                # patch InstanceNorm checkpoints prior to 0.4
                # for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                #    self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                net.load_state_dict(state_dict)

    def print_networks(self, verbose):
        """Print the total number of parameters in the network and (if verbose) network architecture

        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def generate_visuals_for_evaluation(self, data, mode):
        return {}
