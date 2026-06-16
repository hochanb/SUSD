#!/usr/bin/env python3
import tempfile

import dowel_wrapper

assert dowel_wrapper is not None
import dowel

import wandb

import functools
import os
import platform
import torch.multiprocessing as mp
import torch.nn as nn

# from dotenv import load_dotenv
# load_dotenv()
wandb_api_key = os.getenv('WANDB_API_KEY')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

if 'mac' in platform.platform():
    pass
else:
    os.environ['MUJOCO_GL'] = 'egl'
    if 'SLURM_STEP_GPUS' in os.environ:
        os.environ['EGL_DEVICE_ID'] = os.environ['SLURM_STEP_GPUS']

import better_exceptions
import numpy as np

better_exceptions.hook()

import torch

from garage import wrap_experiment
from garage.experiment.deterministic import set_seed
from garage.torch.distributions import TanhNormal

from garagei.replay_buffer.path_buffer_ex import PathBufferEx
from garagei.experiment.option_local_runner import OptionLocalRunner
from garagei.sampler.option_multiprocessing_sampler import OptionMultiprocessingSampler
from garagei.torch.modules.with_encoder import WithEncoder, Encoder
from garagei.torch.modules.gaussian_mlp_module_ex import GaussianMLPTwoHeadedModuleEx, GaussianMLPIndependentStdModuleEx, GaussianMLPModuleEx
from garagei.torch.modules.parameter_module import ParameterModule
from garagei.torch.policies.policy_ex import PolicyEx
from garagei.torch.q_functions.continuous_mlp_q_function_ex import ContinuousMLPQFunctionEx
from garagei.torch.optimizers.optimizer_group_wrapper import OptimizerGroupWrapper
from garagei.torch.utils import xavier_normal_ex
from iod.susd import SUSD
from iod.metra import METRA
from iod.dads import DADS
from iod.dads_poe import DADSPoE, DADSPoEPolicyModule

from src.utils import get_exp_name, get_log_dir, make_env, make_q_function
from src.factorization import get_gaussian_module_construction, factorize_environment, PartitionedTrajectoryEncoder, module_cls_factory, PartitionedTrajectoryEncoderWithInputFactor0
from src.conf import SUSDFrankaKitchenConfig, SUSDParticle, SUSDGunner, SUSDEldenKitchen, SUSDHalfCheetahConfig, SUSDAntConfig


if os.environ.get('START_METHOD') is not None:
    START_METHOD = os.environ['START_METHOD']
else:
    START_METHOD = 'spawn'


# args = SUSDFrankaKitchenConfig()
# args = SUSDParticle()
# args = SUSDHalfCheetahConfig()
# args = SUSDGunner()
# args = SUSDEldenKitchen()


def _make_pretrain_config():
    env_name = os.environ.get('SUSD_ENV', 'ant').lower()
    if env_name == 'ant':
        return SUSDAntConfig()
    if env_name in ['half_cheetah', 'half-cheetah', 'halfcheetah']:
        return SUSDHalfCheetahConfig()
    if env_name in ['kitchen', 'kitchen_franka', 'franka_kitchen']:
        return SUSDFrankaKitchenConfig()
    raise NotImplementedError(
        f"SUSD_ENV={env_name!r} is not implemented in src/pretrain.py. "
        "Supported envs: ant, half_cheetah, kitchen."
    )


args = _make_pretrain_config()


def _env_override(name, current, cast=str):
    value = os.environ.get(name)
    if value is None or value == '':
        return current
    return cast(value)


def _apply_pretrain_env_overrides(args):
    method = os.environ.get('SUSD_TRAIN_METHOD', 'susd').lower()
    if method not in ['susd', 'metra', 'dads', 'dads_poe']:
        raise NotImplementedError(
            f"SUSD_TRAIN_METHOD={method!r} is not implemented in src/pretrain.py. "
            "This checkout currently supports pretraining for 'susd', 'metra', 'dads', and 'dads_poe'."
        )

    args.train_method = method
    default_env_tag = args.env.upper()
    if args.env == 'kitchen_franka':
        default_env_tag = 'KITCHEN'
    if method == 'metra':
        args.run_group = os.environ.get('SUSD_RUN_GROUP', f'METRA_{default_env_tag}')
        args.algo = 'metra'
        args.susd_ablation_mode = 0
    elif method == 'dads':
        args.run_group = os.environ.get('SUSD_RUN_GROUP', f'DADS_{default_env_tag}')
        args.algo = 'dads'
        args.susd_ablation_mode = 0
    elif method == 'dads_poe':
        args.run_group = os.environ.get('SUSD_RUN_GROUP', f'DADS_POE_{default_env_tag}')
        args.algo = 'dads_poe'
        args.susd_ablation_mode = 0
    else:
        args.run_group = os.environ.get('SUSD_RUN_GROUP', args.run_group)

    args.seed = _env_override('SUSD_SEED', args.seed, int)
    args.n_epochs = _env_override('SUSD_N_EPOCHS', args.n_epochs, int)
    args.max_path_length = _env_override('SUSD_MAX_PATH_LENGTH', args.max_path_length, int)
    args.n_parallel = _env_override('SUSD_N_PARALLEL', args.n_parallel, int)
    args.traj_batch_size = _env_override('SUSD_TRAJ_BATCH_SIZE', args.traj_batch_size, int)
    args.trans_optimization_epochs = _env_override('SUSD_TRANS_OPTIMIZATION_EPOCHS', args.trans_optimization_epochs, int)
    args.n_epochs_per_eval = _env_override('SUSD_N_EPOCHS_PER_EVAL', args.n_epochs_per_eval, int)
    args.n_epochs_per_log = _env_override('SUSD_N_EPOCHS_PER_LOG', args.n_epochs_per_log, int)
    args.n_epochs_per_save = _env_override('SUSD_N_EPOCHS_PER_SAVE', args.n_epochs_per_save, int)
    args.n_epochs_per_pt_save = _env_override('SUSD_N_EPOCHS_PER_PT_SAVE', args.n_epochs_per_pt_save, int)
    args.n_epochs_per_pkl_update = _env_override('SUSD_N_EPOCHS_PER_PKL_UPDATE', args.n_epochs_per_pkl_update, int)
    args.eval_record_video = _env_override('SUSD_EVAL_RECORD_VIDEO', args.eval_record_video, int)
    args.use_gpu = _env_override('SUSD_USE_GPU', args.use_gpu, int)
    args.sample_cpu = _env_override('SUSD_SAMPLE_CPU', args.sample_cpu, int)
    args.use_wandb = bool(_env_override('SUSD_USE_WANDB', int(args.use_wandb), int))
    return args


args = _apply_pretrain_env_overrides(args)


@wrap_experiment(log_dir=get_log_dir(args), name=get_exp_name(args)[0])
def run(ctxt=None):
    if args.use_wandb:
        if wandb_api_key:
            wandb_output_dir = tempfile.mkdtemp()
            wandb.init(project=args.wandb_project, entity=args.wandb_entity, group=args.run_group, name=get_exp_name()[0],
                    config=vars(args), dir=wandb_output_dir)

    dowel.logger.log('ARGS: ' + str(args))

    if args.n_thread is not None:
        torch.set_num_threads(args.n_thread)

    set_seed(args.seed)
    runner = OptionLocalRunner(ctxt)

    # args.resume = True
    # args.resume_path = "exp/DSD/sd000_1753356503_kitchen_franka_metra"
    if args.resume:
        dowel.logger.log(f"Resuming from checkpoint: {args.resume_path}")
        restored_train_args = runner.restore(
            from_dir=args.resume_path,
            make_env=functools.partial(make_env, args=args, max_path_length=args.max_path_length),
            from_epoch='last',
        )

        # set new saving arguments 
        runner._algo.n_epochs_per_pkl_update = 1000 # params
        runner._algo.n_epochs_per_save = 1000 # phi encoder
        runner._algo.n_epochs_per_pt_save = 1000 # option policy
        # runner._algo.n_epochs_per_log = 1 # save logs
        # runner._algo.n_epochs_per_eval = 1 # save eval
        # runner._algo._trans_optimization_epochs = 20
        # runner._algo.csd_logs = []
        # print(runner._algo.csd_logs)
        # exit()

        runner.train(n_epochs=restored_train_args.n_epochs, batch_size=restored_train_args.batch_size)
        return



    max_path_length = args.max_path_length
    contextualized_make_env = functools.partial(make_env, args=args, max_path_length=max_path_length)
    env = contextualized_make_env()

    example_ob = env.reset()

    if args.encoder:
        if hasattr(env, 'ob_info'):
            if env.ob_info['type'] in ['hybrid', 'pixel']:
                pixel_shape = env.ob_info['pixel_shape']
        else:
            pixel_shape = (64, 64, 3)
    else:
        pixel_shape = None

    device = torch.device('cuda' if args.use_gpu else 'cpu')
    master_dims = [args.model_master_dim] * args.model_master_num_layers

    if args.model_master_nonlinearity == 'relu':
        nonlinearity = torch.relu
    elif args.model_master_nonlinearity == 'tanh':
        nonlinearity = torch.tanh
    else:
        nonlinearity = None
    


    obs_dim = env.spec.observation_space.flat_dim
    action_dim = env.spec.action_space.flat_dim

    if args.encoder:
        def make_encoder(**kwargs):
            return Encoder(pixel_shape=pixel_shape, **kwargs)

        def with_encoder(module, encoder=None):
            if encoder is None:
                encoder = make_encoder()

            return WithEncoder(encoder=encoder, module=module)

        example_encoder = make_encoder()
        module_obs_dim = example_encoder(torch.as_tensor(example_ob).float().unsqueeze(0)).shape[-1]
    else:
        module_obs_dim = obs_dim

    
    partition_points = factorize_environment(args)
    if getattr(args, 'train_method', 'susd') == 'metra':
        partition_points = [0, obs_dim]
    args.N = len(partition_points) - 1
    dowel.logger.log(f'observation space: {obs_dim}, action space: {action_dim}, partition_points: {partition_points},  #factors: {args.N}')

    option_info = {
        'dim_option': args.dim_option,
    }

    policy_kwargs = dict(
        name='option_policy',
        option_info=option_info,
    )

    module_kwargs = dict(
        hidden_sizes=master_dims,
        layer_normalization=False,
    )
    if nonlinearity is not None:
        module_kwargs.update(hidden_nonlinearity=nonlinearity)


    module_cls = GaussianMLPTwoHeadedModuleEx
    module_kwargs.update(dict(
        max_std=np.exp(2.),
        normal_distribution_cls=TanhNormal,
        output_w_init=functools.partial(xavier_normal_ex, gain=1.),
        init_std=1.,
    ))

    option_dim_for_policy = args.dim_option if (
        getattr(args, 'train_method', 'susd') == 'metra' or args.algo in ['dads', 'dads_poe']
    ) else args.dim_option * args.N
    policy_q_input_dim = module_obs_dim + option_dim_for_policy  
    if args.algo == 'dads_poe':
        policy_module = DADSPoEPolicyModule(
            input_dim=policy_q_input_dim,
            output_dim=action_dim,
            dim_option=option_dim_for_policy,
            num_heads=getattr(args, 'poe_num_heads', 4),
            hidden_sizes=master_dims,
            hidden_nonlinearity=nonlinearity or torch.relu,
            temperature=getattr(args, 'poe_temperature', 1.0),
            init_std=1.0,
            max_std=np.exp(2.),
        )
    else:
        policy_module = module_cls(
            input_dim=policy_q_input_dim,
            output_dim=action_dim,
            **module_kwargs
        )

    if args.encoder:
        policy_module = with_encoder(policy_module)

    policy_kwargs['module'] = policy_module
    option_policy = PolicyEx(**policy_kwargs) #  π(a∣s,z), O + Nd -> A

    output_dim = args.dim_option

    if getattr(args, 'train_method', 'susd') == 'metra':
        te_module_cls, te_module_kwargs = module_cls_factory(
            args=args,
            master_dims=master_dims,
            nonlinearity=nonlinearity,
            input_dim=module_obs_dim,
            output_dim=output_dim,
        )
        traj_encoder = te_module_cls(**te_module_kwargs)
    elif args.susd_input_factor0:
        traj_encoder = PartitionedTrajectoryEncoderWithInputFactor0(
            args=args,
            partition_points=partition_points,
            master_dims = master_dims,
            nonlinearity=nonlinearity,
            output_dim=output_dim,
            module_cls_factory=module_cls_factory
        ) # π(z | s), O -> Nd

    else:
        traj_encoder = PartitionedTrajectoryEncoder(
            args=args,
            partition_points=partition_points,
            master_dims = master_dims,
            nonlinearity=nonlinearity,
            output_dim=output_dim,
            module_cls_factory=module_cls_factory
        ) # π(z | s), O -> Nd

    if args.encoder:
        if args.spectral_normalization:
            te_encoder = make_encoder(spectral_normalization=True)
        else:
            te_encoder = None
        traj_encoder = with_encoder(traj_encoder, encoder=te_encoder)


    module_cls, module_kwargs = get_gaussian_module_construction(
        args,
        hidden_sizes=master_dims,
        hidden_nonlinearity=nonlinearity or torch.relu,
        w_init=torch.nn.init.xavier_uniform_,
        input_dim=obs_dim,
        output_dim=obs_dim,
        min_std=1e-6,
        max_std=1e6,
    )
    if args.dual_dist == 's2_from_s':
        dist_predictor = module_cls(**module_kwargs)
    else:
        dist_predictor = None

    dual_lam = ParameterModule(torch.Tensor([np.log(args.dual_lam)]))


    sd_dim_option = args.dim_option
    skill_dynamics_obs_dim = obs_dim
    skill_dynamics_input_dim = skill_dynamics_obs_dim + sd_dim_option
    module_cls, module_kwargs = get_gaussian_module_construction(
        args,
        const_std=args.sd_const_std,
        hidden_sizes=master_dims,
        hidden_nonlinearity=nonlinearity or torch.relu,
        input_dim=skill_dynamics_input_dim,
        output_dim=skill_dynamics_obs_dim,
        min_std=0.3,
        max_std=10.0,
    )

    if args.algo in ['dads', 'dads_poe']:
        skill_dynamics = module_cls(**module_kwargs)
    else:
        skill_dynamics = None

    def _finalize_lr(lr):
        if lr is None:
            lr = args.common_lr
        else:
            assert bool(lr), 'To specify a lr of 0, use a negative value'
        if lr < 0.0:
            dowel.logger.log(f'Setting lr to ZERO given {lr}')
            lr = 0.0
        return lr

    optimizers = {
        'option_policy': torch.optim.Adam([
            {'params': option_policy.parameters(), 'lr': _finalize_lr(args.lr_op)},
        ]),
    }


    optimizers['dual_lam'] = torch.optim.Adam([{'params': dual_lam.parameters(), 'lr': _finalize_lr(args.dual_lr)}])


    if hasattr(traj_encoder, 'encoders'):
        for i, encoder in enumerate(traj_encoder.encoders):
            optimizers[f'traj_encoder_{i}'] = torch.optim.Adam(
                encoder.parameters(), lr=_finalize_lr(args.lr_te)
            )
    else:
        optimizers['traj_encoder'] = torch.optim.Adam(
            traj_encoder.parameters(), lr=_finalize_lr(args.lr_te)
        )


    if skill_dynamics is not None:
        optimizers.update({
            'skill_dynamics': torch.optim.Adam([
                {'params': skill_dynamics.parameters(), 'lr': _finalize_lr(args.lr_te)},
            ]),
        })

    if dist_predictor is not None:
        optimizers.update({
            'dist_predictor': torch.optim.Adam([
                {'params': dist_predictor.parameters(), 'lr': _finalize_lr(args.lr_op)},
            ]),
        })

    replay_buffer = PathBufferEx(capacity_in_transitions=int(args.sac_max_buffer_size), pixel_shape=pixel_shape)

    if args.algo in ['metra', 'dads', 'dads_poe']:
        if args.susd_q_function:
            q1_list = []
            for i in range(args.N):
                # start = partition_points[i]
                # end = partition_points[i + 1]
                # input_dim = end - start + args.dim_option
                input_dim = policy_q_input_dim # full state and skill
                q1_i = make_q_function(input_dim, action_dim, master_dims, nonlinearity, args.alpha)

                optimizers.update({
                    f'qf_{i}': torch.optim.Adam([
                        {'params': list(q1_i.parameters()), 'lr': _finalize_lr(args.sac_lr_q)},
                    ]),
                })
                q1_list.append(q1_i)


        qf1 = ContinuousMLPQFunctionEx(
            obs_dim=policy_q_input_dim,
            action_dim=action_dim,
            hidden_sizes=master_dims,
            hidden_nonlinearity=nonlinearity or torch.relu,
        )
        if args.encoder:
            qf1 = with_encoder(qf1)
        qf2 = ContinuousMLPQFunctionEx(
            obs_dim=policy_q_input_dim,
            action_dim=action_dim,
            hidden_sizes=master_dims,
            hidden_nonlinearity=nonlinearity or torch.relu,
        )
        if args.encoder:
            qf2 = with_encoder(qf2)
        log_alpha = ParameterModule(torch.Tensor([np.log(args.alpha)]))

        optimizers.update({
            'qf': torch.optim.Adam([
                {'params': list(qf1.parameters()) + list(qf2.parameters()), 'lr': _finalize_lr(args.sac_lr_q)},
            ]),
            'log_alpha': torch.optim.Adam([
                {'params': log_alpha.parameters(), 'lr': _finalize_lr(args.sac_lr_a)},
            ])
        })

    optimizer = OptimizerGroupWrapper(
        optimizers=optimizers,
        max_optimization_epochs=None,
    )

    algo_kwargs = dict(
        env_name=args.env,
        algo=args.algo,
        env_spec=env.spec,
        option_policy=option_policy,
        traj_encoder=traj_encoder,
        skill_dynamics=skill_dynamics,
        dist_predictor=dist_predictor,
        dual_lam=dual_lam,
        optimizer=optimizer,
        alpha=args.alpha,
        max_path_length=args.max_path_length,
        n_epochs_per_eval=args.n_epochs_per_eval,
        n_epochs_per_log=args.n_epochs_per_log, 
        n_epochs_per_tb=args.n_epochs_per_log, 
        n_epochs_per_save=args.n_epochs_per_save, 
        n_epochs_per_pt_save=args.n_epochs_per_pt_save, 
        n_epochs_per_pkl_update=args.n_epochs_per_eval if args.n_epochs_per_pkl_update is None else args.n_epochs_per_pkl_update,
        dim_option=args.dim_option,
        N = args.N,
        num_random_trajectories=args.num_random_trajectories,
        num_video_repeats=args.num_video_repeats,
        eval_record_video=args.eval_record_video,
        video_skip_frames=args.video_skip_frames,
        eval_plot_axis=args.eval_plot_axis,
        name='METRA',
        device=device,
        sample_cpu=args.sample_cpu,
        num_train_per_epoch=1,
        sd_batch_norm=args.sd_batch_norm,
        skill_dynamics_obs_dim=skill_dynamics_obs_dim,
        trans_minibatch_size=args.trans_minibatch_size,
        trans_optimization_epochs=args.trans_optimization_epochs,
        discount=args.sac_discount,
        discrete=args.discrete,
        unit_length=args.unit_length,
    )

    skill_common_args = dict(
        qf1=qf1,
        qf2=qf2,
        q1_list = q1_list if args.susd_q_function else [],
        log_alpha=log_alpha,
        tau=args.sac_tau,
        scale_reward=args.sac_scale_reward,
        target_coef=args.sac_target_coef,

        replay_buffer=replay_buffer,
        min_buffer_size=args.sac_min_buffer_size,
        inner=args.inner,

        num_alt_samples=args.num_alt_samples,
        split_group=args.split_group,

        dual_reg=args.dual_reg,
        dual_slack=args.dual_slack,
        dual_dist=args.dual_dist,

        pixel_shape=pixel_shape,
        partition_points=partition_points,
        exp_name = get_exp_name(args)[0],
        susd_dist_norm=args.susd_dist_norm,
        susd_input_factor0 = args.susd_input_factor0,
        susd_q_function = args.susd_q_function,
        susd_ablation_mode = args.susd_ablation_mode
    )

    dads_common_args = dict(
        qf1=qf1,
        qf2=qf2,
        log_alpha=log_alpha,
        tau=args.sac_tau,
        scale_reward=args.sac_scale_reward,
        target_coef=args.sac_target_coef,
        replay_buffer=replay_buffer,
        min_buffer_size=args.sac_min_buffer_size,
        inner=args.inner,
        num_alt_samples=args.num_alt_samples,
        split_group=args.split_group,
        dual_reg=args.dual_reg,
        dual_slack=args.dual_slack,
        dual_dist=args.dual_dist,
        pixel_shape=pixel_shape,
    )

    if getattr(args, 'train_method', 'susd') == 'metra':
        metra_common_args = dict(
            qf1=qf1,
            qf2=qf2,
            log_alpha=log_alpha,
            tau=args.sac_tau,
            scale_reward=args.sac_scale_reward,
            target_coef=args.sac_target_coef,
            replay_buffer=replay_buffer,
            min_buffer_size=args.sac_min_buffer_size,
            inner=args.inner,
            num_alt_samples=args.num_alt_samples,
            split_group=args.split_group,
            dual_reg=args.dual_reg,
            dual_slack=args.dual_slack,
            dual_dist=args.dual_dist,
            pixel_shape=pixel_shape,
        )
        algo = METRA(
            **algo_kwargs,
            **metra_common_args,
        )
    elif args.algo == 'metra':
        algo = SUSD(
            **algo_kwargs,
            **skill_common_args,
        )
    elif args.algo == 'dads':
        algo = DADS(
            **algo_kwargs,
            **dads_common_args,
        )
    elif args.algo == 'dads_poe':
        algo = DADSPoE(
            **algo_kwargs,
            **dads_common_args,
            poe_head_diversity_coef=getattr(args, 'poe_head_diversity_coef', 0.0),
            poe_responsibility_coef=getattr(args, 'poe_responsibility_coef', 0.0),
            poe_entropy_coef=getattr(args, 'poe_entropy_coef', 0.0),
            poe_head_kl_margin=getattr(args, 'poe_head_kl_margin', 0.5),
        )
    else:
        raise NotImplementedError


    if args.sample_cpu:
        algo.option_policy.cpu()
    else:
        algo.option_policy.to(device)

    runner.setup(
        algo=algo,
        env=env,
        make_env=contextualized_make_env,
        sampler_cls=OptionMultiprocessingSampler,
        sampler_args=dict(n_thread=args.n_thread),
        n_workers=args.n_parallel,
    )

    algo.option_policy.to(device)
    runner.train(n_epochs=args.n_epochs, batch_size=args.traj_batch_size)
    if os.environ.get("SUSD_SAVE_FINAL", "1") != "0":
        runner.save(args.n_epochs, new_save=True, pt_save=True, pkl_update=False)


if __name__ == '__main__':
    mp.set_start_method(START_METHOD)
    run()
