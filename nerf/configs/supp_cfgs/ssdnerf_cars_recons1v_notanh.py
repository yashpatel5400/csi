name = 'ssdnerf_cars_recons1v_notanh'

model = dict(
    type='DiffusionNeRF',
    code_size=(3, 6, 128, 128),
    code_reshape=(18, 128, 128),
    code_activation=dict(type='IdentityCode'),
    grid_size=64,
    diffusion=dict(
        type='GaussianDiffusion',
        num_timesteps=1000,
        betas_cfg=dict(type='linear'),
        denoising=dict(
            type='DenoisingUnetMod',
            image_size=128,
            in_channels=18,
            base_channels=128,
            channels_cfg=[1, 2, 2, 4, 4],
            resblocks_per_downsample=2,
            dropout=0.1,
            use_scale_shift_norm=True,
            downsample_conv=True,
            upsample_conv=True,
            num_heads=4,
            attention_res=[32, 16, 8]),
        timestep_sampler=dict(
            type='SNRWeightedTimeStepSampler',
            power=0.5),
        ddpm_loss=dict(
            type='DDPMMSELossMod',
            rescale_mode='timestep_weight',
            log_cfgs=dict(
                type='quartile', prefix_name='loss_mse', total_timesteps=1000),
            data_info=dict(pred='v_t_pred', target='v_t'),
            weight_scale=4.0,
            scale_norm=True)),
    decoder=dict(
        type='TriPlaneDecoder',
        interp_mode='bilinear',
        base_layers=[6 * 3, 64],
        density_layers=[64, 1],
        color_layers=[64, 3],
        use_dir_enc=True,
        dir_layers=[16, 64],
        activation='silu',
        sigma_activation='trunc_exp',
        sigmoid_saturation=0.001,
        max_steps=256),
    decoder_use_ema=True,
    freeze_decoder=False,
    bg_color=1,
    rc_loss=dict(
        type='MSELoss',
        loss_weight=20.0),
    reg_loss=dict(
        type='RegLoss',
        power=2,
        loss_weight=3e-3),
    cache_size=2458)

save_interval = 5000
eval_interval = 20000
code_dir = 'cache/' + name + '/code'
work_dir = 'work_dirs/' + name

train_cfg = dict(
    dt_gamma_scale=0.5,
    density_thresh=0.1,
    extra_scene_step=15,
    n_inverse_rays=2 ** 12,
    n_decoder_rays=2 ** 12,
    loss_coef=0.1 / (128 * 128),
    optimizer=dict(type='Adam', lr=2e-2, weight_decay=0.),
    cache_load_from=code_dir,
    viz_dir=None)
test_cfg = dict(
    img_size=(128, 128),
    num_timesteps=75,
    clip_range=[-2, 2],
    density_thresh=0.1,
    # max_render_rays=16 * 128 * 128,
    dt_gamma_scale=0.5,
    n_inverse_rays=2 ** 14,
    override_cfg={'diffusion_ema.ddpm_loss.weight_scale': 1.0},
    loss_coef=0.1 / (128 * 128),
    guidance_gain=3.2 * (2 ** 14),
    cond_mode='guide_optim',
    n_inverse_steps=25,
    extra_scene_step=3,
    optimizer=dict(type='Adam', lr=0.01, weight_decay=0.),
    lr_scheduler=dict(type='ExponentialLR', gamma=0.998)
)

optimizer = dict(
    diffusion=dict(type='Adam', lr=1e-4, weight_decay=0.),
    decoder=dict(type='Adam', lr=1e-3, weight_decay=0.))
dataset_type = 'ShapeNetSRN'
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_prefix='data/shapenet/cars_train',
        cache_path='data/shapenet/cars_train_cache.pkl'),
    val_uncond=dict(
        type=dataset_type,
        data_prefix='data/shapenet/cars_test',
        load_imgs=False,
        num_test_imgs=251,
        scene_id_as_name=True,
        cache_path='data/shapenet/cars_test_cache.pkl'),
    val_cond=dict(
        type=dataset_type,
        data_prefix='data/shapenet/cars_test',
        specific_observation_idcs=[64],
        cache_path='data/shapenet/cars_test_cache.pkl'),
    train_dataloader=dict(split_data=True))
lr_config = dict(
    policy='Fixed',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001)
checkpoint_config = dict(interval=save_interval, by_epoch=False, max_keep_ckpts=2)

evaluation = [
    dict(
        type='GenerativeEvalHook3D',
        data='val_cond',
        interval=eval_interval,
        feed_batch_size=32,
        viz_step=32,
        metrics=dict(
            type='FID',
            num_images=704 * 250,
            inception_pkl='work_dirs/cache/cars_test_inception_stylegan.pkl',
            inception_args=dict(
                type='StyleGAN',
                inception_path='work_dirs/cache/inception-2015-12-05.pt'),
            bgr2rgb=False),  # already is rgb
        viz_dir=work_dir + '/viz_cond',
        save_best_ckpt=False)]

total_iters = 80000
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ])
# yapf:enable

custom_hooks = [
    dict(
        type='ExponentialMovingAverageHook',
        module_keys=('diffusion_ema', 'decoder_ema'),
        interp_mode='lerp',
        interval=1,
        start_iter=0,
        momentum_policy='rampup',
        momentum_cfg=dict(
            ema_kimg=4, ema_rampup=0.05, batch_size=16, eps=1e-8),
        priority='VERY_HIGH'),
    dict(
        type='SaveCacheHook',
        interval=save_interval,
        by_epoch=False,
        out_dir=code_dir,
        viz_dir='cache/' + name + '/viz'),
    dict(
        type='ModelUpdaterHook',
        step=[2000],
        cfgs=[{'train_cfg.extra_scene_step': 3}],
        by_epoch=False)
]

# use dynamic runner
runner = dict(
    type='DynamicIterBasedRunner',
    is_dynamic_ddp=False,
    pass_training_status=True)
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', save_interval)]
use_ddp_wrapper = True
find_unused_parameters = False
cudnn_benchmark = True
opencv_num_threads = 0
mp_start_method = 'fork'
