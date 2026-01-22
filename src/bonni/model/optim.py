from dataclasses import dataclass

import optax

@dataclass(frozen=True, kw_only=True)
class OptimConfig:
    """
    Configuration for training the ensemble between every sampling step.
    
    This configuration manages the optimizer settings and the learning rate 
    schedule (typically a warmup followed by a decay).

    Attributes:
        total_steps (int): The total number of optimization steps to perform 
            in this training phase.
        warmup_steps (int | None): The number of steps at the start of training 
            during which the learning rate increases linearly from `init_lr` 
            to `peak_lr`. If fixed_lr is not specified, then this parameter is required. Defaults to None.
        peak_lr (float): The maximum learning rate reached after the warmup phase is complete. Defaults to 1e-9.
        final_lr (float): The final learning rate at the end of `total_steps`. 
            The schedule typically decays from `peak_lr` to this value. 
            Defaults to 1e-3.
        init_lr (float): The initial learning rate at step 0, before warmup 
            begins. Defaults to 1e-5.
        fixed_lr (float | None): If provided, overrides the scheduling logic 
            (warmup/peak/final) and uses this constant learning rate for 
            all steps. Defaults to None.
        clip_grad_norm (float | None): The maximum norm for gradient clipping. 
            If gradients exceed this norm, they are rescaled. Set to None to 
            disable clipping. Defaults to 1.0.
        use_adamw (bool): Whether to use the AdamW optimizer. If False, 
            standard Adam without weight decay is used. Defaults to True.
    """
    total_steps: int
    warmup_steps: int | None = None
    peak_lr: float = 1e-3
    final_lr: float = 1e-9
    init_lr: float = 1e-5
    fixed_lr: float | None = None
    clip_grad_norm: float | None = 1.0
    use_adamw: bool = True
    
def get_optimizer_from_cfg(optim_cfg: OptimConfig):
    if optim_cfg.fixed_lr is None:
        assert optim_cfg.warmup_steps is not None, "If lr-schedule is used, then optim_cfg.warmup_steps needs to be specified"
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=optim_cfg.init_lr,
            peak_value=optim_cfg.peak_lr,
            end_value=optim_cfg.final_lr,
            warmup_steps=optim_cfg.warmup_steps,
            decay_steps=round(optim_cfg.total_steps),
        )
    else:
        schedule = optim_cfg.fixed_lr
    
    optim_chain = []
    if optim_cfg.clip_grad_norm is not None:
        optim_chain.append(optax.clip_by_global_norm(optim_cfg.clip_grad_norm))
    
    if optim_cfg.use_adamw:
        optim_chain.append(optax.adamw(learning_rate=schedule))
    else:
        optim_chain.append(optax.adam(learning_rate=schedule))
    
    optimizer = optax.chain(*optim_chain)
    return optimizer
