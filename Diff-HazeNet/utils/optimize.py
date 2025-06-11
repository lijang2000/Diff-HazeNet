import torch.optim as optim

def get_optimizer(config, parameters):
    if config.optim.optimizer == 'Adam':
        return optim.Adam(parameters, lr=config.optim.lr, weight_decay=config.optim.weight_decay,betas=(0.9, 0.999),
                          amsgrad=config.optim.amsgrad, eps=config.optim.eps)
    elif config.optim.optimizer == 'RMSProp':
        return optim.RMSprop(parameters, lr=config.optim.lr, weight_decay=config.optim.weight_decay)
    elif config.optim.optimizer == 'SGD':
        return optim.SGD(parameters, lr=config.optim.lr, momentum=0.9)
    elif config.optim.optimizer == 'cos':
        optimizer = optim.Adam(parameters, lr=config.optim.lr, weight_decay=config.optim.weight_decay,
                          betas=(0.9, 0.999), amsgrad=config.optim.amsgrad, eps=config.optim.eps)
        cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=config.optim.epoch, eta_min=0,
                                                           last_epoch=-1)
        return cosineScheduler

    elif config.optim.optimizer == 'AdamW':
        return optim.AdamW(parameters, lr=config.optim.lr, weight_decay=config.optim.weight_decay,betas=(0.9, 0.999),
                           amsgrad=config.optim.amsgrad, eps=config.optim.eps)
    else:
        raise NotImplementedError('Optimizer {} not understood.'.format(config.optim.optimizer))
