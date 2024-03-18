from functools import partial
from torch import optim as optim


def build_optimizer(hparams, model, is_pretrain):
    if is_pretrain:
        return build_pretrain_optimizer(hparams, model)
    else:
        return build_finetune_optimizer(hparams, model)


def build_pretrain_optimizer(hparams, model):
    skip = {}
    skip_keywords = {}
    if hasattr(model, "no_weight_decay"):
        skip = model.no_weight_decay()
    if hasattr(model, "no_weight_decay_keywords"):
        skip_keywords = model.no_weight_decay_keywords()

    # Extract parameters from the model
    no_decay = []
    has_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if (
            len(param.shape) == 1
            or name.endswith(".bias")
            or (name in skip)
            or check_keywords_in_name(name, skip_keywords)
        ):
            no_decay.append(param)
        else:
            has_decay.append(param)
    parameters = [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": has_decay},
    ]

    # Create optimizer
    opt = hparams['opt'].lower()
    optimizer = None
    if opt == "sgd":
        optimizer = optim.SGD(
            parameters,
            momentum=0.9,
            nesterov=True,
            lr=hparams['lr'],
            weight_decay=hparams['weight_decay'],
        )
    elif opt == "adamw":
        optimizer = optim.AdamW(
            parameters,
            betas=(0.9, 0.999),
            lr=hparams['lr'],
            weight_decay=hparams['weight_decay'],
        )
    return optimizer


def build_finetune_optimizer(hparams, model):
    get_layer_func = None
    scales = None

    # Set layer function helpers if using MViT backbone
    if hparams['arch'] == "mvit" and hparams['layer_decay'] != 1:
        num_layers = 16
        get_layer_func = partial(get_mvit_layer, num_layers=num_layers + 2)
        scales = list(
            hparams['layer_decay']**i for i in reversed(range(num_layers + 2))
        )  # layer_decay=1 disable
    else:
        return build_pretrain_optimizer(hparams, model)

    # Extract parameters from the model
    skip = {}
    skip_keywords = {}
    if hasattr(model, "no_weight_decay"):
        skip = model.no_weight_decay()
    if hasattr(model, "no_weight_decay_keywords"):
        skip_keywords = model.no_weight_decay_keywords()
    parameters = get_finetune_param_groups(
        model,
        hparams['lr'],
        hparams['weight_decay'],
        get_layer_func,
        scales,
        skip,
        skip_keywords,
    )

    # Create optimizer
    opt = hparams['opt'].lower()
    optimizer = None
    if opt == "sgd":
        optimizer = optim.SGD(
            parameters,
            momentum=0.9,
            nesterov=True,
            lr=hparams['lr'],
            weight_decay=hparams['weight_decay'],
        )
    elif opt == "adamw":
        optimizer = optim.AdamW(
            parameters,
            betas=(0.9, 0.999),
            lr=hparams['lr'],
            weight_decay=hparams['weight_decay'],
        )
    return optimizer


def get_mvit_layer(name, num_layers):
    layer_name = name.replace("mvit.", "").replace("model.", "")
    if (
        layer_name in ("mask_token")
        or layer_name.startswith("patch_embed")
        or layer_name.startswith("cls_positional_encoding")
    ):
        return 0
    elif layer_name.startswith("blocks"):
        layer_id = int(layer_name.split(".")[1])
        return layer_id + 1
    else:
        return num_layers - 1


def get_finetune_param_groups(
    model, lr, weight_decay, get_layer_func, scales, skip=(), skip_keywords=()
):
    param_group_vars = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if (
            len(param.shape) == 1
            or name.endswith(".bias")
            or (name in skip)
            or check_keywords_in_name(name, skip_keywords)
        ):
            group_name = "no_decay"
            this_weight_decay = 0.0
        else:
            group_name = "decay"
            this_weight_decay = weight_decay
        layer_id = None
        if get_layer_func is not None:
            layer_id = get_layer_func(name)
            group_name = "layer_%d_%s" % (layer_id, group_name)

        if group_name not in param_group_names:
            if scales is not None:
                scale = scales[layer_id]
            else:
                scale = 1.0
            param_group_vars[group_name] = {
                "group_name": group_name,
                "weight_decay": this_weight_decay,
                "params": [],
                "lr": lr * scale,
                "lr_scale": scale,
            }
        param_group_vars[group_name]["params"].append(param)
        return list(param_group_vars.values())


def check_keywords_in_name(name, keywords=()):
    for keyword in keywords:
        if keyword in name:
            return True
    return False
