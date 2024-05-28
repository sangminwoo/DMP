from models.DiT.models import DiT_models


def create_model(model_config):
    """
    Create various architectures from model_config
    """
    if model_config.name in DiT_models.keys():
        model = DiT_models[model_config.name](
            input_size=model_config.param.latent_size,
            num_classes=model_config.param.num_classes,
            num_tokens=model_config.vp.num_tokens,
            topk=model_config.vp.topk,
            vp_type=model_config.vp.type,
            uncond=model_config.vp.uncond,
            init_type=model_config.vp.init_type,
            gate_type=model_config.vp.gate_type,
            pt_depth=model_config.vp.pt_depth,
        )
    else:
        raise NotImplementedError
    return model
