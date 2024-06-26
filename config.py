import torch


def get_model_config(model_name, config):
    if model_name == "llama2_7b":
        config.activation_func = torch.nn.functional.silu
        config.add_bias_linear = False
        config.bias_activation_fusion = False
        config.gated_linear_unit = True
        config.apply_query_key_layer_scaling = True
        config.layernorm_zero_centered_gamma = False
        config.bias_dropout_fusion = False
        config.te_attn_mask_type = None
        config.rotary_percent = 0.5
        config.apply_rope_fusion = False
        config.attention_softmax_in_fp32 = True
        config.ffn_hidden_size = 11008
        config.position_embedding_type = "rope"
    elif config.my_model_type == "llama3_8b":
        config.activation_func = torch.nn.functional.silu
        config.add_bias_linear = False
        config.bias_activation_fusion = False
        config.gated_linear_unit = True
        config.apply_query_key_layer_scaling = True
        config.layernorm_zero_centered_gamma = (
            False  # Zero centered gamma not supported for RMSNorm
        )
        config.bias_dropout_fusion = False
        config.te_attn_mask_type = None
        config.rotary_percent = 0.5
        config.apply_rope_fusion = False
        config.attention_softmax_in_fp32 = True
        config.ffn_hidden_size = 14336
        config.position_embedding_type = "rope"

    return config
