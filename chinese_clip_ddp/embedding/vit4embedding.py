from transformers.models.vit.configuration_vit import ViTConfig
from transformers.models.vit.modeling_vit import ViTModel, ViTPreTrainedModel
from torch import nn
from typing import Optional
import torch


class VitConfigForEmbedding(ViTConfig):
    def __init__(
        self,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0,
        attention_probs_dropout_prob=0,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        image_size=224,
        patch_size=16,
        num_channels=3,
        qkv_bias=True,
        encoder_stride=16,
        embedding_dim=1024,
        **kwargs,
    ):
        super().__init__(
            hidden_size,
            num_hidden_layers,
            num_attention_heads,
            intermediate_size,
            hidden_act,
            hidden_dropout_prob,
            attention_probs_dropout_prob,
            initializer_range,
            layer_norm_eps,
            image_size,
            patch_size,
            num_channels,
            qkv_bias,
            encoder_stride,
            **kwargs,
        )
        self.embedding_dim = embedding_dim


class ViTForEmbedding(ViTPreTrainedModel):
    def __init__(self, config: VitConfigForEmbedding) -> None:
        super().__init__(config)

        self.vit = ViTModel(config, add_pooling_layer=False, use_mask_token=True)

        self.output_linear = nn.Linear(
            in_features=config.hidden_size, out_features=config.embedding_dim
        )

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        outputs = self.vit(
            pixel_values,
            bool_masked_pos=bool_masked_pos,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
        )

        last_hidden_states = outputs.last_hidden_state[:, 0]
        pool_output = self.output_linear(last_hidden_states)
        return last_hidden_states, pool_output
