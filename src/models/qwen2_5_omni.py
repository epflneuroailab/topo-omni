import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Optional, Union

import os
import torch
import numpy as np
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPast, ModelOutput
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, check_torch_load_is_safe, logging
from transformers.utils.hub import cached_file
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
    Qwen2_5OmniPreTrainedModelForConditionalGeneration,
    Qwen2_5OmniAudioAttention,
    Qwen2_5OmniAttention,
    Qwen2MLP,
    Qwen2_5OmniMLP,
    Qwen2_5OmniVisionAttention,
    apply_multimodal_rotary_pos_emb,
    apply_rotary_pos_emb_vision,
    rotate_half,
    eager_attention_forward,
    repeat_kv,
    Qwen2_5OmniRotaryEmbedding,
    Qwen2_5OmniPatchMerger,
    Qwen2_5_VisionRotaryEmbedding,
    Qwen2_5_VisionPatchEmbed,
    Qwen2_5OmniVisionBlock,
    SinusoidsPositionEmbedding,
    Qwen2_5OmniAudioEncoderLayer,
    Qwen2_5OmniDecoderLayer,
)

from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import (
    Qwen2_5OmniAudioEncoderConfig,
    Qwen2_5OmniBigVGANConfig,
    Qwen2_5OmniConfig,
    Qwen2_5OmniDiTConfig,
    Qwen2_5OmniTalkerConfig,
    Qwen2_5OmniTextConfig,
    Qwen2_5OmniThinkerConfig,
    Qwen2_5OmniToken2WavConfig,
    Qwen2_5OmniVisionEncoderConfig,
)

from src.models.spatial_utils import (
    DEFAULT_SHEET_LAYOUT,
    LayerPositions,
    RadiusSchedule,
    SheetLayout,
    attach_tile_scoping,
    tile_grid_pos,
    build_sheet_coordinates,
    build_unit_radii,
    describe_radius_schedule,
    spatial_loss_fn,
)


logger = logging.get_logger(__name__)

@dataclass
class TopoBaseModelOutputWithPast(ModelOutput):

    last_hidden_state: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None
    cortical_sheet: Optional[tuple[torch.FloatTensor, ...]] = None

@dataclass
class TopoBaseModelOutput(ModelOutput):
    """
    Base class for model's outputs, with potential hidden states and attentions.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    last_hidden_state: Optional[torch.FloatTensor] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None
    cortical_sheet: Optional[tuple[torch.FloatTensor, ...]] = None

@auto_docstring
class Qwen2_5OmniPreTrainedModel(PreTrainedModel):
    config: Qwen2_5OmniConfig
    base_model_prefix = "model"
    input_modalities = ["image", "video", "audio", "text"]
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen2_5OmniDecoderLayer", "Qwen2_5OmniVisionBlock"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn = True
    _supports_sdpa = True
    _can_compile_fullgraph = False
    _supports_attention_backend = True

def _band_from_layers(states, layout, band_rows, band_cols, tile, device):
    """`[L, T, D]` layer states -> `[T, band_rows, band_cols]`.

    strip mode is the original `permute(1,0,2).reshape(-1, rows, cols)`: layer `l` lands on
    a full-width strip because its D values run contiguously across `D/cols` whole rows.

    tiled mode places each layer into its own 2-D tile instead. The scatter is done with a
    precomputed index so it stays a single vectorised write, and is cached on the layout
    because it depends only on geometry.
    """
    L, T, D = states.shape
    flat = states.permute(1, 0, 2).reshape(T, L * D)          # [T, L*D], layer-major
    if layout.mode == "strip":
        return flat.reshape(T, band_rows, band_cols)

    # tile_order is part of the key: the same band dims under a different order are a
    # DIFFERENT scatter, and reusing a cached one would silently transpose the band.
    key = f"_band_idx_{band_rows}x{band_cols}_{layout.tile_order}"
    idx = getattr(layout, key, None)
    if idx is None:
        th, tw, nr, nc = tile
        serp = layout.tile_order == "serpentine"
        idx = torch.empty(L * D, dtype=torch.long)
        for l in range(L):
            tr, tc = tile_grid_pos(l, nc, serp)
            # a layer's D values fill its tile in row-major order
            rows = torch.arange(tr * th, (tr + 1) * th).repeat_interleave(tw)
            cols = torch.arange(tc * tw, (tc + 1) * tw).repeat(th)
            idx[l * D:(l + 1) * D] = rows * band_cols + cols
        setattr(layout, key, idx)
    out = torch.zeros(T, band_rows * band_cols, dtype=flat.dtype, device=flat.device)
    out[:, idx.to(flat.device)] = flat
    return out.reshape(T, band_rows, band_cols)


def _sheet_has_frontend_slot(num_blocks: int) -> bool:
    """Whether the encoder towers put their conv front-end (`conv2` / `patch_embed`) on
    the cortical sheet.

    Read off the active LAYOUT rather than a config flag, so an adaptor can never be built
    without a tile to put it in, nor a tile left unfilled. The 33-slot geometries (strip,
    strip-4096, continuous, tiled) carry the front-end; `tiled32` has 32 slots and drops it.
    """
    return DEFAULT_SHEET_LAYOUT.num_encoder_layers == num_blocks + 1


class CorticalAdaptor(nn.Module):
    def __init__(self, config: Qwen2_5OmniConfig):
        super().__init__()
        self.input_dim = config.hidden_size if hasattr(config, "hidden_size") else config.d_model
        self.cortical_dim = config.cortical_dim if hasattr(config, "cortical_dim") else self.input_dim

        # [input_dim, cortical_dim], matching how it is used: `Z = X @ self.W`. It was
        # declared transposed, which happened to work only while the two dims were equal --
        # so `cortical_dim != input_dim` was unreachable. For the square default the shape
        # is unchanged, so existing checkpoints load exactly as before.
        self.W = nn.Parameter(torch.randn(self.input_dim, self.cortical_dim))
        self.z_hook = nn.Identity()  # hook point for cortical states
        # W_pinv is only ever a cache of pinv(W); any write to W invalidates it.
        self._pinv_stale = True

        is_training = hasattr(config, "is_training") and config.is_training
        if not is_training:
            # Register as a non-persistent buffer (won't be saved in state_dict)
            self.register_buffer("W_pinv", torch.zeros(self.cortical_dim, self.input_dim), persistent=False)

            # Do NOT fill the cache here: `from_pretrained` builds the module under
            # `init_empty_weights`, so `self.W` is still a meta tensor with no data, and
            # `torch.autocast(device_type="meta")` raises "unsupported scalarType". The
            # cache is filled on the first forward, once the checkpoint has been loaded.

            # Recompute W_pinv every time state_dict is loaded
            self.register_load_state_dict_post_hook(self._on_load)

    def _recompute_pinv(self):
        # W_pinv is a plain cache, never a differentiated path: no_grad keeps it from
        # pinning an autograd graph when refreshed outside an inference context.
        with torch.no_grad():
            self.W_pinv = self._pinv()
        self._pinv_stale = False

    def _pinv(self):
        # `torch.linalg.pinv` is on torch.autocast's reduced-precision list, so inside an
        # autocast region it is dispatched in bf16 even when handed a float32 input. Its
        # backward then mixes a bf16 grad with the float32 saved input and dies with
        # "expected mat1 and mat2 to have the same dtype". Pin it to float32 explicitly.
        with torch.autocast(device_type=self.W.device.type, enabled=False):
            return torch.linalg.pinv(self.W.float()).to(self.W.dtype)

    @staticmethod
    def _on_load(module, incompatible_keys):
        module._pinv_stale = True

    def forward(self, X):
        Z = X @ self.W
        Z = self.z_hook(Z)  # hook for cortical states
        if self.training:
            W_pinv = self._pinv()
        else:
            if self._pinv_stale:
                self._recompute_pinv()
            W_pinv = self.W_pinv

        X_hat = Z @ W_pinv
        return X_hat, Z

    def initialize(self, epsilon: float = 1e-3, identity=False):
        with torch.no_grad():
            if identity:
                # Tile the identity across cortical_dim rather than zero-padding it. With
                # padding, an overcomplete adaptor starts with only the first input_dim
                # channels carrying signal and the rest at noise scale (~30x less variance),
                # so most new units begin effectively dead and `drop_dead_units` discards
                # them. Tiling makes every channel a live copy of an input dim, which the
                # spatial loss then differentiates. Reconstruction is unaffected: W keeps
                # full ROW rank, so W @ pinv(W) = I exactly and the base model is untouched.
                reps = -(-self.cortical_dim // self.input_dim)          # ceil
                W_init = torch.eye(self.input_dim).repeat(1, reps)[:, :self.cortical_dim]
            else:
                W_init = torch.zeros(self.input_dim, self.cortical_dim)
            W_init += epsilon * torch.randn_like(self.W).to(W_init.device)
            self.W.copy_(W_init)
        self._pinv_stale = True


############################
#      Start Thinker       #
############################


@dataclass
@auto_docstring(
    custom_intro="""
    Base class for Qwen2.5OmniThinker causal language model (or autoregressive) outputs.
    """
)
class Qwen2_5OmniThinkerCausalLMOutputWithPast(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Language modeling loss (for next-token prediction).
    logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`, *optional*):
        Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

        Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
        `past_key_values` input) to speed up sequential decoding.
    rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
        The rope index difference between sequence length and multimodal rope.
    """

    loss: Optional[torch.FloatTensor] = None
    task_loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None
    spatial_loss: Optional[torch.FloatTensor] = None
    #: The two spatial-loss terms, reported separately so wandb shows whether each is
    #: actually descending. `spatial_loss` is alpha * (fine + coarse_weight * coarse), so
    #: without these a stalled coarse term is invisible -- the sum keeps falling on the
    #: back of the fine term alone. Names must end in "_loss" for the trainer to log them.
    fine_spatial_loss: Optional[torch.FloatTensor] = None
    coarse_spatial_loss: Optional[torch.FloatTensor] = None
    multimodal_spatial_loss: Optional[torch.FloatTensor] = None
    audio_spatial_loss: Optional[torch.FloatTensor] = None
    visual_spatial_loss: Optional[torch.FloatTensor] = None

    multimodal_cortical_sheet: Optional[torch.FloatTensor] = None
    visual_cortical_sheet: Optional[torch.FloatTensor] = None
    audio_cortical_sheet: Optional[torch.FloatTensor] = None
    unified_sheet: Optional[torch.FloatTensor] = None


@auto_docstring(
    custom_intro="""
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`Qwen2_5OmniAudioEncoderLayer`].
    """
)
class Qwen2_5OmniAudioEncoder(Qwen2_5OmniPreTrainedModel):
    config: Qwen2_5OmniAudioEncoderConfig
    main_input_name = "input_features"
    input_modalities = "audio"
    _no_split_modules = ["Qwen2_5OmniAudioEncoderLayer"]
    _supports_sdpa = True

    def __init__(self, config: Qwen2_5OmniAudioEncoderConfig):
        super().__init__(config)
        self.dropout = config.dropout

        embed_dim = config.d_model
        self.num_mel_bins = config.num_mel_bins
        self.max_source_positions = config.max_source_positions
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0
        self.n_window = config.n_window
        self.conv1 = nn.Conv1d(self.num_mel_bins, embed_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1)
        self.positional_embedding = SinusoidsPositionEmbedding(self.max_source_positions, embed_dim)
        self.audio_bos_eos_token = nn.Embedding(2, config.output_dim)
        self.layers = nn.ModuleList([Qwen2_5OmniAudioEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.cortical_adaptors = nn.ModuleList([CorticalAdaptor(config) for _ in range(config.encoder_layers)])
        # Sheet slot 0: the conv front-end, the earliest audio stage with any
        # spectrotemporal structure for the spatial loss to organize. Kept as its
        # own attribute rather than prepended to `cortical_adaptors` so existing
        # checkpoints' `cortical_adaptors.N.W` keys still map to the same layer;
        # prepending would shift every index by one and load silently misaligned.
        # The name must contain "cortical_adaptor" -- `train.py` selects trainable
        # parameters by that substring.
        #
        # Present only when the layout allots it a slot. `tiled32` does not: the slot
        # existed to keep the audio and vision bands row-aligned for a column-wise concat,
        # and tiling each band independently removes that constraint -- while the adaptor
        # is a rank-constrained bottleneck sitting IN the residual stream, so keeping it
        # would go on distorting the whole audio pathway before layer 0 to buy an
        # alignment nothing needs any more.
        self.has_frontend_slot = _sheet_has_frontend_slot(config.encoder_layers)
        if self.has_frontend_slot:
            self.frontend_cortical_adaptor = CorticalAdaptor(config)
        self.ln_post = nn.LayerNorm(config.d_model)
        self.avg_pooler = nn.AvgPool1d(2, stride=2)
        self.proj = nn.Linear(config.d_model, config.output_dim)
        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()
    
    def _freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False
        self._requires_grad = False

    def get_input_embeddings(self) -> nn.Module:
        return self.conv1

    def set_input_embeddings(self, value: nn.Module):
        self.conv1 = value

    def _pool_cortical_states(self, cortical_states, aftercnn_lens):
        """Reduce one adaptor's cortical states to a `[B, T, D]` sheet slot.

        Splits the flat per-token states back into the batch, halves the time axis
        with `avg_pooler`, and right-pads each item to the longest so the slots can
        be stacked. Shared by the conv front-end slot and every transformer layer.
        """
        cortical_states_list = cortical_states.split(aftercnn_lens.tolist(), dim=0)
        max_len = max([state.shape[0] for state in cortical_states_list]) // 2
        final_cortical_states = []
        for each_cortical_state in cortical_states_list:
            each_cortical_state = self.avg_pooler(each_cortical_state.transpose(0, 1)).transpose_(0, 1)
            zero_pad = torch.zeros((max_len - each_cortical_state.shape[0], each_cortical_state.shape[1])).to(each_cortical_state.device)
            each_cortical_state = torch.cat([each_cortical_state, zero_pad], dim=0)
            final_cortical_states.append(each_cortical_state)
        return torch.stack(final_cortical_states, dim=0)

    def _prepare_attention_mask(self, inputs_tensor: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
        # Flash Attention 2 doesn't need a 4D mask and relies on `cu_seqlens/max_seqlen`
        # NOTE: the created attention masl only approximates the ragged FA2 attention by
        # allowing bidirectional attention within `cu_seqlens` blocks, and not attending between
        # blocks. Though it will not be a 100% match for FA2's `varlen` path
        if self.config._attn_implementation == "flash_attention_2":
            return None

        seq_length = inputs_tensor.shape[0]
        attention_mask = torch.full(
            [1, 1, seq_length, seq_length],
            torch.finfo(inputs_tensor.dtype).min,
            device=inputs_tensor.device,
            dtype=inputs_tensor.dtype,
        )
        for i in range(1, len(cu_seqlens)):
            attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = 0
        return attention_mask

    @auto_docstring
    def forward(
        self,
        input_features,
        feature_lens=None,
        aftercnn_lens=None,
        **kwargs,
    ):
        r"""
        feature_lens (`torch.LongTensor` of shape `(batch_size,)`):
            mel length
        aftercnn_lens (`torch.LongTensor` of shape `(batch_size,)`):
            mel length after cnn
        """
        chunk_num = torch.ceil(feature_lens / (self.n_window * 2)).long()

        chunk_lengths = torch.tensor(
            [self.n_window * 2] * chunk_num.sum(),
            dtype=torch.long,
            device=feature_lens.device,
        )
        tail_chunk_index = F.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]
        chunk_lengths[tail_chunk_index] = feature_lens % (self.n_window * 2)
        chunk_lengths = torch.where(chunk_lengths == 0, self.n_window * 2, chunk_lengths)

        chunk_list = input_features.split(chunk_lengths.tolist(), dim=1)
        padded_feature, padded_mask, padded_mask_after_cnn = self.padded_and_mask_function(
            chunk_list, chunk_lengths, padding_value=0, padding_side="right"
        )
        padded_embed = nn.functional.gelu(self.conv1(padded_feature)) * padded_mask
        padded_embed = nn.functional.gelu(self.conv2(padded_embed)).transpose(1, 2)

        # Front-end cortical adaptor, applied *before* the positional embedding so
        # the sheet slot carries the pure spectrotemporal code rather than the
        # sinusoidal time index. Like every other adaptor it sits in the forward
        # path, so the whole audio pathway now flows through this bottleneck.
        # It acts independently per position, so applying it to the padded tensor
        # and selecting afterwards is identical to selecting first.
        frontend_cortical_states = None
        if self.has_frontend_slot:
            padded_embed, frontend_cortical_states = self.frontend_cortical_adaptor(padded_embed)
            frontend_cortical_states = frontend_cortical_states[padded_mask_after_cnn]

        padded_embed = padded_embed + self.positional_embedding.positional_embedding[
            : padded_embed.shape[1], :
        ].unsqueeze(0).to(padded_embed.dtype)
        hidden_states = padded_embed[padded_mask_after_cnn]
        cu_seqlens = torch.cat(
            (
                torch.zeros(1, device=padded_mask_after_cnn.device, dtype=torch.int32),
                padded_mask_after_cnn.sum(1).cumsum(0),
            )
        ).to(torch.int32)
        attention_mask = self._prepare_attention_mask(hidden_states, cu_seqlens)

        # With a front-end slot: slot 0 is the conv front-end and slots 1..L the
        # transformer layers. Without one: slots 0..L-1 are the transformer layers.
        cortical_sheet = ((self._pool_cortical_states(frontend_cortical_states, aftercnn_lens),)
                          if self.has_frontend_slot else ())
        for layer_idx, encoder_layer in enumerate(self.layers):
            layer_outputs = encoder_layer(
                hidden_states,
                cu_seqlens=cu_seqlens,
                attention_mask=attention_mask,
                **kwargs,
            )
            hidden_states = layer_outputs[0]

            # Cortical adaptor
            hidden_states, cortical_states = self.cortical_adaptors[layer_idx](hidden_states)

            cortical_sheet += (self._pool_cortical_states(cortical_states, aftercnn_lens),)


        hidden_states_list = hidden_states.split(aftercnn_lens.tolist(), dim=0)
        token_audio_list = []
        for each_audio_states in hidden_states_list:
            each_audio_states = self.avg_pooler(each_audio_states.transpose(0, 1)).transpose_(0, 1)
            each_audio_states = self.ln_post(each_audio_states)
            each_audio_states = self.proj(each_audio_states)
            token_audio_list.append(each_audio_states)
        token_audio = torch.cat(token_audio_list, dim=0)
        return TopoBaseModelOutput(last_hidden_state=token_audio, cortical_sheet=cortical_sheet)

    def padded_and_mask_function(self, tensor_list, tensor_len, padding_value=0, padding_side="right"):
        """
        Pads a sequence of tensors to their maximum length on indicated `padding_side`.
        Then prepares a mask so that pad tokens are not attended to.
        """
        max_len = tensor_len.max()
        dim = tensor_list[0].shape[0]
        padded_tensor = torch.full(
            size=(len(tensor_list), dim, max_len),
            fill_value=padding_value,
            dtype=self.dtype,
            device=tensor_list[0].device,
        )

        batch_mask = torch.zeros(
            (len(tensor_len), max_len),
            dtype=torch.long,
            device=padded_tensor.device,
        )
        for i, length in enumerate(tensor_len):
            batch_mask[i, :length] = 1
            padded_tensor[i, :, :length] = tensor_list[i]

        feature_lens_after_cnn = (tensor_len - 1) // 2 + 1
        max_len_after_cnn = feature_lens_after_cnn.max()
        batch_mask_after_cnn = torch.zeros(
            (len(tensor_len), max_len_after_cnn),
            dtype=torch.long,
            device=padded_tensor.device,
        )
        for i, length in enumerate(feature_lens_after_cnn):
            batch_mask_after_cnn[i, :length] = 1
        return (
            padded_tensor,
            batch_mask.unsqueeze(1),
            batch_mask_after_cnn.bool(),
        )

    # Ignore copy
    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """
        Computes the output length of the convolutional layers and the output length of the audio encoder
        """
        input_lengths = (input_lengths - 1) // 2 + 1
        output_lengths = (input_lengths - 2) // 2 + 1
        return input_lengths, output_lengths


class Qwen2_5OmniVisionEncoder(Qwen2_5OmniPreTrainedModel):
    config: Qwen2_5OmniVisionEncoderConfig
    _no_split_modules = ["Qwen2_5OmniVisionBlock"]
    input_modalities = ["image", "video"]

    def __init__(self, config: Qwen2_5OmniVisionEncoderConfig, *inputs, **kwargs) -> None:
        super().__init__(config, *inputs, **kwargs)
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.fullatt_block_indexes = config.fullatt_block_indexes
        self.window_size = config.window_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size

        self.patch_embed = Qwen2_5_VisionPatchEmbed(
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            in_channels=config.in_channels,
            embed_dim=config.hidden_size,
        )

        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = Qwen2_5_VisionRotaryEmbedding(head_dim // 2)
        self.blocks = nn.ModuleList([Qwen2_5OmniVisionBlock(config) for _ in range(config.depth)])
        self.cortical_adaptors = nn.ModuleList([CorticalAdaptor(config) for _ in range(config.depth)])
        # Sheet slot 0: the patch-embedding front-end. The vision band never needed this
        # slot for its own sake -- it existed only to keep the two encoder bands
        # row-aligned while the unified sheet concatenated them along columns. Under a
        # per-band tiling (`tiled32`) that constraint is gone and the slot is dropped. See
        # the audio tower for why this is a separate attribute and why the name must
        # contain "cortical_adaptor".
        self.has_frontend_slot = _sheet_has_frontend_slot(config.depth)
        if self.has_frontend_slot:
            self.frontend_cortical_adaptor = CorticalAdaptor(config)
        self.merger = Qwen2_5OmniPatchMerger(
            dim=config.out_hidden_size,
            context_dim=config.hidden_size,
            spatial_merge_size=config.spatial_merge_size,
        )
        self.gradient_checkpointing = False
    
    def _pool_cortical_states(self, cortical_states, grid_thw):
        """Reduce one adaptor's cortical states to a `[B, T, D]` sheet slot.

        Splits the flat per-patch states back into the batch, pools each group of
        `spatial_merge_unit` patches, and right-pads each item to the longest so the
        slots can be stacked. Shared by the patch-embed front-end slot and every
        transformer block.
        """
        cortical_states_list = cortical_states.split(tuple(grid_thw.prod(dim=1)), dim=0)
        max_len = max([state.shape[0] for state in cortical_states_list]) // 4
        final_cortical_states = []
        for each_cortical_state in cortical_states_list:
            each_cortical_state = each_cortical_state.reshape(-1, self.spatial_merge_unit, each_cortical_state.shape[1]).mean(dim=1)
            if max_len > each_cortical_state.shape[0]:
                zero_pad = torch.zeros((max_len - each_cortical_state.shape[0], each_cortical_state.shape[1])).to(each_cortical_state.device)
                each_cortical_state = torch.cat([each_cortical_state, zero_pad], dim=0)
            final_cortical_states.append(each_cortical_state)
        return torch.stack(final_cortical_states, dim=0)

    def rot_pos_emb(self, grid_thw):
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    def get_window_index(self, grid_thw):
        window_index: list = []
        cu_window_seqlens: list = [0]
        window_index_id = 0
        vit_merger_window_size = self.window_size // self.spatial_merge_size // self.patch_size

        for grid_t, grid_h, grid_w in grid_thw:
            llm_grid_h, llm_grid_w = (
                grid_h // self.spatial_merge_size,
                grid_w // self.spatial_merge_size,
            )
            index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(grid_t, llm_grid_h, llm_grid_w)
            pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
            pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
            num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
            num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
            index_padded = F.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
            index_padded = index_padded.reshape(
                grid_t,
                num_windows_h,
                vit_merger_window_size,
                num_windows_w,
                vit_merger_window_size,
            )
            index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
                grid_t,
                num_windows_h * num_windows_w,
                vit_merger_window_size,
                vit_merger_window_size,
            )
            seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
            index_padded = index_padded.reshape(-1)
            index_new = index_padded[index_padded != -100]
            window_index.append(index_new + window_index_id)
            cu_seqlens_tmp = seqlens.cumsum(0) * self.spatial_merge_unit + cu_window_seqlens[-1]
            cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
            window_index_id += (grid_t * llm_grid_h * llm_grid_w).item()
        window_index = torch.cat(window_index, dim=0)

        return window_index, cu_window_seqlens

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(seq_len, hidden_size)`):
                The final hidden states of the model.
            grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
                The temporal, height and width of feature shape of each image in LLM.

        Returns:
            `torch.Tensor`: hidden_states.
        """
        hidden_states = self.patch_embed(hidden_states)
        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        window_index, cu_window_seqlens = self.get_window_index(grid_thw)
        cu_window_seqlens = torch.tensor(
            cu_window_seqlens,
            device=hidden_states.device,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        hidden_states = hidden_states[window_index, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            # Select dtype based on the following factors:
            #  - FA2 requires that cu_seqlens_q must have dtype int32
            #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
            # See https://github.com/huggingface/transformers/pull/34852 for more information
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        reverse_indices = torch.argsort(window_index)

        # Front-end cortical adaptor. Applied here rather than straight after
        # `patch_embed` because `hidden_states` has since been permuted into window
        # order, which is the layout every block adaptor below sees -- tapping
        # pre-permutation would put slot 0 in a different patch order from slots
        # 1..L and scramble it relative to the rest of the band.
        cortical_sheet = ()
        if self.has_frontend_slot:
            hidden_states, frontend_cortical_states = self.frontend_cortical_adaptor(hidden_states)
            # Slot 0 is the patch-embed front-end; slots 1..L are the transformer blocks.
            cortical_sheet = (self._pool_cortical_states(frontend_cortical_states, grid_thw),)

        # Modification here
        for layer_num, blk in enumerate(self.blocks):
            if layer_num in self.fullatt_block_indexes:
                cu_seqlens_now = cu_seqlens
            else:
                cu_seqlens_now = cu_window_seqlens

            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens_now,
                rotary_pos_emb=rotary_pos_emb,
                **kwargs,
            )

            # Cortical adaptor
            hidden_states, cortical_states = self.cortical_adaptors[layer_num](hidden_states)

            cortical_sheet += (self._pool_cortical_states(cortical_states, grid_thw),)

            # cortical_sheet += (cortical_states.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1).mean(dim=1),)

        hidden_states = self.merger(hidden_states)
        hidden_states = hidden_states[reverse_indices, :]

        return hidden_states, cortical_sheet


@auto_docstring
class Qwen2_5OmniThinkerTextModel(Qwen2_5OmniPreTrainedModel):
    config: Qwen2_5OmniTextConfig
    input_modalities = "text"
    _no_split_modules = ["Qwen2_5OmniDecoderLayer"]

    def __init__(self, config: Qwen2_5OmniTextConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([Qwen2_5OmniDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.cortical_adaptors = nn.ModuleList([CorticalAdaptor(config) for _ in range(config.num_hidden_layers)])
        self._attn_implementation = config._attn_implementation
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types
        self.rotary_emb = Qwen2_5OmniRotaryEmbedding(config=config)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # torch.jit.trace() doesn't support cache objects in the output
        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            past_key_values = DynamicCache(config=self.config)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        # the hard coded `3` is for temporal, height and width.
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        # NOTE: we need to pass text position ids for packing. Qwen2-VL uses 3D positions
        # where each dim indicates visual spatial positions for temporal/height/width grids.
        # There are two scenarios when FA2-like packed masking might be activated.
        # 1. User specifically passed packed `position_ids` and no attention mask.
        #    In this case we expect the useer to create correct position ids for all 3 grids
        #    and prepend text-only position ids to it. The final tensor will be [4, bs, seq-len]
        # 2. User runs forward with no attention mask and no position ids. In this case, position ids
        #    are prepared by the model (`get_rope_index`) as `[4, bs, seq-len]` tensor. Text-only positions are
        #    prepended by us when creating positions so that the mask is constructed correctly. NOTE: failing to pass
        #    text-only positions will cause incorrect mask construction, do not change `prepare_input_for_generation`
        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            # If inputs are not packed (usual 3D positions), do not prepare mask from position_ids
            text_position_ids = None

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": text_position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            # The sliding window alternating layers are not always activated depending on the config
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        cortical_sheet = ()

        for layer_idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[ self.config.layer_types[layer_idx]],
                position_embeddings=position_embeddings,
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

            hidden_states = layer_outputs[0]

            if len(hidden_states.shape) == 2:
                hidden_states = hidden_states.unsqueeze(0)

            # Cortical adaptor
            hidden_states, cortical_states = self.cortical_adaptors[layer_idx](hidden_states)

            cortical_sheet += (cortical_states,)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(
                v for v in [hidden_states, past_key_values, all_hidden_states, all_self_attns] if v is not None
            )
        return TopoBaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cortical_sheet=cortical_sheet,
        )


@auto_docstring(
    custom_intro="""
    The Qwen2.5OmniThinker model which consists of a audio backbone and a language model.
    """
)
class Qwen2_5OmniThinkerForConditionalGeneration(Qwen2_5OmniPreTrainedModelForConditionalGeneration, GenerationMixin):
    config: Qwen2_5OmniThinkerConfig
    base_model_prefix = "thinker"
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _no_split_modules = ["Qwen2_5OmniAudioEncoder", "Qwen2_5OmniVisionEncoder"]

    def __init__(self, config: Qwen2_5OmniThinkerConfig):
        super().__init__(config)
        self.audio_tower = Qwen2_5OmniAudioEncoder._from_config(config.audio_config)
        self.visual = Qwen2_5OmniVisionEncoder._from_config(config.vision_config)
        self.vocab_size = config.text_config.vocab_size
        self.model = Qwen2_5OmniThinkerTextModel._from_config(config.text_config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        self.spatial_merge_size = config.vision_config.spatial_merge_size
        self.rope_deltas = None
        self.position_dir = config.position_dir if hasattr(config, "position_dir") else None
        self.apply_spatial_loss = config.apply_spatial_loss if hasattr(config, "apply_spatial_loss") else True

        self.alpha = config.alpha if hasattr(config, "alpha") else 5
        self.post_init()

    def load_positions(self):
        """
        Build the cortical coordinates and, in dynamic mode, the per-unit
        neighborhood radii used by the spatial loss.

        In `dynamic` mode nothing is read from disk beyond `coords.npy`:
        neighborhoods are resolved on the fly in `spatial_loss_fn`, with a radius
        that grows with depth (small in the early vision / audio encoder layers,
        `radius_max` by the last encoder layer and throughout the thinker).

        In `fixed` mode (the default, and what every pre-existing config gets) the
        precomputed single-radius neighborhood table is loaded as before.
        """

        neighborhood_cfg = getattr(self.config, "neighborhood", None) or {}
        mode = neighborhood_cfg.get("mode", "fixed")
        neighborhoods_per_batch = int(neighborhood_cfg.get("neighborhoods_per_batch", 100))

        if mode not in ("dynamic", "fixed"):
            raise ValueError(
                f"unknown neighborhood mode '{mode}', expected 'dynamic' or 'fixed'"
            )
        if mode == "fixed" and self.position_dir is None:
            raise ValueError("position_dir is not set in the configuration.")

        # Must be the SAME object the forward pass reshapes with, or the coordinates
        # would describe a different geometry from the sheet they index into.
        layout = DEFAULT_SHEET_LAYOUT
        # An encoder band holds either one slot per transformer block, or that plus one
        # for the conv front-end (audio `conv2` / vision `patch_embed`). Anything else is
        # a layout/config mismatch, and the towers read the same condition through
        # `_sheet_has_frontend_slot`, so the two cannot drift apart.
        n_blocks = self.config.vision_config.depth
        if (layout.num_encoder_layers not in (n_blocks, n_blocks + 1)
                or layout.num_thinker_layers != self.config.text_config.num_hidden_layers):
            raise ValueError(
                f"sheet layout has {layout.num_encoder_layers} encoder slots / "
                f"{layout.num_thinker_layers} thinker layers but the config has "
                f"{n_blocks} blocks (+1 iff the layout carries a front-end slot) / "
                f"{self.config.text_config.num_hidden_layers}"
            )
        print(f">> Sheet layout '{layout.mode}': {layout.n_row}x{layout.n_col}"
              f" | front-end slot: "
              f"{'on' if layout.num_encoder_layers == n_blocks + 1 else 'off'}")

        # the sheet geometry drives the reshapes in `forward`; fail loudly rather than
        # silently mis-assigning depths if a config ever drifts from it. How many units a
        # layer owns is rows*cols of its strip (strip mode) or of its tile (tiled mode).
        if layout.mode == "strip":
            enc_units = layout.rows_per_encoder_layer * (layout.n_col // 2)
            thk_units = layout.rows_per_thinker_layer * layout.n_col
        else:
            enc_units = layout.enc_tile[0] * layout.enc_tile[1]
            thk_units = layout.thk_tile[0] * layout.thk_tile[1]
        # A layer occupies `cortical_dim` sheet units, which equals its hidden size only for
        # a square adaptor. A widened adaptor buys sheet extent, so compare against the
        # adaptor width actually configured.
        want_enc = getattr(self.config.vision_config, "cortical_dim", None) \
                   or self.config.vision_config.hidden_size
        want_thk = getattr(self.config.text_config, "cortical_dim", None) \
                   or self.config.text_config.hidden_size
        if enc_units != want_enc:
            raise ValueError(
                f"sheet layout {layout.mode} gives {enc_units} units per encoder slot, "
                f"but the vision adaptor is {want_enc} wide"
            )
        if thk_units != want_thk:
            raise ValueError(
                f"sheet layout {layout.mode} gives {thk_units} units per thinker layer, "
                f"but the thinker adaptor is {want_thk} wide"
            )

        coordinates = self._load_or_build_coordinates(
            layout, seed=int(neighborhood_cfg.get("coords_seed", 42))
        )

        unit_radii = None
        neighbor_indices = None

        if mode == "dynamic":
            schedule = RadiusSchedule(
                radius_min=float(neighborhood_cfg.get("radius_min", 5)),
                radius_max=float(neighborhood_cfg.get("radius_max", 25)),
                schedule=str(neighborhood_cfg.get("schedule", "linear")),
            )
            unit_radii = build_unit_radii(coordinates, layout, schedule)
            print(">> " + describe_radius_schedule(layout, schedule))
        else:
            print(">> Loading neighborhoods from ", self.position_dir)
            neighbor_indices = torch.from_numpy(
                np.load(f"{self.position_dir}/neighborhood_indices.npy")
            )

        # Held for a whole accumulation group when > 1; see spatial_loss_fn.
        center_reuse = int(neighborhood_cfg.get("center_reuse", 1) or 1)

        self.positions = LayerPositions(
            "unified_sheet",
            coordinates,
            neighbor_indices,
            neighborhoods_per_batch,
            unit_radii,
        )

        self.positions.center_reuse = center_reuse
        if center_reuse > 1:
            print(f">> Centre reuse: one draw held for {center_reuse} forward(s)")

        # kept for depth-stratified center sampling in `spatial_loss_fn`
        self.sheet_layout = layout

        # Gradient-balance options. All default False so an existing config reproduces the
        # original loss exactly; see `spatial_loss_fn` for what each one fixes.
        self.spatial_drop_dead_units = bool(neighborhood_cfg.get("drop_dead_units", False))
        self.spatial_scale_balance = bool(neighborhood_cfg.get("scale_balance", False))
        self.spatial_stratify_by_depth = bool(neighborhood_cfg.get("stratify_by_depth", False))
        self.spatial_center_allocation = str(neighborhood_cfg.get("center_allocation", "uniform"))
        _mu = neighborhood_cfg.get("max_neighborhood_units", None)
        self.spatial_max_units = None if _mu in (None, 0) else int(_mu)
        print(f">> Spatial loss options: drop_dead_units={self.spatial_drop_dead_units}, "
              f"scale_balance={self.spatial_scale_balance}, "
              f"stratify_by_depth={self.spatial_stratify_by_depth}, "
              f"center_allocation={self.spatial_center_allocation}, "
              f"max_units={self.spatial_max_units}")

        # --- optional coarse, tile-scoped term ---------------------------------------
        # A second spatial-loss term with a much larger radius, every neighborhood confined
        # to its center's own layer tile. This is what grows a large within-layer patch:
        # the map's wavelength tracks the radius (~2.4x, measured), so the fine term above
        # can only make Lambda ~ 12px clusters no matter how long it trains. Scoping it to
        # a tile is what keeps the large radius meaningful -- unscoped, a tile-wide
        # neighborhood spills into the layers tiled next to it and the loss ends up
        # organizing the tile grid instead of the layer. The fine term stays UNSCOPED on
        # purpose: its cross-tile pairs are the only thing asking adjacent layers to
        # resemble each other.
        coarse_cfg = neighborhood_cfg.get("coarse", None)
        self.positions_coarse = None
        self.spatial_coarse_weight = 0.0
        self.spatial_coarse_max_units = None

        if coarse_cfg:
            if layout.mode != "tiled":
                raise ValueError(
                    f"the coarse tile-scoped term needs a tiled layout, got '{layout.mode}'"
                )
            coarse_schedule = RadiusSchedule(
                radius_min=float(coarse_cfg.get("radius_min", 4)),
                radius_max=float(coarse_cfg.get("radius_max", 32)),
                schedule=str(coarse_cfg.get("schedule", "linear")),
            )
            coarse_positions = LayerPositions(
                "unified_sheet_coarse",
                coordinates,
                None,
                int(coarse_cfg.get("neighborhoods_per_batch", 500)),
                build_unit_radii(coordinates, layout, coarse_schedule),
            )
            # OFF by default. Scoping was the obvious way to keep a tile-wide radius
            # from bleeding into the layers tiled next door -- but bleeding across a tile
            # border is exactly what a UNIFIED sheet is for: a cluster should be free to
            # span adjacent layers rather than stop dead at a layer boundary, and hard
            # scoping would etch the tile grid into every map. Serpentine tile order (see
            # SheetLayout.tile_order) makes consecutive layers spatially adjacent so that
            # spanning is continuous. Kept available as an ablation: turning it on is the
            # clean test of how much the cross-tile pairs actually contribute.
            if coarse_cfg.get("tile_scoped", False):
                attach_tile_scoping(coarse_positions, layout)
            coarse_positions.center_reuse = center_reuse

            self.positions_coarse = coarse_positions
            self.spatial_coarse_weight = float(coarse_cfg.get("weight", 1.0))
            _cmu = coarse_cfg.get("max_neighborhood_units", 768)
            self.spatial_coarse_max_units = None if _cmu in (None, 0) else int(_cmu)

            print(">> Coarse spatial-loss term: "
                  + describe_radius_schedule(layout, coarse_schedule))
            print(f">> Coarse term: weight={self.spatial_coarse_weight}, "
                  f"centers={coarse_positions.neighborhoods_per_batch}, "
                  f"max_units={self.spatial_coarse_max_units}, "
                  f"tile_scoped={coarse_positions.tile_scoped}, "
                  f"tile_order={layout.tile_order}")

        print(f">> Positions loaded ({neighborhoods_per_batch} neighborhoods per batch).")

    def _load_or_build_coordinates(self, layout: SheetLayout, seed: int = 42) -> torch.Tensor:
        coords_path = (
            f"{self.position_dir}/coords.npy" if self.position_dir is not None else None
        )

        if coords_path is not None and os.path.exists(coords_path):
            print(">> Loading coordinates from ", coords_path)
            coordinates = torch.from_numpy(np.load(coords_path)).float()
            # A coords.npy built for a different sheet geometry has the wrong number
            # of units, but nothing downstream would notice: the spatial loss would
            # just index a mismatched set and produce a scrambled map instead of an
            # error. Cheap to check, so check.
            # Unit COUNT alone is not enough: the strip (309x512) and tiled (824x192)
            # layouts have exactly the same 158,208 units, so check the extent too or a
            # strip coords.npy would load silently into a tiled run.
            got = (int(coordinates.shape[0]),
                   int(coordinates[:, 0].max()) + 1, int(coordinates[:, 1].max()) + 1)
            want = (layout.num_units, layout.n_row, layout.n_col)
            if got != want:
                raise ValueError(
                    f"{coords_path} holds {got[0]} units spanning {got[1]}x{got[2]}, but "
                    f"the '{layout.mode}' sheet layout is {want[1]}x{want[2]} "
                    f"({want[0]} units). Regenerate coordinates with "
                    f"`TOPO_SHEET_LAYOUT={os.environ.get('TOPO_SHEET_LAYOUT', 'strip')} "
                    f"python -m src.training.init_coords` after pointing `position-dir` at a "
                    f"fresh directory."
                )
            return coordinates

        print(f">> No coords.npy found, rebuilding sheet coordinates from seed {seed}.")
        return build_sheet_coordinates(layout, seed=seed)

    def init_cortical_layers(self, epsilon: float = 1e-3, identity=False):
        for module in self.model.modules():
            if isinstance(module, CorticalAdaptor):
                module.initialize(epsilon=epsilon, identity=identity)

        for module in self.visual.modules():
            if isinstance(module, CorticalAdaptor):
                module.initialize(epsilon=epsilon, identity=identity)

        for module in self.audio_tower.modules():
            if isinstance(module, CorticalAdaptor):
                module.initialize(epsilon=epsilon, identity=identity)

        self.refresh_cortical_pinv()

    def refresh_cortical_pinv(self):
        """Materialize every adaptor's cached ``pinv(W)`` up front.

        Adaptors fill this cache lazily on their first forward (it cannot be built at
        construction time, when `from_pretrained` still has W on the meta device). Calling
        this once after loading keeps the cost out of the first forward, and fills the
        towers a given input never reaches.
        """
        for module in self.modules():
            if isinstance(module, CorticalAdaptor) and module._pinv_stale and "W_pinv" in module._buffers:
                module._recompute_pinv()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_video_features(
        self, pixel_values_videos: torch.FloatTensor, video_grid_thw: Optional[torch.LongTensor] = None
    ):
        """
        Encodes videos into continuous embeddings that can be forwarded to the language model.

        Args:
            pixel_values_videos (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
                The tensors corresponding to the input videos.
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
        """
        pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
        video_embeds, cortical_sheet = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
        return video_embeds, cortical_sheet

    def get_image_features(self, pixel_values: torch.FloatTensor, image_grid_thw: Optional[torch.LongTensor] = None):
        """
        Encodes images into continuous embeddings that can be forwarded to the language model.

        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
                The tensors corresponding to the input images.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
        """
        pixel_values = pixel_values.type(self.visual.dtype)
        image_embeds, cortical_sheet = self.visual(pixel_values, grid_thw=image_grid_thw)
        return image_embeds, cortical_sheet

    def get_audio_features(
        self,
        input_features: torch.FloatTensor,
        feature_attention_mask: Optional[torch.LongTensor] = None,
        audio_feature_lengths: Optional[torch.LongTensor] = None,
    ):
        """
        Encodes audios into continuous embeddings that can be forwarded to the language model.

        Args:
            input_features (`torch.FloatTensor`):
                The tensors corresponding to the input audios.
            feature_attention_mask (`torch.LongTensor`, *optional*):
                Mask to avoid performing attention on padding feature indices. Mask values selected in `[0, 1]`:
            audio_feature_lengths (`torch.LongTensor` of shape `(num_audios)`, *optional*):
                The length of feature shape of each audio in LLM.
        """
        if feature_attention_mask is not None:
            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
            input_features = input_features.permute(0, 2, 1)[feature_attention_mask.bool()].permute(1, 0)
        else:
            audio_feature_lengths = None

        audio_feat_lengths, audio_output_lengths = self.audio_tower._get_feat_extract_output_lengths(
            audio_feature_lengths if audio_feature_lengths is not None else feature_attention_mask.sum(-1)
        )
        feature_lens = audio_feature_lengths if audio_feature_lengths is not None else feature_attention_mask.sum(-1)
        audio_outputs = self.audio_tower(
            input_features,
            feature_lens=feature_lens,
            aftercnn_lens=audio_feat_lengths,
        )
        audio_features = audio_outputs.last_hidden_state
        cortical_sheet = audio_outputs.cortical_sheet

        if audio_features.shape[0] != sum(audio_output_lengths.tolist()):
            raise ValueError("length of audio_features should match audio_output_lengths")
        
        return audio_features, cortical_sheet

    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        image_features: Optional[torch.FloatTensor] = None,
        video_features: Optional[torch.FloatTensor] = None,
    ):
        """
        Obtains multimodal placeholder mask from `input_ids` or `inputs_embeds`, and checks that the placeholder token count is
        equal to the length of multimodal features. If the lengths are different, an error is raised.
        """
        if input_ids is None:
            special_image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_image_mask = special_image_mask.all(-1)
            special_video_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.video_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_video_mask = special_video_mask.all(-1)
            special_audio_mask = (
                inputs_embeds
                == self.get_input_embeddings()(
                    torch.tensor(self.config.audio_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
            ).all(-1)
        else:
            special_image_mask = input_ids == self.config.image_token_id
            special_video_mask = input_ids == self.config.video_token_id
            special_audio_mask = input_ids == self.config.audio_token_id

        n_image_tokens = special_image_mask.sum()
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if image_features is not None and inputs_embeds[special_image_mask].numel() != image_features.numel():
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {image_features.shape[0]}"
            )

        n_video_tokens = special_video_mask.sum()
        special_video_mask = special_video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if video_features is not None and inputs_embeds[special_video_mask].numel() != video_features.numel():
            raise ValueError(
                f"Videos features and image tokens do not match: tokens: {n_video_tokens}, features {video_features.shape[0]}"
            )

        special_audio_mask = special_audio_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        return special_image_mask, special_video_mask, special_audio_mask

    def print_tokens(self, input_ids):
        from transformers import Qwen2_5OmniProcessor
        processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-3B")

        counter = {}
        prev_token_id = None
        for token_id in input_ids.tolist():
            counter[token_id] = counter.get(token_id, 0) + 1
            if token_id != prev_token_id and prev_token_id is not None:
                decoded_token = processor.decode(prev_token_id)
                print(f"Token ID: {decoded_token}, Count: {counter[prev_token_id]}")
                counter[prev_token_id] = 0
            prev_token_id = token_id

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        input_features: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        feature_attention_mask: Optional[torch.Tensor] = None,
        audio_feature_lengths: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        use_audio_in_video: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        video_second_per_grid: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, Qwen2_5OmniThinkerCausalLMOutputWithPast]:
        r"""
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        feature_attention_mask (`torch.Tensor` of shape `(batch_size, feature_sequence_length)`, *optional*):
            Mask to avoid performing attention on padding feature indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        audio_feature_lengths (`torch.LongTensor` of shape `(num_audios)`, *optional*):
            The length of feature shape of each audio in LLM.
        rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
            The rope index difference between sequence length and multimodal rope.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        use_audio_in_video (`bool`, *optional*):
            Whether or not use audio track in video, should same as the parameter in `process_audio_info`.
        video_second_per_grid (`torch.LongTensor` of shape `(num_videos)`, *optional*):
            Number of seconds per grid for each video, used for temporal feature mapping.

        Example:

        ```python
        >>> from io import BytesIO
        >>> from urllib.request import urlopen
        >>> import librosa
        >>> from qwen_vl_utils import process_vision_info
        >>> from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration

        >>> thinker = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-Omni-7B")
        >>> processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")

        >>> conversations = [
        >>>         {'role': 'system', 'content': 'You are a helpful voice chat bot, and please respond to me in a casual conversation manner using random voice.'},
        >>>         {"role": "user", "content": [
        >>>             {"type": "image", "image_url": "https://www.ilankelman.org/stopsigns/australia.jpg"},
        >>>             {"type": "audio", "audio_url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/glass-breaking-151256.mp3"},
        >>>         ]},
        >>> ]

        >>> text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        >>> audios = [ librosa.load(BytesIO(urlopen( conversations[1]['content'][1]['audio_url'] ).read()), sr=self.processor.feature_extractor.sampling_rate) ]
        >>> images, videos = process_vision_info(conversations)
        >>> inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True)

        >>> # Generate
        >>> inputs['use_audio_in_video'] = `True` or `False`
        >>> generation = thinker.generate(**inputs, max_new_tokens=2048)
        >>> generate_ids = generation[:, inputs.input_ids.size(1):]

        >>> response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        ```"""

        if cache_position is not None and cache_position[0] != 0:
            pixel_values = None
            pixel_values_videos = None
            input_features = None

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            # 1. Extract the input embeddings
            inputs_embeds = self.get_input_embeddings()(input_ids)

        # 2. Merge text , audios , image and video
        audio_cortical_sheet = None
        if input_features is not None:
            audio_features, audio_cortical_sheet = self.get_audio_features(
                input_features,
                feature_attention_mask=feature_attention_mask,
                audio_feature_lengths=audio_feature_lengths,
            )
            audio_features = audio_features.to(inputs_embeds.device, inputs_embeds.dtype)
            _, _, audio_mask = self.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_features)

        visual_cortical_sheet = None
        if pixel_values is not None:
            image_embeds, visual_cortical_sheet = self.get_image_features(pixel_values, image_grid_thw)
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)


        if pixel_values_videos is not None:
            video_embeds, visual_cortical_sheet = self.get_video_features(pixel_values_videos, video_grid_thw)
            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        if feature_attention_mask is not None:
            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
        else:
            audio_feature_lengths = None

        if attention_mask is not None and position_ids is None:
            if (
                cache_position is None
                or (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
            ):
                delta0 = (1 - attention_mask).sum(dim=-1).unsqueeze(1)
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    attention_mask,
                    use_audio_in_video,
                    audio_feature_lengths,
                    video_second_per_grid,
                )
                rope_deltas = rope_deltas - delta0
                self.rope_deltas = rope_deltas
            else:
                batch_size, seq_length = input_ids.shape
                delta = cache_position[0] + self.rope_deltas if cache_position is not None else 0
                position_ids = torch.arange(seq_length, device=input_ids.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]
        multimodal_cortical_sheet = outputs.cortical_sheet

        logits = self.lm_head(hidden_states)

        visual_cortical_sheet = torch.stack(visual_cortical_sheet, dim=0) if visual_cortical_sheet is not None else None
        audio_cortical_sheet = torch.stack(audio_cortical_sheet, dim=0) if audio_cortical_sheet is not None else None
        multimodal_cortical_sheet = torch.stack(multimodal_cortical_sheet, dim=0) if multimodal_cortical_sheet is not None else None

        loss = None
        task_loss = None
        spatial_loss = None
        fine_spatial_loss = None
        coarse_spatial_loss = None
        
        unified_sheet = None

        if self.apply_spatial_loss:

            # average every 2 seconds for cortical sheet
            audio_time_steps_all = []
            audio_cortical_sheets_final = []
            if audio_cortical_sheet is not None:
                # Tokens per spatial-loss SAMPLE. The cortical states are pooled over
                # `audio_chunk` tokens, and each pooled step becomes one row of
                # `activations` -- i.e. one sample in the unit x unit `corrcoef` the
                # spatial loss is built on. So this directly sets how many samples that
                # correlation is estimated from: smaller chunk -> more samples -> less
                # noisy r, at the cost of more memory for the sheet.
                #
                # The video and text chunks are DERIVED from the resulting audio step
                # count (see `video_chunk` below), so lowering this raises the sample
                # count for every modality, keeping the three bands time-aligned.
                #
                # 50 reproduces the original behaviour exactly.
                audio_chunk = int(getattr(self.config, "spatial_time_chunk", 50))
                audio_cortical_sheet = audio_cortical_sheet.permute(1,0,2,3)
                for cortical_sheet in audio_cortical_sheet:
                    num_layers, _, hidden_dim = cortical_sheet.shape
                    mask = cortical_sheet.abs().sum(dim=-1) != 0
                    cortical_sheet = cortical_sheet[mask].view(num_layers, -1, hidden_dim)
                    audio_time_steps = cortical_sheet.shape[1] if input_features is not None else 0
                    audio_time_steps = audio_time_steps - (audio_time_steps % audio_chunk)
                    cortical_sheet = cortical_sheet[:, :audio_time_steps] 
                    num_layers, _, dim = cortical_sheet.shape
                    # mean over the activations of every 2 seconds chunk
                    cortical_sheet = cortical_sheet.view(num_layers, -1, audio_chunk, dim).mean(dim=2) 
                    cortical_sheet = cortical_sheet.to(inputs_embeds.device) 
                    audio_time_steps = cortical_sheet.shape[1]
                    audio_time_steps_all.append(audio_time_steps)
                    audio_cortical_sheets_final.append(cortical_sheet)

                audio_cortical_sheet = torch.cat(audio_cortical_sheets_final, dim=1)

            video_time_steps = None
            video_cortical_sheets_final = []
            video_chunks = []
            if visual_cortical_sheet is not None:

                visual_cortical_sheet = visual_cortical_sheet.permute(1,0,2,3)
                for j, cortical_sheet in enumerate(visual_cortical_sheet):
                    num_layers, _, hidden_dim = cortical_sheet.shape
                    mask = cortical_sheet.abs().sum(dim=-1) != 0
                    cortical_sheet = cortical_sheet[mask].view(num_layers, -1, hidden_dim)
                    video_time_steps = cortical_sheet.shape[1] if pixel_values_videos is not None or pixel_values is not None else 0
                    
                    if len(audio_time_steps_all) == 0:
                        video_chunk = video_time_steps
                    else:
                        video_chunk = (video_time_steps // audio_time_steps_all[j]) if audio_time_steps_all[j] > 0 else 598 # 2 second video chunk

                    video_chunks.append(video_chunk)
                    
                    video_time_steps = video_time_steps - (video_time_steps % video_chunk)
                    cortical_sheet = cortical_sheet[:, :video_time_steps] 
                    num_layers, _, dim = cortical_sheet.shape
                    cortical_sheet = cortical_sheet.view(num_layers, -1, video_chunk, dim).mean(dim=2) 
                    cortical_sheet = cortical_sheet.to(inputs_embeds.device) 
                    video_cortical_sheets_final.append(cortical_sheet)

                visual_cortical_sheet = torch.cat(video_cortical_sheets_final, dim=1)

            multimodal_cortical_sheets_final = []
            if multimodal_cortical_sheet is not None:

                multimodal_cortical_sheet = multimodal_cortical_sheet.permute(1,0,2,3)
                text_time_steps = multimodal_cortical_sheet.shape[2] 

                for j, cortical_sheet in enumerate(multimodal_cortical_sheet):
                    
                    if len(audio_time_steps_all) > 0:
                        audio_start_token_id = 151647
                        audio_end_token_id = 151648
                        # find audio start index
                        audio_start_indices = (input_ids[j] == audio_start_token_id).nonzero(as_tuple=True)[0].item()
                        audio_end_indices = (input_ids[j] == audio_end_token_id).nonzero(as_tuple=True)[0].item()
                        cortical_sheet = cortical_sheet[:, audio_start_indices:audio_end_indices]
                        text_time_steps = cortical_sheet.shape[1]
                        if video_time_steps is not None:
                            video_audio_chunk = video_chunks[j] + audio_chunk # 2 second video + audio chunk
                        else:
                            video_audio_chunk = audio_chunk
                    elif video_time_steps is not None:
                        video_audio_chunk = video_chunks[j]
                    else:
                        video_audio_chunk = text_time_steps

                    text_time_steps = text_time_steps - (text_time_steps % video_audio_chunk)
                    cortical_sheet = cortical_sheet[:, :text_time_steps]
                    num_layers, _, dim = cortical_sheet.shape
                    cortical_sheet = cortical_sheet.view(num_layers, -1, video_audio_chunk, dim).mean(dim=2)
                    cortical_sheet = cortical_sheet.to(inputs_embeds.device) 
                    multimodal_cortical_sheets_final.append(cortical_sheet)

                multimodal_cortical_sheet = torch.cat(multimodal_cortical_sheets_final, dim=1)
                multimodal_time = multimodal_cortical_sheet.shape[1] #if multimodal_cortical_sheet is not None else 0

                _LO = DEFAULT_SHEET_LAYOUT
                _ER, _EC = _LO.encoder_rows, _LO.n_col // 2
                _mk = lambda st: (_band_from_layers(st, _LO, _ER, _EC, _LO.enc_tile,
                                                    inputs_embeds.device)
                                  if st is not None else
                                  torch.zeros((multimodal_time, _ER, _EC),
                                              device=inputs_embeds.device))
                visual_cortical_sheet = _mk(visual_cortical_sheet)
                audio_cortical_sheet = _mk(audio_cortical_sheet)
                #^ [L, T, D] -> [T, encoder_rows, n_col/2]. L = 32 transformer blocks,
                #  plus a leading conv/patch front-end slot on the layouts that carry one
                #  (L=33). strip: each layer a full-width strip; tiled: each layer a 2-D
                #  tile (see _band_from_layers).

                min_len = min(visual_cortical_sheet.shape[0], audio_cortical_sheet.shape[0])
                visual_cortical_sheet = visual_cortical_sheet[:min_len]
                audio_cortical_sheet = audio_cortical_sheet[:min_len]            

                unified_sheet = torch.cat([visual_cortical_sheet, audio_cortical_sheet], dim=-1)
                #^ [T, 2D/n, n*L]

                multimodal_cortical_sheet = _band_from_layers(
                    multimodal_cortical_sheet, _LO, _LO.thinker_rows, _LO.n_col,
                    _LO.thk_tile, inputs_embeds.device)
                #^ [L, T, D] -> [T, thinker_rows, n_col], L=36, D=2048

                min_len = min(unified_sheet.shape[0], multimodal_cortical_sheet.shape[0])
                unified_sheet = unified_sheet[:min_len]
                multimodal_cortical_sheet = multimodal_cortical_sheet[:min_len]

                unified_sheet = torch.cat([unified_sheet, multimodal_cortical_sheet], dim=1)
                #^ [T, n_row, n_col] -- 309x512 (strip) or 824x192 (tiled)

                num_time_steps = unified_sheet.shape[0]

        if labels is not None:
            task_loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.get_text_config().vocab_size
            )

            if unified_sheet is not None and self.apply_spatial_loss:
                activations = unified_sheet.reshape(num_time_steps, -1)
                self.last_spatial_stats = {}
                fine_spatial_loss = spatial_loss_fn(
                    activations=activations,
                    positions=self.positions,
                    accum=self.config.accum,
                    drop_dead_units=getattr(self, "spatial_drop_dead_units", False),
                    scale_balance=getattr(self, "spatial_scale_balance", False),
                    stratify_by_depth=getattr(self, "spatial_stratify_by_depth", False),
                    center_allocation=getattr(self, "spatial_center_allocation", "uniform"),
                    max_units=getattr(self, "spatial_max_units", None),
                    layout=getattr(self, "sheet_layout", None),
                    stats=self.last_spatial_stats,
                )
                unified_spatial_loss = fine_spatial_loss
                if getattr(self, "positions_coarse", None) is not None:
                    self.last_spatial_stats_coarse = {}
                    coarse_spatial_loss = spatial_loss_fn(
                        activations=activations,
                        positions=self.positions_coarse,
                        accum=self.config.accum,
                        drop_dead_units=getattr(self, "spatial_drop_dead_units", False),
                        scale_balance=getattr(self, "spatial_scale_balance", False),
                        stratify_by_depth=getattr(self, "spatial_stratify_by_depth", False),
                        center_allocation=getattr(self, "spatial_center_allocation", "uniform"),
                        max_units=getattr(self, "spatial_coarse_max_units", None),
                        layout=getattr(self, "sheet_layout", None),
                        stats=self.last_spatial_stats_coarse,
                    )
                    unified_spatial_loss = (
                        fine_spatial_loss
                        + self.spatial_coarse_weight * coarse_spatial_loss
                    )

                spatial_loss = self.alpha * unified_spatial_loss
            else:
                print(">> Warning: unified_sheet is None, spatial loss set to 0.")
                spatial_loss = 0.0

            loss = spatial_loss + task_loss 

        if not return_dict:
            output = (logits,) + outputs
            return (loss,) + output if loss is not None else output

        return Qwen2_5OmniThinkerCausalLMOutputWithPast(
            loss=loss,
            task_loss=task_loss.item() if task_loss is not None and not isinstance(task_loss, float) else task_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
            spatial_loss=spatial_loss.item() if spatial_loss is not None and not isinstance(spatial_loss, float) else spatial_loss,
            fine_spatial_loss=fine_spatial_loss.item() if fine_spatial_loss is not None and not isinstance(fine_spatial_loss, float) else fine_spatial_loss,
            coarse_spatial_loss=coarse_spatial_loss.item() if coarse_spatial_loss is not None and not isinstance(coarse_spatial_loss, float) else coarse_spatial_loss,
            multimodal_cortical_sheet=multimodal_cortical_sheet.detach().cpu() if multimodal_cortical_sheet is not None else None,
            visual_cortical_sheet=visual_cortical_sheet.detach().cpu() if visual_cortical_sheet is not None else None,
            audio_cortical_sheet=audio_cortical_sheet.detach().cpu() if audio_cortical_sheet is not None else None,
            unified_sheet=unified_sheet.detach().cpu() if unified_sheet is not None else None,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        input_features=None,
        feature_attention_mask=None,
        use_audio_in_video=False,
        video_second_per_grid=None,
        **kwargs,
    ):

        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            use_cache=use_cache,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            use_audio_in_video=use_audio_in_video,
            video_second_per_grid=video_second_per_grid,
            **kwargs,
        )

        model_inputs["position_ids"] = None

        if cache_position[0] != 0:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None
            model_inputs["input_features"] = None

        return model_inputs




__all__ = [
    "Qwen2_5OmniForConditionalGeneration",
    "Qwen2_5OmniThinkerTextModel",
    "Qwen2_5OmniThinkerForConditionalGeneration",
    "Qwen2_5OmniTalkerModel",
    "Qwen2_5OmniTalkerForConditionalGeneration",
    "Qwen2_5OmniToken2WavDiTModel",
    "Qwen2_5OmniToken2WavBigVGANModel",
    "Qwen2_5OmniToken2WavModel",
    "Qwen2_5OmniPreTrainedModel",
    "Qwen2_5OmniPreTrainedModelForConditionalGeneration",
]