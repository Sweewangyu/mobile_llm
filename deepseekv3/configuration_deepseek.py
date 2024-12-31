from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)

DEEPSEEK_PRETRAINED_CONFIG_ARCHIVE_MAP = {}
class DeepseekV3Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`DeepseekV3Model`]. It is used to instantiate an DeepSeek
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the DeepSeek-V3.
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    Args:
        vocab_size（`int`，*可选*，默认为 129280）：
        Deep 模型的词汇量。定义调用 [`DeepseekV3Model`] 时传递的 `inputs_ids` 可以表示的不同标记的数量
        hidden_size（`int`，*可选*，默认为 4096）：
        隐藏表示的维度。
        middle_size（`int`，*可选*，默认为 11008）：
        MLP 表示的维度。
        moe_intermediate_size（`int`，*可选*，默认为 1407）：
        MoE 表示的维度。
        num_hidden_layers（`int`，*可选*，默认为 32）：
        Transformer 解码器中的隐藏层数量。
        num_nextn_predict_layers（`int`，*可选*，默认为 1）：
        DeepSeekV3 模型中的 nextn 预测层的数量。
        num_attention_heads（`int`，*可选*，默认为 32）：
        Transformer 解码器中每个注意层的注意头数量。
        n_shared_experts（`int`，*可选*，默认为 None）：
        共享专家的数量，None 表示密集模型。
        n_routed_experts（`int`，*可选*，默认为 None）：
        路由专家的数量，None 表示密集模型。
        routed_scaling_factor（`float`，*可选*，默认为 1.0）：
        缩放因子或路由专家。
        topk_method（`str`，*可选*，默认为 `gready`）：
        路由门中使用的 Topk 方法。
        n_group (`int`, *可选*, 默认为 None):
        路由专家的组数。
        topk_group (`int`, *可选*, 默认为 None):
        每个 token 的选定组数（对于每个 token，确保选定的专家仅在 `topk_group` 组内）。
        num_experts_per_tok (`int`, *可选*, 默认为 None):
        选定专家的数量，None 表示密集模型。
        moe_layer_freq (`int`, *可选*, 默认为 1):
        MoE 层的频率：每 `moe_layer_freq - 1` 个密集层有一个专家层。
        first_k_dense_replace (`int`, *可选*, 默认为 0):
        浅层中的密集层数（embed->dense->dense->...->dense->moe->moe...->lm_head）。
        \--k 密集层--/
        norm_topk_prob（`bool`，*可选*，默认为 False）：
        是否对路由专家的权重进行归一化。
        scoring_func（`str`，*可选*，默认为“softmax”）：
        计算专家权重的方法。
        aux_loss_alpha（`float`，*可选*，默认为 0.001）：
        辅助损失权重系数。
        seq_aux =（`bool`，*可选*，默认为 True）：
        是否计算每个单独样本的辅助损失。
        num_key_value_heads（`int`，*可选*）：
        这是应用于实现分组查询注意的 key_value 头的数量。如果
        `num_key_value_heads=num_attention_heads`，则模型将使用多头注意力 (MHA)，如果
        `num_key_value_heads=1，则模型将使用多查询注意力 (MQA)，否则将使用 GQA。将多头检查点转换为 GQA 检查点时，应通过对该组内的所有原始头进行均值池化来构建每个组键和值头。有关更多详细信息，请查看 [这篇
        论文](https://arxiv.org/pdf/2305.13245.pdf)。如果未指定，则默认为
        `num_attention_heads`。
        hidden_​​act（`str` 或 `function`，*可选*，默认为 `"silu"`）：
        解码器中的非线性激活函数（函数或字符串）。
        max_position_embeddings（`int`，*可选*，默认为 2048）：
        此模型可能使用的最大序列长度。
        initializer_range（`float`，*可选*，默认为 0.02）：
        用于初始化所有权重矩阵的 truncated_normal_initializer 的标准偏差。
        rms_norm_eps（`float`，*可选*，默认为 1e-06）：
        rms 正则化层使用的 epsilon。
        use_cache（`bool`，*可选*，默认为 `True`）：
        模型是否应返回最后的键/值注意（并非所有模型都使用）。仅当 `config.is_decoder=True` 时才相关。
        pad_token_id（`int`，*可选*）：
        填充标记 ID。
        bos_token_id（`int`，*可选*，默认为 1）：
        流标记 ID 的开头。
        eos_token_id（`int`，*可选*，默认为 2）：
        流标记 ID 的结尾。
        pretraining_tp (`int`，*可选*，默认为 1)：
        实验性功能。预训练期间使用的张量并行度。请参阅[此
        文档](https://huggingface.co/docs/transformers/parallelism) 了解更多信息。此值是确保预训练结果准确可重复性的必要值。请参阅[此
        问题](https://github.com/pytorch/pytorch/issues/76232)。
        tie_word_embeddings (`bool`，*可选*，默认为`False`)：
        是否绑定权重嵌入
        rope_theta (`float`，*可选*，默认为 10000.0)：
        RoPE 嵌入的基准周期。
        rope_scaling (`Dict`，*可选*)：
        包含 RoPE 嵌入的缩放配置的字典。当前支持两种缩放
        策略：线性和动态。它们的缩放因子必须是大于 1 的浮点数。预期格式为
        `{"type": 策略名称，"factor": 缩放因子}`。使用此标志时，请勿将
        `max_position_embeddings` 更新为预期的新最大值。
        attention_bias（`bool`，默认为 `False`，*可选*，默认为 `False`）：
        在自注意力期间是否在查询、键、值和输出投影层中使用偏差。
        attention_dropout（`float`，*可选*，默认为 0.0）：
        注意概率的 dropout 比率。
    ```python
    >>> from transformers import DeepseekV3Model, DeepseekV3Config
    >>> # Initializing a Deepseek-V3 style configuration
    >>> configuration = DeepseekV3Config()
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "deepseek_v3"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=64793,
        hidden_size=1024,
        intermediate_size=2048,
        moe_intermediate_size = 512,
        num_hidden_layers=28,
        num_nextn_predict_layers=1,
        num_attention_heads=18,
        num_key_value_heads=6,
        n_shared_experts = 1,
        n_routed_experts = 32,
        ep_size = 1,
        routed_scaling_factor = 2.5,
        kv_lora_rank = 512,
        q_lora_rank = 1536,
        qk_rope_head_dim = 64,
        v_head_dim = 128,
        qk_nope_head_dim = 128,
        topk_method = 'noaux_tc',
        n_group = 2,
        topk_group = 4,
        num_experts_per_tok = 8,
        moe_layer_freq = 1,
        first_k_dense_replace = 3,
        norm_topk_prob = True,
        scoring_func = 'sigmoid',
        aux_loss_alpha = 0.001,
        seq_aux = True,
        hidden_act="silu",
        max_position_embeddings=512,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=0,
        eos_token_id=1,
        pretraining_tp=1,
        tie_word_embeddings=True,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.num_attention_heads = num_attention_heads
        self.n_shared_experts = n_shared_experts
        self.n_routed_experts = n_routed_experts
        self.ep_size = ep_size
        self.routed_scaling_factor = routed_scaling_factor
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.topk_method = topk_method
        self.n_group = n_group
        self.topk_group = topk_group
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_layer_freq = moe_layer_freq
        self.first_k_dense_replace = first_k_dense_replace
        self.norm_topk_prob = norm_topk_prob
        self.scoring_func = scoring_func
        self.aux_loss_alpha = aux_loss_alpha
        self.seq_aux = seq_aux
        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )