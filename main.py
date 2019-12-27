import tensorflow as tf
from relative_transformer_module import relative_multi_head_attention, feedforward


def adapting_transformer_layer(self, batch_input, num_heads=6, attn_blocks_num=3):
    """
    (Temporary abandoning)
    This is a adapting transformer structure
    For NER, it only needs transformer encoder
    Original paper: https://arxiv.org/abs/1706.03762
    NER version (TENER) paper: https://arxiv.org/abs/1911.04474

    input shape: [batch_size, seq_length, num_heads*dim_heads]
    output shape: [batch_size, seq_length, num_heads*dim_heads]

    :param num_heads: The number of parallel headers in multi head attention
    :param attn_blocks_num: The number of multi head attentions and FNNs in serial
    """
    attention_size = batch_input.get_shape().as_list()[-1]
    attn_outs = batch_input

    for block_id in range(attn_blocks_num):
        with tf.variable_scope("num_blocks_{}".format(block_id)):
            attn_outs = relative_multi_head_attention(
                attn_outs, num_heads, self.nc.dropout_keep_rate, reuse=False)
            attn_outs = feedforward(attn_outs, [int(attention_size * 1.5), attention_size],
                                    self.nc.dropout_keep_rate, reuse=False)

    return attn_outs, attention_size

