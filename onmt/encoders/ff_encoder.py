import torch
from onmt.encoders.encoder import EncoderBase
from onmt.modules import MultiHeadedAttention


class FeedForwardEncoderLayers(torch.nn.Module):
    def __init__(self, input_size, units, layers_num, dropout=0.1):
        super(FeedForwardEncoderLayers, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.dropout = torch.nn.ModuleList()
        self.layers_num = layers_num
        self.activate = torch.nn.ReLU()
        input_size = input_size
        for i in range(layers_num):
            self.layers.append(torch.nn.Linear(input_size, units))
            input_size = units
            self.dropout.append(torch.nn.Dropout(dropout))
        input_size = units

    def forward(self, x):
        for i in range(self.layers_num):
            x = self.layers[i](x)
            if i != self.layers_num - 1:
                x = self.activate(x)
            x = self.dropout[i](x)
        return x

    def update_dropout(self, dropout):
        for i in range(self.layers_num):
            self.dropout[i].p = dropout

def pad_mask(src, pad_idx):
    return torch.eq(src, pad_idx)

class FeedForwardEncoder(EncoderBase):
    """
    Args:
        num_layers (int): number of encoder layers
        d_model (int): size of the model
        heads (int): number of heads
        d_ff (int): size of the inner FF layer
        dropout (float): dropout parameters
        embeddings (onmt.modules.Embeddings):
          embeddings to use, should have positional encodings

    Returns:
        (torch.FloatTensor, torch.FloatTensor):

        * embeddings ``(src_len, batch_size, model_dim)``
        * memory_bank ``(src_len, batch_size, model_dim)``
    """

    def __init__(self, num_layers, size, heads, embeddings, dropout):
        super(FeedForwardEncoder, self).__init__()
        self.embeddings = embeddings
        self.size = size
        self.encode_layers = FeedForwardEncoderLayers(embeddings.embedding_size, size, num_layers, dropout)
        # self.attention = MultiHeadedAttention(heads, embeddings.embedding_size)

    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        return cls(
            opt.enc_layers,
            opt.enc_rnn_size,
            opt.heads,
            embeddings,
            opt.dropout[0] if type(opt.dropout) is list else opt.dropout)

    def forward(self, src, lengths=None):
        """See :func:`EncoderBase.forward()`"""
        self._check_args(src, lengths)

        emb = self.embeddings(src)

        # mask = pad_mask(src, self.embeddings.word_padding_idx)
        # print("mask shape", mask.shape)

        # context, _ = self.attention(emb, emb, emb, attn_type="self")

        out = self.encode_layers(emb)

        final_state = out.mean(dim = 0).unsqueeze(0)

        return (final_state, final_state), out, lengths

    def update_dropout(self, dropout):
        self.embeddings.update_dropout(dropout)
        self.encode_layers.update_dropout(dropout)





























