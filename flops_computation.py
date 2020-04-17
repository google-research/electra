"""Computes the flops needed for training/running transformer networks."""

import collections

# We checked this code with TensorFlow"s FLOPs counting, although we had to
# correct for this issue: https://github.com/tensorflow/tensorflow/issues/22071
# Assumptions going into the FLOPs counting
#   - An "operation" is a mathematical operation, not a machine instruction. So
#     an "exp" takes one opp like and add, even though in practice an exp
#     might be slower. This is not too bad an assumption because
#     matrix-multiplies dominate the compute for most models, so minor details
#     about activation functions don"t matter too much. Similarly, we count
#     matrix-multiplies as 2*m*n flops instead of m*n, as one might if
#     if considering fused multiply-add ops.
#   - Backward pass takes the same number of FLOPs as forward pass. No exactly
#     right (e.g., for softmax cross entropy loss the backward pass is faster).
#     Importantly, it really is the same for matrix-multiplies, which is most of
#     the compute anyway.
#   - We assume "dense" embedding lookups (i.e., multiplication by a one-hot
#     vector). On some hardware accelerators, these dense operations are
#     actually faster than sparse lookups.
# Please open a github issue if you spot a problem with this code!

# I am not sure if the below constants are 100% right, but they are only applied
# to O(hidden_size) activations, which is generally a lot less compute than the
# matrix-multiplies, which are O(hidden_size^2), so they don't affect the total
# number of FLOPs much.

# random number, >=, multiply activations by dropout mask, multiply activations
# by correction (1 / (1 - dropout_rate))
DROPOUT_FLOPS = 4

# compute mean activation (sum), computate variance of activation
# (square and sum), bias (add), scale (multiply)
LAYER_NORM_FLOPS = 5

# GELU: 0.5 * x * (1 + tanh(sqrt(2 / np.pi) * (x + 0.044715 * pow(x, 3))))
ACTIVATION_FLOPS = 8

# max/substract (for stability), exp, sum, divide
SOFTMAX_FLOPS = 5


class TransformerHparams(object):
  """Computes the train/inference FLOPs for transformers."""

  def __init__(self, h, l, s=512, v=30522, e=None, i=None, heads=None,
      head_size=None, output_frac=0.15625, sparse_embed_lookup=False,
      decoder=False):
    self.h = h  # hidden size
    self.l = l  # number of layers
    self.s = s  # sequence length
    self.v = v  # vocab size
    self.e = h if e is None else e  # embedding size
    self.i = h * 4 if i is None else i  # intermediate size
    self.kqv = h if head_size is None else head_size * heads  # attn proj sizes
    self.heads = max(h // 64, 1) if heads is None else heads  # attention heads
    self.output_frac = output_frac  # percent of tokens using an output softmax
    self.sparse_embed_lookup = sparse_embed_lookup  # sparse embedding lookups
    self.decoder = decoder  # decoder has extra attn to encoder states

  def get_block_flops(self):
    """Get the forward-pass FLOPs for a single transformer block."""
    attn_mul = 2 if self.decoder else 1
    block_flops = dict(
        kqv=3 * 2 * self.h * self.kqv * attn_mul,
        kqv_bias=3 * self.kqv * attn_mul,
        attention_scores=2 * self.kqv * self.s * attn_mul,
        attn_softmax=SOFTMAX_FLOPS * self.s * self.heads * attn_mul,
        attention_dropout=DROPOUT_FLOPS * self.s * self.heads * attn_mul,
        attention_scale=self.s * self.heads * attn_mul,
        attention_weighted_avg_values=2 * self.h * self.s * attn_mul,
        attn_output=2 * self.h * self.h * attn_mul,
        attn_output_bias=self.h * attn_mul,
        attn_output_dropout=DROPOUT_FLOPS * self.h * attn_mul,
        attn_output_residual=self.h * attn_mul,
        attn_output_layer_norm=LAYER_NORM_FLOPS * attn_mul,
        intermediate=2 * self.h * self.i,
        intermediate_act=ACTIVATION_FLOPS * self.i,
        intermediate_bias=self.i,
        output=2 * self.h * self.i,
        output_bias=self.h,
        output_dropout=DROPOUT_FLOPS * self.h,
        output_residual=self.h,
        output_layer_norm=LAYER_NORM_FLOPS * self.h,
    )
    return sum(block_flops.values()) * self.s

  def get_embedding_flops(self, output=False):
    """Get the forward-pass FLOPs the transformer inputs or output softmax."""
    embedding_flops = {}
    if output or (not self.sparse_embed_lookup):
      embedding_flops["main_multiply"] = 2 * self.e * self.v
    # input embedding post-processing
    if not output:
      embedding_flops.update(dict(
          tok_type_and_position=2 * self.e * (self.s + 2),
          add_tok_type_and_position=2 * self.e,
          emb_layer_norm=LAYER_NORM_FLOPS * self.e,
          emb_dropout=DROPOUT_FLOPS * self.e
      ))
    # projection layer if e != h
    if self.e != self.h or output:
      embedding_flops.update(dict(
          hidden_kernel=2 * self.h * self.e,
          hidden_bias=self.e if output else self.h
      ))
      # extra hidden layer and output softmax
      if output:
        embedding_flops.update(dict(
            hidden_activation=ACTIVATION_FLOPS * self.e,
            hidden_layernorm=LAYER_NORM_FLOPS * self.e,
            output_softmax=SOFTMAX_FLOPS * self.v,
            output_target_word=2 * self.v
        ))
        return self.output_frac * sum(embedding_flops.values()) * self.s
    return sum(embedding_flops.values()) * self.s

  def get_binary_classification_flops(self):
    classification_flops = dict(
        hidden=2 * self.h * self.h,
        hidden_bias=self.h,
        hidden_act=ACTIVATION_FLOPS * self.h,
        logits=2 * self.h
    )
    return sum(classification_flops.values()) * self.s

  def get_train_flops(self, batch_size, train_steps, discriminator=False):
    """Get the FLOPs for pre-training the transformer."""
    # 2* for forward/backward pass
    return 2 * batch_size * train_steps * (
        (self.l * self.get_block_flops()) +
        self.get_embedding_flops(output=False) +
        (self.get_binary_classification_flops() if discriminator else
         self.get_embedding_flops(output=True))
    )

  def get_infer_flops(self):
    """Get the FLOPs for running inference with the transformer on a
    classification task."""
    return ((self.l * self.get_block_flops()) +
            self.get_embedding_flops(output=False) +
            self.get_binary_classification_flops())


def get_electra_train_flops(
    h_d, l_d, h_g, l_g, batch_size, train_steps, tied_embeddings,
    e=None, s=512, output_frac=0.15625):
  """Get the FLOPs needed for  pre-training ELECTRA."""
  if e is None:
    e = h_d
  disc = TransformerHparams(
      h_d, l_d, s=s, e=e,
      output_frac=output_frac).get_train_flops(batch_size, train_steps, True)
  gen = TransformerHparams(
      h_g, l_g, s=s, e=e if tied_embeddings else None,
      output_frac=output_frac).get_train_flops(batch_size, train_steps)
  return disc + gen


MODEL_FLOPS = collections.OrderedDict([
    # These runtimes were computed with tensorflow FLOPs counting instead of the
    # script, as the neural architectures are quite different.
    # 768648884 words in LM1b benchmark, 10 epochs with batch size 20,
    # seq length 128, 568093262680 FLOPs per example.
    ("elmo", 2 * 10 * 768648884 * 568093262680 / (20.0 * 128)),
    # 15064773691518 is FLOPs for forward pass on 32 examples.
    # Therefore 2 * steps * batch_size * 15064773691518 / 32 is XLNet compute
    ("xlnet", 2 * 500000 * 8192 * 15064773691518 / 32.0),

    # Runtimes computed with the script
    ("gpt", TransformerHparams(768, 12, v=40000, output_frac=1.0).get_train_flops(
        128, 960800)),
    ("bert_small", TransformerHparams(256, 12, e=128, s=128).get_train_flops(128, 1.45e6)),
    ("bert_base", TransformerHparams(768, 12).get_train_flops(256, 1e6)),
    ("bert_large", TransformerHparams(1024, 24).get_train_flops(256, 1e6)),
    ("electra_small", get_electra_train_flops(256, 12, 64, 12, 128, 1e6, True, s=128, e=128)),
    ("electra_base", get_electra_train_flops(768, 12, 256, 12, 256, 766000, True)),
    ("electra_400k", get_electra_train_flops(1024, 24, 256, 24, 2048, 400000, True)),
    ("electra_1.75M", get_electra_train_flops(1024, 24, 256, 24, 2048, 1750000, True)),

    # RoBERTa, ALBERT, and T5 have  minor architectural differences from
    # BERT/ELECTRA, but I believe they don't significantly effect the runtime,
    # so we use this script for those models as well.
    ("roberta", TransformerHparams(1024, 24, v=50265).get_train_flops(8000, 500000)),
    ("albert", TransformerHparams(4096, 12, v=30000, e=128).get_train_flops(
        4096, 1.5e6)),
    ("t5_11b", TransformerHparams(
        1024,  # hidden size
        24,  # layers
        v=32000,  # vocab size
        i=65536,  # ff intermediate hidden size
        heads=128, head_size=128,  # heads/head size
        output_frac=0.0  # encoder has no output softmax
    ).get_train_flops(2048, 1e6) +  # 1M steps with batch size 2048
     TransformerHparams(
         1024,
         24,
         v=32000,
         i=65536,
         heads=128, head_size=128,
         output_frac=1.0,  # decoder has output softmax for all positions
         decoder=True
     ).get_train_flops(2048, 1e6))
])


def main():
  for k, v in MODEL_FLOPS.items():
    print(k, v)


if __name__ == "__main__":
  main()

