import torch
import torch.nn.functional as F

from mist.glacier.decoder import _MembershipDecoderLayer


def test_membership_attention_matches_explicit_tagged_memory():
    torch.manual_seed(51)
    layer = _MembershipDecoderLayer(8, 2, 16, 0., self_attn_mode="diagonal").eval()
    target = torch.randn(2, 3, 8)
    memory = torch.randn(2, 4, 8)
    groups = torch.randn(5, 8)
    membership = torch.randint(0, 5, (2, 3, 4))
    atom_pad = torch.tensor([[False, False, True, True], [False, False, False, False]])
    fragment_pad = torch.zeros(2, 3, dtype=torch.bool)
    actual = layer(target, memory, groups, membership, atom_pad, fragment_pad)
    x = layer.norm1(target + layer.self_transform(target))
    tagged = memory[:, None] + groups[membership]
    q = layer.q_proj(x).reshape(2, 3, 2, 4).permute(0, 2, 1, 3)
    k = layer.k_proj(tagged).reshape(2, 3, 4, 2, 4).permute(0, 3, 1, 2, 4)
    v = layer.v_proj(tagged).reshape(2, 3, 4, 2, 4).permute(0, 3, 1, 2, 4)
    scores = torch.einsum('bhfd,bhfnd->bhfn', q, k) * layer.scale
    scores = scores.masked_fill(atom_pad[:, None, None, :], float('-inf'))
    context = torch.einsum('bhfn,bhfnd->bhfd', scores.softmax(-1), v)
    context = context.permute(0, 2, 1, 3).reshape(2, 3, 8)
    x = layer.norm2(x + layer.out_proj(context))
    expected = layer.norm3(x + layer.linear2(F.relu(layer.linear1(x))))
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
