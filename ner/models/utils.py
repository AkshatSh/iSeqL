import torch

# def argmax(vec):
#     # TODO(AkshatSh): can this just be torch.argmax(vec) ? 
#     _, idx = torch.max(vec, 1)
#     print('here', idx.shape)
#     return idx.item()

# Compute log sum exp in a numerically stable way for the forward algorithm
# def log_sum_exp(vec):
#     # TODO(AkshatSh): is this purposeful here? what exactly does this compute
#     max_score = vec[0, argmax(vec)]
#     max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
#     return max_score + \
#         torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

def log_sum_exp(x):
    m = torch.max(x, -1)[0]
    return m + torch.log(torch.sum(torch.exp(x - m.unsqueeze(-1)), -1))