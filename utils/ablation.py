import torch


def get_active_streams(cfg):
    if 'active_streams' not in cfg.Data:
        return [1, 1, 1]
    active_streams = list(cfg.Data.active_streams)
    if len(active_streams) != 3:
        raise ValueError(f'Data.active_streams must have length 3, got {active_streams}')
    return active_streams


def compute_bag_loss(loss_fn, logits1, logits2, logits3, merge_logits, y, cfg, active_streams):
    branch_losses = []
    for logits, active in zip([logits1, logits2, logits3], active_streams):
        if active:
            branch_losses.append(loss_fn(logits, y))

    if not branch_losses:
        raise ValueError('At least one stream must be active for bag loss computation.')

    branch_loss = sum(branch_losses) / len(branch_losses)
    merge_loss = loss_fn(merge_logits, y)
    return cfg.Model.loss1 * branch_loss + cfg.Model.mergeloss * merge_loss


def compute_pred_prob(cfg, logits1, logits2, logits3, merge_logits, active_streams):
    active_count = sum(active_streams)
    if active_count == 1:
        active_idx = active_streams.index(1)
        logits = [logits1, logits2, logits3][active_idx]
        return torch.softmax(logits, dim=-1)

    y_prob1 = torch.softmax(logits1, dim=-1)
    y_prob2 = torch.softmax(logits2, dim=-1)
    y_prob3 = torch.softmax(logits3, dim=-1)
    y_prob_merge = torch.softmax(merge_logits, dim=-1)
    return cfg.Model.loss1 * (y_prob1 + y_prob2 + y_prob3) / 3 + cfg.Model.mergeloss * y_prob_merge
