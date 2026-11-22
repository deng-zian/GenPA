from genpa.robotwin.multiview.model import MultiViewPolicy
from genpa.robotwin.depth.model import create_policy as create_base
from .contract import MODE, checkpoint_contract


def create_policy(*, stats_path, mode=MODE, **kwargs):
    if mode != MODE:
        raise ValueError(mode)
    contract = checkpoint_contract(stats_path)
    policy = create_base(stats_path=None, mode="absolute_depth", _policy_class=MultiViewPolicy, **kwargs)
    policy.action_contract = contract
    return policy
