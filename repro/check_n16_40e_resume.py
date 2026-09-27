#!/usr/bin/env python3
"""Offline invariants for exact N16 epoch-20 continuation."""
import copy
import json

import torch
from torch.utils.data import DataLoader


def consume_epochs(generator, epochs, workers):
    loader = DataLoader(
        range(41), batch_size=4, shuffle=True,
        num_workers=workers, generator=generator,
    )
    for _ in range(epochs):
        for _ in loader:
            pass


def check_data_stream_replay():
    continuous = torch.Generator().manual_seed(1)
    replay = torch.Generator().manual_seed(1)
    consume_epochs(continuous, 20, workers=2)
    consume_epochs(replay, 20, workers=0)
    assert torch.equal(continuous.get_state(), replay.get_state())

    next_continuous = list(DataLoader(
        range(41), batch_size=4, shuffle=True,
        num_workers=0, generator=continuous,
    ))
    next_replay = list(DataLoader(
        range(41), batch_size=4, shuffle=True,
        num_workers=0, generator=replay,
    ))
    assert len(next_continuous) == len(next_replay)
    assert all(
        torch.equal(left, right)
        for left, right in zip(next_continuous, next_replay)
    )


def make_training_state(seed):
    torch.manual_seed(seed)
    model = torch.nn.Linear(5, 3)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=150, eta_min=1e-6
    )
    return model, optimizer, scheduler


def step(model, optimizer, scheduler, inputs, target):
    optimizer.zero_grad(set_to_none=True)
    loss = (model(inputs) - target).square().mean()
    loss.backward()
    optimizer.step()
    scheduler.step()
    return loss.detach()


def check_optimizer_scheduler_resume():
    torch.manual_seed(9)
    batches = [
        (torch.randn(4, 5), torch.randn(4, 3)) for _ in range(21)
    ]
    model, optimizer, scheduler = make_training_state(5)
    for inputs, target in batches[:20]:
        step(model, optimizer, scheduler, inputs, target)
    assert scheduler.last_epoch == 20
    model_state = copy.deepcopy(model.state_dict())
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    scheduler_state = copy.deepcopy(scheduler.state_dict())
    expected_lr = optimizer.param_groups[0]['lr']

    resumed_model, resumed_optimizer, resumed_scheduler = make_training_state(77)
    resumed_model.load_state_dict(model_state, strict=True)
    resumed_optimizer.load_state_dict(optimizer_state)
    resumed_scheduler.load_state_dict(scheduler_state)
    assert resumed_scheduler.last_epoch == 20
    assert resumed_scheduler.state_dict()['T_max'] == 150
    assert resumed_optimizer.param_groups[0]['lr'] == expected_lr

    inputs, target = batches[20]
    reference_loss = step(model, optimizer, scheduler, inputs, target)
    resumed_loss = step(
        resumed_model, resumed_optimizer, resumed_scheduler, inputs, target
    )
    torch.testing.assert_close(resumed_loss, reference_loss, rtol=0, atol=0)
    for expected, actual in zip(
        model.parameters(), resumed_model.parameters()
    ):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert scheduler.last_epoch == resumed_scheduler.last_epoch == 21
    assert (
        optimizer.param_groups[0]['lr']
        == resumed_optimizer.param_groups[0]['lr']
    )


def main():
    check_data_stream_replay()
    check_optimizer_scheduler_resume()
    print(json.dumps({
        'n16_40e_resume_check': 'passed',
        'data_generator_replay_20_epochs': 'exact',
        'next_epoch_sampler_order': 'exact',
        'adam_state_resume': 'exact',
        'cosine_tmax': 150,
        'scheduler_epoch_20_to_21': 'exact',
    }, indent=2))


if __name__ == '__main__':
    main()
