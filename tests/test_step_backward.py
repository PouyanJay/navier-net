"""A step's backward pass reaches the parameters the optimiser steps and
nothing else: the sample points are leaves of the graph too, and gradients
carried on to them are never read."""

from __future__ import annotations

import torch

from naviernet import training


def _problem():
    torch.manual_seed(0)
    net = torch.nn.Linear(2, 1)
    frozen = torch.nn.Linear(2, 1).requires_grad_(False)
    opt = torch.optim.Adam([*net.parameters(), *frozen.parameters()], lr=1e-3)
    points = torch.rand(8, 2, requires_grad=True)
    return net, frozen, opt, points


def _loss(net, frozen, points) -> torch.Tensor:
    value = net(points) + frozen(points)
    slope = torch.autograd.grad(value.sum(), points, create_graph=True)[0]
    return (value**2).mean() + (slope**2).mean()


def test_the_stepped_parameters_are_the_optimisers_trainable_ones():
    net, _, opt, _ = _problem()

    stepped = training._stepped(opt)

    assert [id(p) for p in stepped] == [id(p) for p in net.parameters()]


def test_the_parameters_get_the_gradients_a_whole_backward_pass_gives():
    net, frozen, opt, points = _problem()
    _loss(net, frozen, points).backward()
    whole = [p.grad.clone() for p in net.parameters()]
    opt.zero_grad()

    _loss(net, frozen, points).backward(inputs=training._stepped(opt))

    for p, expected in zip(net.parameters(), whole, strict=True):
        assert torch.equal(p.grad, expected)


def test_no_gradient_is_carried_on_to_the_sample_points():
    net, frozen, opt, points = _problem()

    _loss(net, frozen, points).backward(inputs=training._stepped(opt))

    assert points.grad is None
