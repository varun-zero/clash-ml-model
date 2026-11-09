import numpy as np
import pytest
import torch

from cr_alphastar_agent.env.crsim_replica_env import CrSimReplicaEnv
from cr_alphastar_agent.model.alphastar_cr_net import AlphaStarCRNet


@pytest.fixture
def simulator():
    return CrSimReplicaEnv(seed=42)


@pytest.fixture
def initial_observation(simulator):
    observation, _ = simulator.reset()
    return observation


def test_observation_shapes_and_dtypes(initial_observation):
    expected_shapes = {
        "scalars": (4,),
        "hand_cards": (5,),
        "entities": (40, 7),
        "entity_mask": (40,),
        "arena_grid": (6, 32, 18),
        "card_mask": (5,),
        "tile_mask": (576,),
        "card_tile_mask": (5, 576),
    }

    assert set(initial_observation) == set(expected_shapes)
    for name, shape in expected_shapes.items():
        assert initial_observation[name].shape == shape, name
        assert isinstance(initial_observation[name], np.ndarray), name


def test_crown_towers_are_present_in_reserved_entity_slots(initial_observation):
    assert not initial_observation["entity_mask"][0]
    assert not initial_observation["entity_mask"][5]
    assert np.all(initial_observation["entities"][:6, 6] == 1.0)


def test_policy_network_returns_finite_action_values(initial_observation):
    model = AlphaStarCRNet()
    model.eval()
    batch = {
        key: torch.from_numpy(value).unsqueeze(0)
        for key, value in initial_observation.items()
    }

    with torch.no_grad():
        actions, logprobs, (card_entropy, tile_entropy), values = (
            model.get_action_and_value(batch, return_entropy_breakdown=True)
        )

    assert actions.shape == (1, 2)
    assert logprobs.shape == (1,)
    assert values.shape == (1,)
    assert torch.isfinite(actions).all()
    assert torch.isfinite(logprobs).all()
    assert torch.isfinite(card_entropy).all()
    assert torch.isfinite(tile_entropy).all()
    assert torch.isfinite(values).all()


def test_step_both_advances_a_valid_wait_action(simulator):
    simulator.reset()
    wait = np.array([4, 0], dtype=np.int64)

    obs_blue, obs_red, reward_blue, reward_red, terminated, truncated, info = (
        simulator.step_both(wait, wait)
    )

    assert obs_blue["scalars"].shape == (4,)
    assert obs_red["scalars"].shape == (4,)
    assert np.isfinite([reward_blue, reward_red]).all()
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
