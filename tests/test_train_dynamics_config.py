from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from omegaconf.errors import MissingMandatoryValue

from dreamer.actions import Actions
from scripts.train_dynamics import (
    validate_action_batch,
    validate_dynamics_config,
)


CONFIG_DIR = str((Path(__file__).parents[1] / "configs").resolve())


def compose_config(config_name: str, overrides: list[str] | None = None):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        return compose(config_name=config_name, overrides=overrides or [])


@pytest.mark.parametrize("config_name", ["coinrun_dynamics", "dynamics"])
def test_action_dimensions_accept_supported_resolved_configs(config_name):
    validate_dynamics_config(compose_config(config_name))


def test_action_dimensions_reject_dataset_model_mismatch():
    cfg = compose_config(
        "coinrun_dynamics",
        overrides=["dynamics.categorical_action_dim=15"],
    )

    with pytest.raises(ValueError, match="categorical_action_dim.*dataset=16, dynamics=15"):
        validate_dynamics_config(cfg)


def test_config_validation_rejects_missing_resolved_value():
    cfg = compose_config(
        "coinrun_dynamics",
        overrides=["tokenizer_ckpt=???"],
    )

    with pytest.raises(MissingMandatoryValue):
        validate_dynamics_config(cfg)


def test_coinrun_categorical_action_batch_shape_is_accepted():
    cfg = compose_config("coinrun_dynamics")
    actions = Actions(
        binary=None,
        categorical=np.zeros((2, 16), dtype=np.int32),
        continuous=None,
    )

    validate_action_batch(actions, cfg, (2, 16))


def test_minecraft_action_batch_shape_is_accepted():
    cfg = compose_config("dynamics")
    actions = Actions(
        binary=np.zeros((2, 16, 27), dtype=np.int32),
        categorical=np.zeros((2, 16), dtype=np.int32),
        continuous=None,
    )

    validate_action_batch(actions, cfg, (2, 16))


@pytest.mark.parametrize(
    ("actions", "message"),
    [
        (
            Actions(binary=None, categorical=None, continuous=None),
            "missing categorical actions",
        ),
        (
            Actions(
                binary=None,
                categorical=np.zeros((2, 16, 1), dtype=np.int32),
                continuous=None,
            ),
            "Expected categorical actions with shape",
        ),
        (
            Actions(
                binary=np.zeros((2, 16, 1), dtype=np.int32),
                categorical=np.zeros((2, 16), dtype=np.int32),
                continuous=None,
            ),
            "configured dimension is 0",
        ),
    ],
)
def test_coinrun_action_batch_rejects_missing_extra_or_misshaped_modalities(
    actions,
    message,
):
    cfg = compose_config("coinrun_dynamics")

    with pytest.raises(ValueError, match=message):
        validate_action_batch(actions, cfg, (2, 16))
