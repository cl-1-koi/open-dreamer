from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.errors import ConfigCompositionException
from omegaconf import OmegaConf

# Importing the entry point registers the arithmetic resolvers used by configs.
from scripts import train_dynamics  # noqa: F401


CONFIG_DIR = str((Path(__file__).parents[1] / "configs").resolve())


def compose_config(config_name: str, overrides: list[str] | None = None):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        return compose(config_name=config_name, overrides=overrides or [])


def test_coinrun_tokenizer_smoke_profile_composes():
    cfg = compose_config("coinrun_tokenizer")
    resolved = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    assert OmegaConf.is_struct(cfg)
    assert resolved["run_name"] == "coinrun-tokenizer-smoke"
    assert resolved["max_steps"] == 10
    assert resolved["dataset"]["name"] == "coinrun"
    assert resolved["dataset"]["data_type"] == "video"
    assert resolved["dataset"]["dataloader_cfg"]["B"] == 2
    assert resolved["dataset"]["dataloader_cfg"]["long_T"] == 16
    assert resolved["tokenizer"]["encoder"]["n_latents"] == 32
    assert resolved["tokenizer"]["encoder"]["d_bottleneck"] == 8
    assert resolved["tokenizer"]["encoder"]["depth"] == 2
    assert resolved["tokenizer"]["encoder"]["d_model"] == 128
    assert resolved["tokenizer"]["decoder"]["depth"] == 2
    assert resolved["tokenizer"]["decoder"]["d_model"] == 128
    assert resolved["lpips_weight"] == 0.0


def test_coinrun_dynamics_smoke_profile_composes():
    cfg = compose_config("coinrun_dynamics")
    resolved = train_dynamics.validate_dynamics_config(cfg)

    assert OmegaConf.is_struct(cfg)
    assert resolved["run_name"] == "coinrun-dynamics-smoke"
    assert resolved["tokenizer_ckpt"] == "logs/coinrun-tokenizer-smoke/checkpoints"
    assert resolved["max_steps"] == 10
    assert resolved["dataset"]["name"] == "coinrun"
    assert resolved["dataset"]["data_type"] == "video"
    assert resolved["dataset"]["dataloader_cfg"]["B"] == 2
    assert resolved["dataset"]["dataloader_cfg"]["long_T"] == 16
    assert resolved["dynamics"]["depth"] == 2
    assert resolved["dynamics"]["d_model"] == 128
    assert resolved["dynamics"]["d_bottleneck"] == 8
    assert resolved["dynamics"]["context_length"] == 16
    assert resolved["dynamics"]["n_register"] == 4
    assert resolved["dynamics"]["categorical_action_dim"] == 16
    assert resolved["dynamics"]["latent_mean"] is None
    assert resolved["dynamics"]["latent_std"] is None
    assert resolved["bootstrap_start"] == 100
    assert resolved["ot"]["enabled"] is False


def test_coinrun_profiles_do_not_change_minecraft_defaults():
    tokenizer_cfg = compose_config("tokenizer")
    dynamics_cfg = compose_config("dynamics")

    assert tokenizer_cfg.dataset.name == "minecraft_vpt"
    assert dynamics_cfg.dataset.name == "minecraft_vpt_latent"
    assert dynamics_cfg.dataset.num_binary_actions == 27
    assert dynamics_cfg.dataset.categorical_action_dim == 121
    train_dynamics.validate_dynamics_config(dynamics_cfg)


def test_coinrun_config_rejects_unknown_override_without_explicit_add():
    with pytest.raises(ConfigCompositionException):
        compose_config(
            "coinrun_dynamics",
            overrides=["dynamics.unrecognized_action_dim=16"],
        )
