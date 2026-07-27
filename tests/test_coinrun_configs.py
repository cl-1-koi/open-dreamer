from pathlib import Path
import unittest

from hydra import compose, initialize_config_dir
from hydra.errors import ConfigCompositionException
from omegaconf import OmegaConf

# Importing the entry point registers the arithmetic resolvers used by configs.
from scripts import train_dynamics  # noqa: F401


CONFIG_DIR = str((Path(__file__).parents[1] / "configs").resolve())


def compose_config(config_name: str, overrides: list[str] | None = None):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        return compose(config_name=config_name, overrides=overrides or [])


class CoinRunConfigTest(unittest.TestCase):
    def test_coinrun_tokenizer_smoke_profile_composes(self):
        cfg = compose_config("coinrun_tokenizer")
        resolved = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

        self.assertTrue(OmegaConf.is_struct(cfg))
        self.assertEqual(resolved["run_name"], "coinrun-tokenizer-smoke")
        self.assertEqual(resolved["max_steps"], 10)
        self.assertEqual(resolved["dataset"]["name"], "coinrun")
        self.assertEqual(resolved["dataset"]["data_type"], "video")
        self.assertEqual(resolved["dataset"]["dataloader_cfg"]["B"], 2)
        self.assertEqual(resolved["dataset"]["dataloader_cfg"]["long_T"], 16)
        self.assertEqual(resolved["tokenizer"]["encoder"]["n_latents"], 32)
        self.assertEqual(resolved["tokenizer"]["encoder"]["d_bottleneck"], 8)
        self.assertEqual(resolved["tokenizer"]["encoder"]["depth"], 2)
        self.assertEqual(resolved["tokenizer"]["encoder"]["d_model"], 128)
        self.assertEqual(resolved["tokenizer"]["decoder"]["depth"], 2)
        self.assertEqual(resolved["tokenizer"]["decoder"]["d_model"], 128)
        self.assertEqual(resolved["lpips_weight"], 0.0)

    def test_coinrun_dynamics_smoke_profile_composes(self):
        cfg = compose_config("coinrun_dynamics")
        resolved = train_dynamics.validate_dynamics_config(cfg)

        self.assertTrue(OmegaConf.is_struct(cfg))
        self.assertEqual(resolved["run_name"], "coinrun-dynamics-smoke")
        self.assertEqual(
            resolved["tokenizer_ckpt"],
            "logs/coinrun-tokenizer-smoke/checkpoints",
        )
        self.assertEqual(resolved["max_steps"], 10)
        self.assertEqual(resolved["dataset"]["name"], "coinrun")
        self.assertEqual(resolved["dataset"]["data_type"], "video")
        self.assertEqual(resolved["dataset"]["dataloader_cfg"]["B"], 2)
        self.assertEqual(resolved["dataset"]["dataloader_cfg"]["long_T"], 16)
        self.assertEqual(resolved["dynamics"]["depth"], 2)
        self.assertEqual(resolved["dynamics"]["d_model"], 128)
        self.assertEqual(resolved["dynamics"]["d_bottleneck"], 8)
        self.assertEqual(resolved["dynamics"]["context_length"], 16)
        self.assertEqual(resolved["dynamics"]["n_register"], 4)
        self.assertEqual(resolved["dynamics"]["categorical_action_dim"], 16)
        self.assertIsNone(resolved["dynamics"]["latent_mean"])
        self.assertIsNone(resolved["dynamics"]["latent_std"])
        self.assertEqual(resolved["bootstrap_start"], 100)
        self.assertFalse(resolved["ot"]["enabled"])

    def test_coinrun_profiles_do_not_change_minecraft_defaults(self):
        tokenizer_cfg = compose_config("tokenizer")
        dynamics_cfg = compose_config("dynamics")

        self.assertEqual(tokenizer_cfg.dataset.name, "minecraft_vpt")
        self.assertEqual(dynamics_cfg.dataset.name, "minecraft_vpt_latent")
        self.assertEqual(dynamics_cfg.dataset.num_binary_actions, 27)
        self.assertEqual(dynamics_cfg.dataset.categorical_action_dim, 121)
        train_dynamics.validate_dynamics_config(dynamics_cfg)

    def test_coinrun_config_rejects_unknown_override_without_explicit_add(self):
        with self.assertRaises(ConfigCompositionException):
            compose_config(
                "coinrun_dynamics",
                overrides=["dynamics.unrecognized_action_dim=16"],
            )


if __name__ == "__main__":
    unittest.main()
