from unittest import mock

from bestllm import HardwareSpecs, best_local_llm
from bestllm.models import DEFAULT_MODEL_PROFILES


def test_best_local_llm_invokes_selector_with_detected_specs() -> None:
    specs = HardwareSpecs(total_ram_gb=32, cpu_physical_cores=8, gpu_vram_gb=12)
    expected_profile = DEFAULT_MODEL_PROFILES[1]

    with (
        mock.patch("bestllm.HardwareSpecs.from_system", return_value=specs) as mock_from_system,
        mock.patch(
            "bestllm.recommend_model_for_specs",
            return_value=expected_profile,
        ) as mock_recommend,
    ):
        result = best_local_llm()

    assert result == expected_profile
    mock_from_system.assert_called_once_with()
    mock_recommend.assert_called_once_with(specs)
