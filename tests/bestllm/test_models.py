from bestllm.models import DEFAULT_MODEL_PROFILES, ModelProfile, iter_models


def test_catalog_contains_gpu_and_cpu_profiles() -> None:
    devices = {profile.preferred_device for profile in DEFAULT_MODEL_PROFILES}
    assert "gpu" in devices
    assert "cpu" in devices


def test_iter_models_returns_snapshot() -> None:
    snapshot = iter_models()
    assert isinstance(snapshot, tuple)
    assert snapshot
    assert isinstance(snapshot[0], ModelProfile)


def test_model_profile_serialization() -> None:
    profile = DEFAULT_MODEL_PROFILES[0]
    data = profile.as_dict()
    assert data["name"] == profile.name
    assert data["context_window"] == profile.context_window
