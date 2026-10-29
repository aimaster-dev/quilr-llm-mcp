import os

from hypothesis import HealthCheck, settings

# CI runs a fixed, derandomized set of examples so the suite is deterministic and fast on slow
# runners. Run `HYPOTHESIS_PROFILE=thorough pytest` locally for a much deeper random search.
settings.register_profile(
    "ci",
    max_examples=300,
    deadline=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.register_profile("thorough", max_examples=20_000, deadline=None)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "ci"))
