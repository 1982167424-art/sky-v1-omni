import random

import numpy as np
import pytest

from sky_v1.utils.seed import set_global_seed
from sky_v1.utils.retry import with_retry, RetryableError
from sky_v1.utils.config import load_yaml_config, ConfigError
from sky_v1.utils.logging import get_logger


def test_seed_deterministic():
    set_global_seed(42)
    r1 = random.random()
    n1 = np.random.rand()
    set_global_seed(42)
    r2 = random.random()
    n2 = np.random.rand()
    assert r1 == r2
    assert n1 == n2


def test_retry_success_3times():
    call_counter = {"count": 0}

    @with_retry(max_attempts=3)
    def flaky():
        call_counter["count"] += 1
        if call_counter["count"] < 2:
            raise RetryableError("transient")
        return "ok"

    result = flaky()
    assert result == "ok"
    assert call_counter["count"] == 2


def test_retry_fail_non_retryable():
    call_counter = {"count": 0}

    @with_retry(max_attempts=3)
    def boom():
        call_counter["count"] += 1
        raise RuntimeError("non-retryable")

    with pytest.raises(RuntimeError):
        boom()
    assert call_counter["count"] == 1


def test_config_load_bad_path():
    with pytest.raises(ConfigError):
        load_yaml_config("/nonexistent/123.yaml")


def test_logger_name_prefix():
    logger = get_logger("foo")
    assert logger.name == "sky_v1.foo"
