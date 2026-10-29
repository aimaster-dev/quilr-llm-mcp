from collections.abc import AsyncIterator

import httpx
import pytest

from .helpers import Downstream, gateway_client


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def downstream() -> Downstream:
    return Downstream()


@pytest.fixture
async def gateway(downstream: Downstream) -> AsyncIterator[httpx.AsyncClient]:
    async with gateway_client(downstream) as client:
        yield client
