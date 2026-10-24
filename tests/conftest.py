import pytest
import numpy as np
from unittest.mock import Mock
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

import app.database as db_module
from app.main import app
from app.database import Base, get_db

TEST_DB_URL = "sqlite+aiosqlite:///:memory:"
TEST_WEBHOOK_SECRET = "test-secret"


@pytest.fixture
async def client(monkeypatch):
    monkeypatch.setattr("app.config.settings.WEBHOOK_SECRET", TEST_WEBHOOK_SECRET)
    monkeypatch.setattr("app.routers.webhooks.settings.WEBHOOK_SECRET", TEST_WEBHOOK_SECRET)

    test_engine = create_async_engine(TEST_DB_URL)
    TestSession = async_sessionmaker(test_engine, expire_on_commit=False)

    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    monkeypatch.setattr(db_module, "AsyncSessionLocal", TestSession)

    async def override_get_db():
        async with TestSession() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db

    mock_model = Mock()
    mock_model.encode.return_value = np.zeros((1, 384))

    mock_collection = Mock()
    mock_collection.query.return_value = {
        "documents": [[]],
        "metadatas": [[]],
        "distances": [[]],
    }
    mock_collection.add = Mock()

    app.state.test_mode = True
    app.state.embedding_model = mock_model
    app.state.chroma_collection = mock_collection

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac

    app.dependency_overrides.clear()
    app.state.test_mode = False
    await test_engine.dispose()
