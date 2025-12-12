"""
Test prompt endpoints for version filtering and history
"""

from unittest.mock import MagicMock

import pytest

from litellm.types.prompts.init_prompts import (
    PromptInfo,
    PromptLiteLLMParams,
    PromptSpec,
)


class TestPromptVersioning:
    """
    Test prompt versioning functionality
    """

    def test_get_latest_prompt_versions(self):
        """
        Test that get_latest_prompt_versions returns only the latest version of each prompt
        """
        from litellm.proxy.prompts.prompt_endpoints import get_latest_prompt_versions

        # Create mock prompts with different versions
        prompts = [
            PromptSpec(
                prompt_id="jack.v1",
                litellm_params=PromptLiteLLMParams(
                    prompt_id="jack",
                    prompt_integration="dotprompt",
                    dotprompt_content="v1 content"
                ),
                prompt_info=PromptInfo(prompt_type="db"),
            ),
            PromptSpec(
                prompt_id="jack.v2",
                litellm_params=PromptLiteLLMParams(
                    prompt_id="jack",
                    prompt_integration="dotprompt",
                    dotprompt_content="v2 content"
                ),
                prompt_info=PromptInfo(prompt_type="db"),
            ),
            PromptSpec(
                prompt_id="jane.v1",
                litellm_params=PromptLiteLLMParams(
                    prompt_id="jane",
                    prompt_integration="dotprompt",
                    dotprompt_content="jane v1"
                ),
                prompt_info=PromptInfo(prompt_type="db"),
            ),
            PromptSpec(
                prompt_id="jack.v3",
                litellm_params=PromptLiteLLMParams(
                    prompt_id="jack",
                    prompt_integration="dotprompt",
                    dotprompt_content="v3 content"
                ),
                prompt_info=PromptInfo(prompt_type="db"),
            ),
        ]

        # Get latest versions
        latest = get_latest_prompt_versions(prompts=prompts)

        # Should return 2 prompts (jack.v3 and jane.v1)
        assert len(latest) == 2

        # Find jack and jane in results
        jack_prompt = next((p for p in latest if "jack" in p.prompt_id), None)
        jane_prompt = next((p for p in latest if "jane" in p.prompt_id), None)

        assert jack_prompt is not None
        assert jack_prompt.prompt_id == "jack.v3"
        assert jack_prompt.litellm_params.dotprompt_content == "v3 content"

        assert jane_prompt is not None
        assert jane_prompt.prompt_id == "jane.v1"

    def test_get_version_number(self):
        """
        Test that get_version_number correctly extracts version numbers
        """
        from litellm.proxy.prompts.prompt_endpoints import get_version_number

        assert get_version_number(prompt_id="jack.v1") == 1
        assert get_version_number(prompt_id="jack.v2") == 2
        assert get_version_number(prompt_id="jack.v10") == 10
        assert get_version_number(prompt_id="jack") == 1
        assert get_version_number(prompt_id="jack.vinvalid") == 1

    def test_get_base_prompt_id(self):
        """
        Test that get_base_prompt_id correctly strips version suffixes
        """
        from litellm.proxy.prompts.prompt_endpoints import get_base_prompt_id

        assert get_base_prompt_id(prompt_id="jack.v1") == "jack"
        assert get_base_prompt_id(prompt_id="jack.v2") == "jack"
        assert get_base_prompt_id(prompt_id="jack") == "jack"
        assert get_base_prompt_id(prompt_id="my_prompt.v10") == "my_prompt"

    def test_get_latest_version_prompt_id(self):
        """
        Test that get_latest_version_prompt_id returns the highest version
        """
        from litellm.proxy.prompts.prompt_endpoints import get_latest_version_prompt_id

        # Mock prompt IDs dictionary
        all_prompt_ids = {
            "jack.v1": {},
            "jack.v2": {},
            "jack.v3": {},
            "jane.v1": {},
            "simple_prompt": {},
        }

        # Test with base prompt ID - should return latest version
        assert get_latest_version_prompt_id(
            prompt_id="jack",
            all_prompt_ids=all_prompt_ids
        ) == "jack.v3"

        # Test with versioned prompt ID - should still return latest version
        assert get_latest_version_prompt_id(
            prompt_id="jack.v1",
            all_prompt_ids=all_prompt_ids
        ) == "jack.v3"

        # Test with single version
        assert get_latest_version_prompt_id(
            prompt_id="jane",
            all_prompt_ids=all_prompt_ids
        ) == "jane.v1"

        # Test with non-versioned prompt
        assert get_latest_version_prompt_id(
            prompt_id="simple_prompt",
            all_prompt_ids=all_prompt_ids
        ) == "simple_prompt"

        # Test with non-existent prompt
        assert get_latest_version_prompt_id(
            prompt_id="nonexistent",
            all_prompt_ids=all_prompt_ids
        ) == "nonexistent"

    def test_construct_versioned_prompt_id(self):
        """
        Test that construct_versioned_prompt_id correctly builds versioned IDs
        """
        from litellm.proxy.prompts.prompt_endpoints import construct_versioned_prompt_id

        # Test with base prompt ID and version
        assert construct_versioned_prompt_id(
            prompt_id="jack_success",
            version=4
        ) == "jack_success.v4"

        # Test with None version - should return base ID unchanged
        assert construct_versioned_prompt_id(
            prompt_id="jack_success",
            version=None
        ) == "jack_success"

        # Test with existing versioned ID - should replace version
        assert construct_versioned_prompt_id(
            prompt_id="jack_success.v2",
            version=4
        ) == "jack_success.v4"

        # Test with hyphenated prompt ID
        assert construct_versioned_prompt_id(
            prompt_id="my-prompt",
            version=1
        ) == "my-prompt.v1"

        # Test with double-digit version
        assert construct_versioned_prompt_id(
            prompt_id="test_prompt",
            version=10
        ) == "test_prompt.v10"


class TestPromptVersionsEndpoint:
    """
    Test the /prompts/{prompt_id}/versions endpoint
    """

    @pytest.mark.asyncio
    async def test_get_prompt_versions_returns_all_versions(self):
        """
        Test that get_prompt_versions returns all versions of a prompt sorted by version number
        """
        from unittest.mock import MagicMock, patch

        from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
        from litellm.proxy.prompts.prompt_endpoints import get_prompt_versions

        # Mock user with admin role
        mock_user = UserAPIKeyAuth(
            api_key="test_key",
            user_role=LitellmUserRoles.PROXY_ADMIN
        )

        # Create mock prompt registry with multiple versions
        mock_prompts = {
            "jack.v1": PromptSpec(
                prompt_id="jack.v1",
                litellm_params=PromptLiteLLMParams(
                    prompt_id="jack",
                    prompt_integration="dotprompt",
                    dotprompt_content="v1"
                ),
                prompt_info=PromptInfo(prompt_type="db"),
            ),
            "jack.v2": PromptSpec(
                prompt_id="jack.v2",
                litellm_params=PromptLiteLLMParams(
                    prompt_id="jack",
                    prompt_integration="dotprompt",
                    dotprompt_content="v2"
                ),
                prompt_info=PromptInfo(prompt_type="db"),
            ),
            "jack.v3": PromptSpec(
                prompt_id="jack.v3",
                litellm_params=PromptLiteLLMParams(
                    prompt_id="jack",
                    prompt_integration="dotprompt",
                    dotprompt_content="v3"
                ),
                prompt_info=PromptInfo(prompt_type="db"),
            ),
            "jane.v1": PromptSpec(
                prompt_id="jane.v1",
                litellm_params=PromptLiteLLMParams(
                    prompt_id="jane",
                    prompt_integration="dotprompt",
                    dotprompt_content="jane"
                ),
                prompt_info=PromptInfo(prompt_type="db"),
            ),
        }

        # Mock the IN_MEMORY_PROMPT_REGISTRY at the import location
        with patch("litellm.proxy.prompts.prompt_registry.IN_MEMORY_PROMPT_REGISTRY") as mock_registry:
            mock_registry.IN_MEMORY_PROMPTS = mock_prompts

            # Test with base prompt ID
            response = await get_prompt_versions(
                prompt_id="jack",
                user_api_key_dict=mock_user
            )

            # Should return 3 versions of jack, sorted newest first
            assert len(response.prompts) == 3
            assert response.prompts[0].prompt_id == "jack"
            assert response.prompts[0].version == 3
            assert response.prompts[1].prompt_id == "jack"
            assert response.prompts[1].version == 2
            assert response.prompts[2].prompt_id == "jack"
            assert response.prompts[2].version == 1

            # Test with versioned prompt ID (should strip version)
            response = await get_prompt_versions(
                prompt_id="jack.v1",
                user_api_key_dict=mock_user
            )

            assert len(response.prompts) == 3
            assert response.prompts[0].prompt_id == "jack"
            assert response.prompts[0].version == 3

    @pytest.mark.asyncio
    async def test_get_prompt_versions_not_found(self):
        """
        Test that get_prompt_versions raises 404 when prompt doesn't exist
        """
        from unittest.mock import patch

        from fastapi import HTTPException

        from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
        from litellm.proxy.prompts.prompt_endpoints import get_prompt_versions

        mock_user = UserAPIKeyAuth(
            api_key="test_key",
            user_role=LitellmUserRoles.PROXY_ADMIN
        )

        with patch("litellm.proxy.prompts.prompt_registry.IN_MEMORY_PROMPT_REGISTRY") as mock_registry:
            mock_registry.IN_MEMORY_PROMPTS = {}

            with pytest.raises(HTTPException) as exc_info:
                await get_prompt_versions(
                    prompt_id="nonexistent",
                    user_api_key_dict=mock_user
                )

            assert exc_info.value.status_code == 404
            assert "No versions found" in exc_info.value.detail


class TestPromptDeleteAndUpdate:
    """
    Test prompt delete and update operations using primary key (id)
    """

    @pytest.mark.asyncio
    async def test_delete_prompt_uses_primary_key(self):
        """
        Test that delete_prompt uses the record's primary key (id) instead of prompt_id
        This fixes the Prisma FieldNotFoundError issue
        """
        from unittest.mock import AsyncMock, MagicMock, patch

        from fastapi import HTTPException

        from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
        from litellm.proxy.prompts.prompt_endpoints import delete_prompt

        # Mock user with admin role
        mock_user = UserAPIKeyAuth(
            api_key="test_key",
            user_role=LitellmUserRoles.PROXY_ADMIN
        )

        # Create mock prompt in registry
        mock_prompt = PromptSpec(
            prompt_id="test_prompt.v2",
            litellm_params=PromptLiteLLMParams(
                prompt_id="test_prompt",
                prompt_integration="dotprompt",
                dotprompt_content="test content"
            ),
            prompt_info=PromptInfo(prompt_type="db"),
        )

        # Mock Prisma client with database record
        mock_prisma_client = MagicMock()
        mock_prompt_table = MagicMock()
        
        # Mock database record with primary key id
        mock_db_record = MagicMock()
        mock_db_record.id = "db-record-uuid-12345"
        mock_db_record.prompt_id = "test_prompt"
        mock_db_record.version = 2
        
        # Mock find_first to return the record
        mock_prompt_table.find_first = AsyncMock(return_value=mock_db_record)
        # Mock delete to verify it's called with id (primary key)
        mock_prompt_table.delete = AsyncMock(return_value=None)
        
        mock_prisma_client.db.litellm_prompttable = mock_prompt_table

        # Mock prompt registry
        mock_registry = MagicMock()
        mock_registry.IN_MEMORY_PROMPTS = {"test_prompt.v2": mock_prompt}
        mock_registry.prompt_id_to_custom_prompt = {}
        mock_registry.get_prompt_by_id = MagicMock(return_value=mock_prompt)

        with patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client), \
             patch("litellm.proxy.prompts.prompt_registry.IN_MEMORY_PROMPT_REGISTRY", mock_registry):

            # Call delete_prompt
            result = await delete_prompt(
                prompt_id="test_prompt.v2",
                user_api_key_dict=mock_user
            )

            # Verify find_first was called with prompt_id + version (composite key)
            mock_prompt_table.find_first.assert_called_once()
            call_args = mock_prompt_table.find_first.call_args
            assert call_args[1]["where"]["prompt_id"] == "test_prompt"
            assert call_args[1]["where"]["version"] == 2

            # Verify delete was called with id (primary key), not prompt_id
            mock_prompt_table.delete.assert_called_once()
            delete_call_args = mock_prompt_table.delete.call_args
            assert delete_call_args[1]["where"]["id"] == "db-record-uuid-12345"
            assert "prompt_id" not in delete_call_args[1]["where"]

            # Verify prompt was removed from memory
            assert "test_prompt.v2" not in mock_registry.IN_MEMORY_PROMPTS

            # Verify success message
            assert result["message"] == "Prompt test_prompt.v2 deleted successfully"

    @pytest.mark.asyncio
    async def test_delete_prompt_not_found_in_database(self):
        """
        Test that delete_prompt raises 404 when prompt record is not found in database
        """
        from unittest.mock import AsyncMock, MagicMock, patch

        from fastapi import HTTPException

        from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
        from litellm.proxy.prompts.prompt_endpoints import delete_prompt

        mock_user = UserAPIKeyAuth(
            api_key="test_key",
            user_role=LitellmUserRoles.PROXY_ADMIN
        )

        mock_prompt = PromptSpec(
            prompt_id="test_prompt.v1",
            litellm_params=PromptLiteLLMParams(
                prompt_id="test_prompt",
                prompt_integration="dotprompt",
            ),
            prompt_info=PromptInfo(prompt_type="db"),
        )

        # Mock Prisma client - find_first returns None (not found)
        mock_prisma_client = MagicMock()
        mock_prompt_table = MagicMock()
        mock_prompt_table.find_first = AsyncMock(return_value=None)
        mock_prisma_client.db.litellm_prompttable = mock_prompt_table

        mock_registry = MagicMock()
        mock_registry.IN_MEMORY_PROMPTS = {"test_prompt.v1": mock_prompt}
        mock_registry.get_prompt_by_id = MagicMock(return_value=mock_prompt)

        with patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client), \
             patch("litellm.proxy.prompts.prompt_registry.IN_MEMORY_PROMPT_REGISTRY", mock_registry):

            with pytest.raises(HTTPException) as exc_info:
                await delete_prompt(  # type: ignore[misc]
                    prompt_id="test_prompt.v1",
                    user_api_key_dict=mock_user
                )

            assert exc_info.value.status_code == 404
            assert "not found in database" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_patch_prompt_uses_primary_key(self):
        """
        Test that patch_prompt uses the record's primary key (id) instead of prompt_id
        This fixes the Prisma FieldNotFoundError issue
        """
        from unittest.mock import AsyncMock, MagicMock, patch
        import json

        from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
        from litellm.proxy.prompts.prompt_endpoints import PatchPromptRequest, patch_prompt

        mock_user = UserAPIKeyAuth(
            api_key="test_key",
            user_role=LitellmUserRoles.PROXY_ADMIN
        )

        # Create existing prompt
        existing_prompt = PromptSpec(
            prompt_id="test_prompt.v1",
            litellm_params=PromptLiteLLMParams(
                prompt_id="test_prompt",
                prompt_integration="dotprompt",
                dotprompt_content="original content"
            ),
            prompt_info=PromptInfo(prompt_type="db"),
        )

        # Mock Prisma client
        mock_prisma_client = MagicMock()
        mock_prompt_table = MagicMock()
        
        # Mock database record
        mock_db_record = MagicMock()
        mock_db_record.id = "db-record-uuid-67890"
        mock_db_record.prompt_id = "test_prompt"
        mock_db_record.version = 1
        
        # Mock updated record
        updated_db_record = MagicMock()
        updated_db_record.id = "db-record-uuid-67890"
        updated_db_record.prompt_id = "test_prompt"
        updated_db_record.version = 1
        updated_db_record.litellm_params = json.dumps({
            "prompt_id": "test_prompt",
            "prompt_integration": "dotprompt",
            "dotprompt_content": "updated content"
        })
        updated_db_record.prompt_info = json.dumps({"prompt_type": "db"})
        updated_db_record.model_dump = MagicMock(return_value={
            "prompt_id": "test_prompt",
            "version": 1,
            "litellm_params": {
                "prompt_id": "test_prompt",
                "prompt_integration": "dotprompt",
                "dotprompt_content": "updated content"
            },
            "prompt_info": {"prompt_type": "db"}
        })
        
        mock_prompt_table.find_first = AsyncMock(return_value=mock_db_record)
        mock_prompt_table.update = AsyncMock(return_value=updated_db_record)
        
        mock_prisma_client.db.litellm_prompttable = mock_prompt_table

        # Mock prompt registry
        mock_registry = MagicMock()
        mock_registry.IN_MEMORY_PROMPTS = {"test_prompt.v1": existing_prompt}
        mock_registry.prompt_id_to_custom_prompt = {}
        mock_registry.get_prompt_by_id = MagicMock(return_value=existing_prompt)
        mock_registry.initialize_prompt = MagicMock(return_value=existing_prompt)

        # Create patch request
        patch_request = PatchPromptRequest(
            litellm_params=PromptLiteLLMParams(
                prompt_id="test_prompt",
                prompt_integration="dotprompt",
                dotprompt_content="updated content"
            ),
            prompt_info=PromptInfo(prompt_type="db"),
        )

        with patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client), \
             patch("litellm.proxy.prompts.prompt_registry.IN_MEMORY_PROMPT_REGISTRY", mock_registry):

            # Call patch_prompt
            result = await patch_prompt(
                prompt_id="test_prompt.v1",
                request=patch_request,
                user_api_key_dict=mock_user
            )

            # Verify find_first was called with prompt_id + version
            mock_prompt_table.find_first.assert_called_once()
            call_args = mock_prompt_table.find_first.call_args
            assert call_args[1]["where"]["prompt_id"] == "test_prompt"
            assert call_args[1]["where"]["version"] == 1

            # Verify update was called with id (primary key), not prompt_id
            mock_prompt_table.update.assert_called_once()
            update_call_args = mock_prompt_table.update.call_args
            assert update_call_args[1]["where"]["id"] == "db-record-uuid-67890"
            assert "prompt_id" not in update_call_args[1]["where"]

            # Verify prompt was removed from memory before re-initialization
            assert "test_prompt.v1" not in mock_registry.IN_MEMORY_PROMPTS

    @pytest.mark.asyncio
    async def test_patch_prompt_not_found_in_database(self):
        """
        Test that patch_prompt raises 404 when prompt record is not found in database
        """
        from unittest.mock import AsyncMock, MagicMock, patch

        from fastapi import HTTPException

        from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
        from litellm.proxy.prompts.prompt_endpoints import PatchPromptRequest, patch_prompt

        mock_user = UserAPIKeyAuth(
            api_key="test_key",
            user_role=LitellmUserRoles.PROXY_ADMIN
        )

        existing_prompt = PromptSpec(
            prompt_id="test_prompt.v1",
            litellm_params=PromptLiteLLMParams(
                prompt_id="test_prompt",
                prompt_integration="dotprompt",
            ),
            prompt_info=PromptInfo(prompt_type="db"),
        )

        # Mock Prisma client - find_first returns None
        mock_prisma_client = MagicMock()
        mock_prompt_table = MagicMock()
        mock_prompt_table.find_first = AsyncMock(return_value=None)
        mock_prisma_client.db.litellm_prompttable = mock_prompt_table

        mock_registry = MagicMock()
        mock_registry.IN_MEMORY_PROMPTS = {"test_prompt.v1": existing_prompt}
        mock_registry.get_prompt_by_id = MagicMock(return_value=existing_prompt)

        patch_request = PatchPromptRequest(
            litellm_params=PromptLiteLLMParams(
                prompt_id="test_prompt",
                prompt_integration="dotprompt",
            ),
        )

        with patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client), \
             patch("litellm.proxy.prompts.prompt_registry.IN_MEMORY_PROMPT_REGISTRY", mock_registry):

            with pytest.raises(HTTPException) as exc_info:
                await patch_prompt(  # type: ignore[misc]
                    prompt_id="test_prompt.v1",
                    request=patch_request,
                    user_api_key_dict=mock_user
                )

            assert exc_info.value.status_code == 404
            assert "not found in database" in exc_info.value.detail

