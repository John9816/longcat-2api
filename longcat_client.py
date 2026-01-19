"""
LongCat client - HTTP communication with upstream API

Includes:
- SSE event parsing and conversion
- HTTP client for LongCat API
- OpenAI format conversion
"""
import json
import time
import uuid
from typing import AsyncGenerator, Optional, Dict, Any

import httpx


# ============================================================================
# SSE Parsing and OpenAI Conversion
# ============================================================================

def parse_sse_line(line: str) -> Optional[Dict[str, Any]]:
    """
    Parse a single SSE line.
    
    Format: data: {...json...}
    
    Args:
        line: Raw SSE line
        
    Returns:
        Parsed JSON dict or None if parse failed
    """
    line = line.strip()
    
    if not line or not line.startswith("data:"):
        return None
    
    try:
        json_str = line[5:].strip()  # Remove "data:"
        return json.loads(json_str)
    except (json.JSONDecodeError, ValueError):
        return None


def should_finish(event: Dict[str, Any]) -> bool:
    """
    Check if we should finish streaming.
    
    Conditions (any one is enough):
    - event.lastOne == true
    - event.event.type == "finish"
    - event.event.status == "FINISHED"
    
    Args:
        event: Parsed SSE event
        
    Returns:
        True if should finish
    """
    # Check top-level lastOne
    if event.get("lastOne") is True:
        return True
    
    # Check nested event.type and event.status
    nested_event = event.get("event", {})
    if isinstance(nested_event, dict):
        if nested_event.get("type") == "finish":
            return True
        if nested_event.get("status") == "FINISHED":
            return True
    
    return False


def should_skip_event(event: Dict[str, Any]) -> bool:
    """
    Check if event should be skipped (not sent to client).
    
    Skip events:
    - type == "think"
    - type == "reason"
    
    Args:
        event: Parsed SSE event
        
    Returns:
        True if should skip
    """
    nested_event = event.get("event", {})
    if isinstance(nested_event, dict):
        event_type = nested_event.get("type")
        if event_type in ("think", "reason"):
            return True
    
    return False


def get_content_from_event(event: Dict[str, Any]) -> Optional[str]:
    """
    Extract content from event.
    
    Priority:
    1. event.finalContentX (final answer)
    2. event.content (incremental)
    
    Args:
        event: Parsed SSE event
        
    Returns:
        Content string or None
    """
    nested_event = event.get("event", {})
    if isinstance(nested_event, dict):
        # Priority 1: finalContentX
        if "finalContentX" in nested_event and nested_event["finalContentX"]:
            return nested_event["finalContentX"]
        
        # Priority 2: content (but only if not think/reason)
        if "content" in nested_event and nested_event["content"]:
            return nested_event["content"]
    
    return None


def event_to_openai_chunk(event: Dict[str, Any], model: str = "longcat") -> str:
    """
    Convert upstream event to OpenAI chunk format.
    
    Returns OpenAI format string (already includes "data:" prefix):
    data: {"id":"...","object":"chat.completion.chunk",...}
    
    Args:
        event: Parsed SSE event
        model: Model name
        
    Returns:
        OpenAI chunk JSON string with data: prefix and double newline
    """
    content = get_content_from_event(event)
    
    if not content:
        # If no content, return empty delta (keep-alive chunk)
        content = ""
    
    chunk = {
        "id": str(uuid.uuid4()),
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"content": content},
                "finish_reason": None,
            }
        ],
    }
    
    # Note: Each chunk must end with double newline for proper SSE streaming
    return "data: " + json.dumps(chunk, ensure_ascii=False) + "\n\n"


def create_openai_response(content: str, model: str = "longcat") -> Dict[str, Any]:
    """
    Create OpenAI non-streaming response.
    
    Args:
        content: Final response content
        model: Model name
        
    Returns:
        OpenAI compatible response dict
    """
    return {
        "id": str(uuid.uuid4()),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


# ============================================================================
# LongCat HTTP Client
# ============================================================================

class LongCatClient:
    """Client for calling LongCat API"""
    
    def __init__(self):
        """
        Initialize LongCat client.
        
        Cookies are loaded from config (environment variable LONGCAT_COOKIE).
        """
        # Import here to avoid circular dependency
        from main import get_config
        
        self.config = get_config()
        self.cookies = self.config.longcat_cookie
        self.client = httpx.AsyncClient(timeout=self.config.stream_timeout)
    
    async def create_session(self, agent_id: str = "1") -> str:
        """Create a new conversation session"""
        headers = self._build_common_headers()
        headers["referer"] = "https://longcat.chat/"
        
        response = await self.client.post(
            "https://longcat.chat/api/v1/session-create",
            json={"model": "", "agentId": agent_id},
            headers=headers,
        )
        
        self._check_response_status(response)
        result = response.json()
        
        if result.get("code") != 0:
            raise Exception(f"Session creation failed: {result.get('message', 'Unknown error')}")
        
        conversation_id = result.get("data", {}).get("conversationId")
        if not conversation_id:
            raise Exception("No conversationId in response data")
        
        return conversation_id
    
    def _build_common_headers(self) -> dict:
        """Build common request headers"""
        return {
            "accept": "*/*",
            "content-type": "application/json",
            "origin": self.config.longcat_origin,
            "m-appkey": "fe_com.sankuai.friday.fe.longcat",
            "m-traceid": str(uuid.uuid4()),
            "x-client-language": "zh",
            "x-requested-with": "XMLHttpRequest",
            "user-agent": self.config.user_agent,
            "cookie": self.cookies,
        }
    
    def _check_response_status(self, response: httpx.Response):
        """Check response status and raise appropriate exceptions"""
        if response.status_code == 401:
            raise Exception("401: Unauthorized - Please check your login state")
        elif response.status_code == 403:
            raise Exception("403: Forbidden - Please verify manually or re-login")
        elif response.status_code == 429:
            raise Exception("429: Rate limited - Please try again later")
        elif response.status_code >= 500:
            raise Exception(f"{response.status_code}: LongCat server error")
        elif response.status_code != 200:
            raise Exception(f"Unexpected status: {response.status_code}")
    
    def _build_request_body(
        self,
        conversation_id: str,
        content: str,
        agent_id: str = "1",
    ) -> dict:
        """Build request body for LongCat API"""
        return {
            "conversationId": conversation_id,
            "content": content,
            "agentId": agent_id,
            "files": [],
            "creationParam": {},
            "reasonEnabled": 1,
            "searchEnabled": 1,
            "parentMessageId": 0,
        }
    
    async def chat_completion_stream(
        self,
        conversation_id: str,
        content: str,
        agent_id: str = "1",
    ) -> AsyncGenerator[str, None]:
        """Stream chat completion from LongCat, yields OpenAI-compatible chunks"""
        headers = self._build_common_headers()
        headers["accept"] = "text/event-stream,application/json"
        headers["referer"] = f"https://longcat.chat/c/{conversation_id}"
        
        body = self._build_request_body(conversation_id, content, agent_id)
        
        async with httpx.AsyncClient(timeout=self.config.stream_timeout) as client:
            async with client.stream("POST", self.config.longcat_api_url, json=body, headers=headers) as response:
                self._check_response_status(response)
                
                has_content = False
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    
                    event = parse_sse_line(line)
                    if not event or should_skip_event(event):
                        continue
                    
                    if content := get_content_from_event(event):
                        yield event_to_openai_chunk(event)
                        has_content = True
                    
                    if should_finish(event):
                        if not has_content and (final := get_content_from_event(event)):
                            yield event_to_openai_chunk(event)
                        yield "data: [DONE]\n\n"
                        return
    
    async def close(self):
        """Close client"""
        await self.client.aclose()
