"""
FastAPI main application - OpenAI compatible API gateway

Includes:
- Configuration management
- Authentication
- OpenAI schemas
- API endpoints
"""
import os
import time
import json
import logging
import uuid
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, AsyncGenerator

from fastapi import FastAPI, Header, HTTPException, status
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Load .env file
from dotenv import load_dotenv
load_dotenv()


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class Config:
    """Application configuration"""
    
    api_key: str
    longcat_cookie: str
    host: str = "0.0.0.0"
    port: int = 8000
    longcat_api_url: str = "https://longcat.chat/api/v1/chat-completion-V2"
    longcat_origin: str = "https://longcat.chat"
    user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    request_timeout: int = 60
    stream_timeout: int = 120
    debug: bool = False
    
    @classmethod
    def from_env(cls) -> "Config":
        """Load config from environment variables"""
        api_key = os.getenv("API_KEY")
        longcat_cookie = os.getenv("LONGCAT_COOKIE")
        
        if not api_key:
            raise ValueError(
                "API_KEY environment variable not set. "
                "Set it before running: export API_KEY=your_key"
            )
        
        if not longcat_cookie:
            raise ValueError(
                "LONGCAT_COOKIE environment variable not set. "
                "Please set your LongCat cookie: export LONGCAT_COOKIE='your_cookie_string'"
            )
        
        return cls(
            api_key=api_key,
            longcat_cookie=longcat_cookie,
            host=os.getenv("HOST", "0.0.0.0"),
            port=int(os.getenv("PORT", "8000")),
            longcat_api_url=os.getenv(
                "LONGCAT_API_URL",
                "https://longcat.chat/api/v1/chat-completion-V2"
            ),
            longcat_origin=os.getenv("LONGCAT_ORIGIN", "https://longcat.chat"),
            user_agent=os.getenv(
                "USER_AGENT",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            ),
            request_timeout=int(os.getenv("REQUEST_TIMEOUT", "60")),
            stream_timeout=int(os.getenv("STREAM_TIMEOUT", "120")),
            debug=os.getenv("DEBUG", "false").lower() == "true",
        )


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get or initialize global config"""
    global _config
    if _config is None:
        _config = Config.from_env()
    return _config


# ============================================================================
# Authentication
# ============================================================================

async def verify_token(authorization: str | None) -> bool:
    """
    Verify Bearer token from Authorization header.
    
    Args:
        authorization: "Bearer <token>" or None
        
    Returns:
        True if token is valid
        
    Raises:
        HTTPException: 401 if token is missing or invalid
    """
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header missing",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    parts = authorization.split()
    
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization format. Use: Bearer <token>",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    token = parts[1]
    config = get_config()
    
    if token != config.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return True


# ============================================================================
# OpenAI Schemas
# ============================================================================

class Message(BaseModel):
    """A chat message"""
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    """OpenAI chat completion request"""
    model: str
    messages: List[Message]
    stream: bool = False
    conversation_id: Optional[str] = Field(None, alias="conversation_id")
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    
    class Config:
        allow_population_by_field_name = True


class DeltaChoice(BaseModel):
    """Delta content in streaming response"""
    index: int = 0
    delta: Dict[str, Any]
    finish_reason: Optional[str] = None


class ChatCompletionChunk(BaseModel):
    """OpenAI streaming response chunk"""
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[DeltaChoice]


class Choice(BaseModel):
    """Choice in non-streaming response"""
    index: int = 0
    message: Message
    finish_reason: str = "stop"


class Usage(BaseModel):
    """Token usage information"""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    """OpenAI non-streaming response"""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Usage


class ModelObject(BaseModel):
    """Model object in model list"""
    id: str
    object: str = "model"
    created: int
    owned_by: str = "longcat"


class ModelListResponse(BaseModel):
    """Response for /v1/models"""
    object: str = "list"
    data: List[ModelObject]


# ============================================================================
# FastAPI Application
# ============================================================================

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="LongCat2API",
    description="OpenAI-compatible API gateway for LongCat",
    version="1.0.0",
)

# Global config
config = None


@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    global config
    try:
        config = get_config()
        logger.info(f"Config loaded. API listening on {config.host}:{config.port}")
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        raise


@app.get("/v1/models")
async def list_models() -> ModelListResponse:
    """List available models"""
    return ModelListResponse(
        data=[
            ModelObject(id="longcat-agent-1", created=int(time.time())),
            ModelObject(id="longcat-agent-2", created=int(time.time())),
        ]
    )


@app.post("/v1/chat/completions", response_model=None)
async def chat_completion(
    request: ChatCompletionRequest,
    authorization: str | None = Header(None),
):
    """
    Chat completion endpoint (OpenAI compatible).
    
    Request:
        - model: Model name to use (mapped to agentId)
        - messages: Chat messages (uses last user message)
        - stream: Enable streaming (default: false)
        - conversation_id: LongCat conversation ID (optional, auto-created if not provided)
        
    Response:
        - If stream=false: Standard OpenAI JSON response
        - If stream=true: SSE stream with OpenAI chunks
    """
    # Import here to avoid circular dependency
    from longcat_client import LongCatClient, create_openai_response
    
    try:
        # Verify authentication
        await verify_token(authorization)
        
        # Validate required fields
        if not request.messages:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="messages field is required",
            )
        
        # Extract last user message
        user_content = None
        for msg in reversed(request.messages):
            if msg.role == "user":
                user_content = msg.content
                break
        
        if not user_content:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No user message found in messages",
            )
        
        # Map model to agent_id
        agent_id = request.model.split("-")[-1] if "-" in request.model else "1"
        
        # Get or create conversation_id
        conversation_id = request.conversation_id
        created_new_session = False
        
        if not conversation_id:
            # Auto-create new session if not provided
            logger.info(f"No conversation_id provided, creating new session for agent {agent_id}...")
            try:
                client = LongCatClient()
                conversation_id = await client.create_session(agent_id=agent_id)
                created_new_session = True
                logger.info(f"Created new session: {conversation_id}")
            except Exception as e:
                logger.error(f"Failed to create session: {e}")
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=f"Failed to create conversation: {str(e)}",
                )
            finally:
                await client.close()
        
        logger.info(
            f"Chat completion request: model={request.model}, "
            f"agent_id={agent_id}, conversation_id={conversation_id}, stream={request.stream}"
        )
        
        # Create LongCat client
        client = LongCatClient()
        
        # Handle streaming response
        if request.stream:
            async def stream_generator() -> AsyncGenerator[str, None]:
                """Generate streaming response"""
                try:
                    async for chunk in client.chat_completion_stream(
                        conversation_id=conversation_id,
                        content=user_content,
                        agent_id=agent_id,
                    ):
                        yield chunk
                except Exception as e:
                    logger.error(f"Stream error: {e}")
                    # Map upstream errors to proper HTTP status codes
                    error_msg = str(e)
                    if "401" in error_msg:
                        error_chunk = json.dumps({
                            "error": {
                                "message": "Unauthorized - Please login again",
                                "type": "authentication_error",
                                "code": 401
                            }
                        })
                    elif "403" in error_msg:
                        error_chunk = json.dumps({
                            "error": {
                                "message": "Forbidden - Please complete manual verification",
                                "type": "forbidden_error",
                                "code": 403
                            }
                        })
                    elif "429" in error_msg:
                        error_chunk = json.dumps({
                            "error": {
                                "message": "Rate limited - Please try again later",
                                "type": "rate_limit_error",
                                "code": 429
                            }
                        })
                    else:
                        error_chunk = json.dumps({
                            "error": {
                                "message": f"Upstream error: {error_msg}",
                                "type": "server_error",
                                "code": 502
                            }
                        })
                    yield f"data: {error_chunk}\n\n"
                finally:
                    await client.close()
            
            return StreamingResponse(
                stream_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )
        
        # Handle non-streaming response
        else:
            collected_content = ""
            
            try:
                # Collect full response
                async for chunk_str in client.chat_completion_stream(
                    conversation_id=conversation_id,
                    content=user_content,
                    agent_id=agent_id,
                ):
                    # Skip [DONE] marker
                    if chunk_str.strip() == "data: [DONE]":
                        continue
                    
                    # Parse chunk
                    if chunk_str.startswith("data: "):
                        try:
                            data_str = chunk_str[6:].strip()
                            if not data_str:
                                continue
                            
                            chunk_json = json.loads(data_str)
                            
                            # Extract content from choices
                            if "choices" in chunk_json:
                                for choice in chunk_json["choices"]:
                                    if "delta" in choice and "content" in choice["delta"]:
                                        content = choice["delta"]["content"]
                                        if content:
                                            collected_content += content
                        except json.JSONDecodeError:
                            pass
                
                # Return collected response
                return create_openai_response(collected_content, model=request.model)
                
            except Exception as e:
                logger.error(f"Non-streaming error: {e}")
                # Map errors to proper HTTP status codes
                error_msg = str(e)
                if "401" in error_msg:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Unauthorized - Please login again"
                    )
                elif "403" in error_msg:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Forbidden - Please complete manual verification"
                    )
                elif "429" in error_msg:
                    raise HTTPException(
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        detail="Rate limited - Please try again later"
                    )
                else:
                    raise HTTPException(
                        status_code=status.HTTP_502_BAD_GATEWAY,
                        detail=f"Upstream error: {error_msg}"
                    )
            finally:
                await client.close()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat completion error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Internal server error: {str(e)}",
        )


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "ok"}


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "LongCat2API",
        "version": "1.0.0",
        "endpoints": {
            "models": "GET /v1/models",
            "chat": "POST /v1/chat/completions",
            "health": "GET /health",
        },
    }


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_502_BAD_GATEWAY,
        content={"error": str(exc)},
    )


if __name__ == "__main__":
    config = get_config()
    logger.info(f"Starting LongCat2API on {config.host}:{config.port}")
    
    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        log_level="info" if not config.debug else "debug",
    )
