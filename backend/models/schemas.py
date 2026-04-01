"""
API Models — request and response schemas.

WHAT THESE DO:
FastAPI uses Pydantic models to automatically:
1. VALIDATE incoming requests (reject bad data with clear error messages)
2. SERIALIZE outgoing responses (convert Python objects to JSON)
3. GENERATE API documentation (the /docs page shows these schemas)

When someone sends a POST request with JSON body, FastAPI:
- Parses the JSON
- Validates it against the model (correct types? required fields present?)
- If valid: converts to a Python object your code can use
- If invalid: returns a 422 error with exactly which fields are wrong

You write ZERO validation code. The model IS the validation.

EXAMPLE:
    # Someone sends: {"question": "top sales?", "file_id": 123}
    # FastAPI sees file_id should be str | None, not int
    # Returns: 422 {"detail": [{"field": "file_id", "msg": "str type expected"}]}

WHAT YOU'RE LEARNING:
- Pydantic BaseModel for API schemas (different from BaseSettings in config)
- Optional fields with defaults
- How FastAPI auto-generates documentation from these models
"""

from pydantic import BaseModel


class QueryRequest(BaseModel):
    """Schema for the /query endpoint."""

    question: str
    file_id: str | None = None
    sheet_name: str | None = None
    chat_history: list[dict] | None = None

    # model_config is Pydantic v2's way of providing examples
    # for the auto-generated API docs. When you visit /docs,
    # the "Try it out" button will pre-fill these values.
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "question": "Who had the highest sales in the South region?",
                    "file_id": "abc123",
                    "sheet_name": None,
                    "chat_history": [],
                }
            ]
        }
    }


class FileInfo(BaseModel):
    """Schema for file information in responses."""

    file_id: str
    file_name: str
    total_chunks: int = 0
    total_rows_processed: int = 0
    sheets: list[str] = []
    status: str = "unknown"


class QueryResponse(BaseModel):
    """Schema for non-streaming query responses."""

    answer: str
    sources: list[dict] = []
    chunks_searched: int = 0


class DeleteResponse(BaseModel):
    """Schema for file deletion responses."""

    file_id: str
    deleted: bool
    message: str