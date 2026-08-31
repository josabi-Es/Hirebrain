"""Domain exceptions for the cv-creator pipeline."""


class CVCreatorError(Exception):
    """Base error for cv-creator."""


class LLMTimeoutError(CVCreatorError):
    """LLM request timed out or failed."""


class LLMInvalidSchemaError(CVCreatorError):
    """LLM output did not match expected JSON schema."""


class TemplateRenderError(CVCreatorError):
    """Jinja2 template rendering failed."""


class PdfRenderError(CVCreatorError):
    """HTML to PDF conversion failed."""


class PhotoGenerationError(CVCreatorError):
    """Profile photo generation failed."""
