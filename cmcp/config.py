# cmcp/config.py
# container-mcp © 2025 by Martin Bukowski is licensed under Apache 2.0

"""Configuration module for Container-MCP."""

import os
import logging
import tempfile
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, validator

# Configure logging
logger = logging.getLogger(__name__)


# Detect environment
def is_in_container() -> bool:
    """Check if we're running inside a container."""
    return os.path.exists('/.dockerenv') or os.path.exists('/run/.containerenv')


# Determine base paths
def get_base_paths() -> Dict[str, str]:
    """Get base paths based on environment."""
    if is_in_container():
        logger.info("Running in container environment")
        return {
            "base_dir": "/app",
            "sandbox_root": "/app/sandbox",
            "temp_dir": "/app/temp"
        }
    else:
        logger.info("Running in local environment")
        # When running locally, use current directory or temp directory
        cwd = os.getcwd()
        
        # Check if we can use local directories
        sandbox_dir = os.path.join(cwd, "sandbox")
        temp_dir = os.path.join(cwd, "temp")
        
        # If we can't create/access these directories, fall back to temp
        try:
            os.makedirs(sandbox_dir, exist_ok=True)
            os.makedirs(temp_dir, exist_ok=True)
        except (PermissionError, OSError):
            logger.warning("Cannot use local directories, falling back to temp directory")
            base_temp = tempfile.gettempdir()
            sandbox_dir = os.path.join(base_temp, "cmcp-sandbox")
            temp_dir = os.path.join(base_temp, "cmcp-temp")
            os.makedirs(sandbox_dir, exist_ok=True)
            os.makedirs(temp_dir, exist_ok=True)
        
        return {
            "base_dir": cwd,
            "sandbox_root": sandbox_dir,
            "temp_dir": temp_dir
        }


# Get paths for current environment
BASE_PATHS = get_base_paths()


class BashConfig(BaseModel):
    """Configuration for Bash Manager."""

    sandbox_dir: str = Field(default=BASE_PATHS["sandbox_root"])
    allowed_commands: List[str] = Field(default_factory=list)
    command_restricted: bool = Field(default=True)
    timeout_default: int = Field(default=30)
    timeout_max: int = Field(default=120)


class PythonConfig(BaseModel):
    """Configuration for Python Manager."""

    sandbox_dir: str = Field(default=BASE_PATHS["sandbox_root"])
    memory_limit: int = Field(default=256)
    timeout_default: int = Field(default=30)
    timeout_max: int = Field(default=120)


class FileSystemConfig(BaseModel):
    """Configuration for File Manager."""

    base_dir: str = Field(default=BASE_PATHS["sandbox_root"])
    max_file_size_mb: int = Field(default=10)
    allowed_extensions: List[str] = Field(default_factory=list)


class WebConfig(BaseModel):
    """Configuration for Web Manager."""

    timeout_default: int = Field(default=30)
    allowed_domains: Optional[List[str]] = Field(default=None)


class KBConfig(BaseModel):
    """Configuration for Knowledge Base Manager."""
    
    storage_path: str = Field(default=os.path.join(BASE_PATHS["base_dir"], "kb"))
    timeout_default: int = Field(default=30)
    timeout_max: int = Field(default=120)


class MCPConfig(BaseModel):
    """MCP Server configuration."""

    host: str = Field(default="127.0.0.1")
    port: int = Field(default=8000)
    debug: bool = Field(default=False)


class AppConfig(BaseModel):
    """Main application configuration."""

    mcp_host: str = Field(default="127.0.0.1")
    mcp_port: int = Field(default=8000)
    debug: bool = Field(default=False)
    log_level: str = Field(default="INFO")
    sandbox_root: str = Field(default=BASE_PATHS["sandbox_root"])
    temp_dir: str = Field(default=BASE_PATHS["temp_dir"])
    
    bash_config: BashConfig = Field(default_factory=BashConfig)
    python_config: PythonConfig = Field(default_factory=PythonConfig)
    filesystem_config: FileSystemConfig = Field(default_factory=FileSystemConfig)
    web_config: WebConfig = Field(default_factory=WebConfig)
    kb_config: KBConfig = Field(default_factory=KBConfig)
    matlab_config: "MatlabConfig" # Forward declaration

    @validator("log_level")
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v


def load_env_config() -> Dict[str, Any]:
    """Load configuration from environment variables."""
    config = {}
    
    # MCP Server config
    config["mcp_host"] = os.environ.get("MCP_HOST", "127.0.0.1")
    config["mcp_port"] = int(os.environ.get("MCP_PORT", "8000"))
    config["debug"] = os.environ.get("DEBUG", "false").lower() == "true"
    config["log_level"] = os.environ.get("LOG_LEVEL", "INFO")
    
    # Sandbox config - use env vars if provided, otherwise use detected paths
    config["base_dir"] = os.environ.get("BASE_DIR", BASE_PATHS["base_dir"])
    config["sandbox_root"] = os.environ.get("SANDBOX_ROOT", BASE_PATHS["sandbox_root"])
    config["temp_dir"] = os.environ.get("TEMP_DIR", BASE_PATHS["temp_dir"])
    
    # Create necessary directory
    os.makedirs(config["sandbox_root"], exist_ok=True)
    os.makedirs(config["temp_dir"], exist_ok=True)
    
    # Bash config
    command_restricted = os.environ.get("COMMAND_RESTRICTED", "true").lower() == "true"
    bash_config = BashConfig(
        sandbox_dir=config["sandbox_root"],
        allowed_commands=os.environ.get("BASH_ALLOWED_COMMANDS", "").split(",") if os.environ.get("BASH_ALLOWED_COMMANDS") else [],
        command_restricted=command_restricted,
        timeout_default=int(os.environ.get("BASH_TIMEOUT_DEFAULT", "30")),
        timeout_max=int(os.environ.get("BASH_TIMEOUT_MAX", "120")),
    )
    config["bash_config"] = bash_config
    
    # Python config
    python_config = PythonConfig(
        sandbox_dir=config["sandbox_root"],
        memory_limit=int(os.environ.get("PYTHON_MEMORY_LIMIT", "256")),
        timeout_default=int(os.environ.get("PYTHON_TIMEOUT_DEFAULT", "30")),
        timeout_max=int(os.environ.get("PYTHON_TIMEOUT_MAX", "120")),
    )
    config["python_config"] = python_config
    
    # File system config
    filesystem_config = FileSystemConfig(
        base_dir=config["sandbox_root"],
        max_file_size_mb=int(os.environ.get("FILE_MAX_SIZE_MB", "10")),
        allowed_extensions=os.environ.get("FILE_ALLOWED_EXTENSIONS", "").split(",") if os.environ.get("FILE_ALLOWED_EXTENSIONS") else [],
    )
    config["filesystem_config"] = filesystem_config
    
    # Web config
    web_domains = os.environ.get("WEB_ALLOWED_DOMAINS", "*")
    web_config = WebConfig(
        timeout_default=int(os.environ.get("WEB_TIMEOUT_DEFAULT", "30")),
        allowed_domains=None if web_domains == "*" else web_domains.split(","),
    )
    config["web_config"] = web_config
    
    # Knowledge Base config
    kb_storage_path = os.environ.get("CMCP_KB_STORAGE_PATH", os.path.join(config["base_dir"], "kb"))
    kb_config = KBConfig(
        storage_path=kb_storage_path,
        timeout_default=int(os.environ.get("KB_TIMEOUT_DEFAULT", "30")),
        timeout_max=int(os.environ.get("KB_TIMEOUT_MAX", "120")),
    )
    config["kb_config"] = kb_config

    # Matlab config
    matlab_config = MatlabConfig(
        sandbox_dir=os.environ.get("MATLAB_SANDBOX_DIR", config["sandbox_root"]),
        matlab_executable=os.environ.get("MATLAB_EXECUTABLE_PATH", "matlab"),
        memory_limit=int(os.environ.get("MATLAB_MEMORY_LIMIT", "512")),
        timeout_default=int(os.environ.get("MATLAB_TIMEOUT_DEFAULT", "60")),
        timeout_max=int(os.environ.get("MATLAB_TIMEOUT_MAX", "300")),
        default_image_format=os.environ.get("MATLAB_DEFAULT_IMAGE_FORMAT", "png"),
        enabled=os.environ.get("MATLAB_ENABLED", "true").lower() == "true",
    )
    config["matlab_config"] = matlab_config
    
    return config

# Moved MatlabConfig definition here to be before AppConfig if AppConfig needs to directly initialize it.
# However, Pydantic handles forward references ("MatlabConfig") in type hints,
# so the primary concern is logical grouping and ensuring load_env_config can access it.

class MatlabConfig(BaseModel):
    """Configuration for MATLAB Manager.

    Defines settings for controlling MATLAB code execution, including resource limits,
    paths, and feature toggles. These settings can be overridden by environment
    variables (e.g., `MATLAB_SANDBOX_DIR`, `MATLAB_ENABLED`).
    """

    sandbox_dir: str = Field(
        default=os.path.join(BASE_PATHS["sandbox_root"], "matlab"),
        description="Directory for MATLAB sandbox operations. Each execution gets a subdirectory here."
    )
    matlab_executable: str = Field(
        default="matlab",
        description="Path to the MATLAB executable. Can be just 'matlab' if in PATH, or a full path."
    )
    memory_limit: int = Field(
        default=512,
        description="Memory limit for MATLAB execution in MB (enforced by Firejail, approximate)."
    )
    timeout_default: int = Field(
        default=60,
        description="Default timeout for MATLAB execution in seconds."
    )
    timeout_max: int = Field(
        default=300,
        description="Maximum allowed timeout for MATLAB execution in seconds."
    )
    default_image_format: str = Field(
        default="png",
        description="Default format for saving figures (e.g., 'png', 'jpg', 'fig')."
    )
    enabled: bool = Field(
        default=True,
        description="Enable or disable the MATLAB manager and its tools."
    )

# Update AppConfig to initialize matlab_config properly
AppConfig.model_fields['matlab_config'] = Field(default_factory=MatlabConfig)


def load_config() -> AppConfig:
    """Load configuration from environment variables and validate with Pydantic."""
    try:
        env_config = load_env_config()
        # Ensure matlab_config is initialized within AppConfig if not present in env_config
        if "matlab_config" not in env_config:
            # Use the sandbox_root from the potentially overridden env_config
            # or fall back to the default from BASE_PATHS if sandbox_root itself wasn't overridden.
            effective_sandbox_root = env_config.get("sandbox_root", BASE_PATHS["sandbox_root"])
            env_config["matlab_config"] = MatlabConfig(sandbox_dir=os.path.join(effective_sandbox_root, "matlab"))

        config = AppConfig(**env_config)
        
        # Set logging level
        logging.getLogger().setLevel(config.log_level)
        
        logger.debug("Configuration loaded successfully")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        raise