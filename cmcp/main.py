#!/usr/bin/env python3
# cmcp/main.py
# container-mcp © 2025 by Martin Bukowski is licensed under Apache 2.0

"""Main entry point for Container-MCP."""

import os
import sys
import signal
from pathlib import Path
import logging
import asyncio
from typing import Dict, Any, Optional, List

# Load environment file directly
def load_env_file():
    """Load environment variables from .env file."""
    env_file = Path("config/app.env")
    if not env_file.exists():
        raise FileNotFoundError(f"Environment file {env_file} not found")
    
    print(f"Loading environment from {env_file}")
    with open(env_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, value = line.split("=", 1)
                os.environ[key] = value

# Load environment variables BEFORE importing any other modules
load_env_file()

# Override critical paths if we're not in a container
if not (os.path.exists('/.dockerenv') or os.path.exists('/run/.containerenv')):
    # When running locally, use local paths instead of /app
    print("Running in local environment, overriding /app paths")
    current_dir = os.getcwd()
    
    # Overwrite environment paths to use local directories
    os.environ["SANDBOX_ROOT"] = os.path.join(current_dir, "sandbox")
    os.environ["TEMP_DIR"] = os.path.join(current_dir, "temp")
    
    # Create those directories if they don't exist
    os.makedirs(os.environ["SANDBOX_ROOT"], exist_ok=True)
    os.makedirs(os.environ["TEMP_DIR"], exist_ok=True)

from mcp.server.fastmcp import FastMCP

from cmcp.config import load_config
from cmcp.managers.bash_manager import BashManager
from cmcp.managers.python_manager import PythonManager
from cmcp.managers.file_manager import FileManager
from cmcp.managers.web_manager import WebManager
from cmcp.managers.matlab_manager import MatlabManager
from cmcp.utils.logging import setup_logging
from cmcp.tools import register_all_tools

# Configure logging
logger = logging.getLogger(__name__)

# Initialize the MCP server
mcp = FastMCP("Container-MCP")

# Load configuration
config = load_config()

# Initialize managers using from_env pattern
bash_manager = BashManager.from_env(config)
python_manager = PythonManager.from_env(config)
file_manager = FileManager.from_env(config)
web_manager = WebManager.from_env(config)

# Initialize MatlabManager if enabled
matlab_manager = None
if config.matlab_config.enabled:
    try:
        matlab_manager = MatlabManager.from_env(config)
        logger.info("MatlabManager initialized successfully.")
    except EnvironmentError as e:
        logger.warning(f"Failed to initialize MatlabManager: {e}. MATLAB tools will be unavailable.")
    except Exception as e:
        logger.error(f"An unexpected error occurred during MatlabManager initialization: {e}", exc_info=True)
        matlab_manager = None # Ensure it's None if any error occurs

# Set up logging
log_file = os.path.join("logs", "cmcp.log") if os.path.exists("logs") else None
setup_logging(config.log_level, log_file)

# Register all tools
register_all_tools(mcp, bash_manager, python_manager, file_manager, web_manager, matlab_manager)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Parse command line arguments
    test_mode = "--test-mode" in sys.argv
    
    # Make sure environment variables are set (from .env file)
    # For containers, always use port 8000 internally, but bind to specified host
    if os.path.exists('/.dockerenv') or os.path.exists('/run/.containerenv'):
        port = 8000  # Fixed internal port 
        host = os.environ.get("MCP_HOST", "0.0.0.0")  # Default to all interfaces in container
    else:
        # For local development, use the configured port
        port = int(os.environ.get("MCP_PORT", config.mcp_port))
        host = os.environ.get("MCP_HOST", config.mcp_host)
    
    # Allow command-line arguments to override environment settings
    for i, arg in enumerate(sys.argv):
        if arg == "--port" and i+1 < len(sys.argv):
            port = int(sys.argv[i+1])
        elif arg == "--host" and i+1 < len(sys.argv):
            host = sys.argv[i+1]
    
    # Directly set host and port in MCP settings
    mcp.settings.host = host
    mcp.settings.port = port
    
    # Set up signal handlers for graceful shutdown
    def handle_shutdown_signal(signum, frame):
        signal_name = signal.Signals(signum).name
        logger.info(f"Received {signal_name}, shutting down gracefully...")
        
        # Perform cleanup for managers if needed
        all_managers = [bash_manager, python_manager, file_manager, web_manager]
        if matlab_manager:
            all_managers.append(matlab_manager)

        for manager in all_managers:
            if hasattr(manager, 'close'): # Prefer 'close' if available
                try:
                    logger.info(f"Closing {manager.__class__.__name__}")
                    if asyncio.iscoroutinefunction(manager.close):
                        # If in a running loop, use await; if not, use asyncio.run
                        # Signal handlers run in the main thread, may not have a running loop
                        try:
                            loop = asyncio.get_running_loop()
                            # If loop is running, schedule it. This is complex from sync signal handler.
                            # For simplicity here, and assuming close is quick or blocking is acceptable in shutdown:
                            asyncio.run(manager.close())
                        except RuntimeError: # No running event loop
                            asyncio.run(manager.close())
                    else:
                        manager.close()
                except Exception as e:
                    logger.error(f"Error during close of {manager.__class__.__name__}: {e}")
            elif hasattr(manager, 'cleanup'): # Fallback to 'cleanup'
                try:
                    logger.info(f"Cleaning up {manager.__class__.__name__}")
                    manager.cleanup()
                except Exception as e:
                    logger.error(f"Error during cleanup of {manager.__class__.__name__}: {e}")
        
        # Stop the MCP server
        if hasattr(mcp, 'shutdown'):
            logger.info("Shutting down MCP server")
            mcp.shutdown()
        
        # Exit
        sys.exit(0)
    
    # Register signal handlers
    signal.signal(signal.SIGINT, handle_shutdown_signal)  # Ctrl+C
    signal.signal(signal.SIGTERM, handle_shutdown_signal)  # Termination signal
    
    # Run the server with the appropriate transport
    if test_mode:
        logger.info("Starting Container-MCP in test mode with stdio transport")
        mcp.run(transport="stdio")
    else:
        logger.info(f"Container-MCP server running at {host}:{port}")
        mcp.run(transport="sse") 
