"""Tools package for Container-MCP.

This package contains tools for various operations like system commands,
file operations, web access, and knowledge base management.
"""

import logging
from typing import Any

# Import tool creation functions
from .system import create_system_tools
from .file import create_file_tools
from .web import create_web_tools
from .kb import create_kb_tools
from .matlab import create_matlab_tools
from cmcp.managers.matlab_manager import MatlabManager # For type hinting
from typing import Optional # For type hinting

# Configure logging
logger = logging.getLogger(__name__)

def register_all_tools(
    mcp,
    config,
    bash_manager,
    python_manager,
    file_manager,
    web_manager,
    kb_manager,
    matlab_manager: Optional[MatlabManager] = None
):
    """Register all tools with the MCP instance.
    
    Args:
        mcp: The MCP instance
        config: The application configuration
        bash_manager: The bash manager instance
        python_manager: The python manager instance
        file_manager: The file manager instance
        web_manager: The web manager instance
        kb_manager: The knowledge base manager instance
    """
    logger.info("Registering tools based on configuration...")
    
    # System Tools (Bash, Python)
    if config.tools_enable_system:
        create_system_tools(mcp, bash_manager, python_manager)
        logger.info("System tools (bash, python) ENABLED.")
    else:
        logger.warning("System tools (bash, python) DISABLED by configuration.")
    
    # File Tools
    if config.tools_enable_file:
        create_file_tools(mcp, file_manager)
        logger.info("File tools ENABLED.")
    else:
        logger.warning("File tools DISABLED by configuration.")
    
    # Web Tools
    if config.tools_enable_web:
        create_web_tools(mcp, web_manager)
        logger.info("Web tools (search, scrape, browse) ENABLED.")
    else:
        logger.warning("Web tools (search, scrape, browse) DISABLED by configuration.")
    
    # Knowledge Base Tools
    if config.tools_enable_kb:
        create_kb_tools(mcp, kb_manager)
        logger.info("Knowledge Base tools ENABLED.")
    else:
        logger.warning("Knowledge Base tools DISABLED by configuration.")

    # MATLAB Tools
    if config.tools_enable_matlab:
        if matlab_manager:
            create_matlab_tools(mcp, matlab_manager)
            logger.info("MATLAB tools ENABLED.")
    else:
        logger.warning("MATLAB tools DISABLED by configuration.")
    
    logger.info("Tool registration complete.") 
