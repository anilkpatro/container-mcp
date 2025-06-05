"""Tools package for Container-MCP.

This package contains tools for various operations like system commands,
file operations, web access, and knowledge base management.
"""

from .system import create_system_tools
from .file import create_file_tools
from .web import create_web_tools
from .kb import create_kb_tools
from .matlab import create_matlab_tools
from cmcp.managers.matlab_manager import MatlabManager # For type hinting
from typing import Optional # For type hinting

def register_all_tools(
    mcp,
    bash_manager,
    python_manager,
    file_manager,
    web_manager,
    matlab_manager: Optional[MatlabManager] = None
):
    """Register all tools with the MCP instance.
    
    Args:
        mcp: The MCP instance
        bash_manager: The bash manager instance
        python_manager: The python manager instance
        file_manager: The file manager instance
        web_manager: The web manager instance
        matlab_manager: The MATLAB manager instance (optional)
    """
    create_system_tools(mcp, bash_manager, python_manager)
    create_file_tools(mcp, file_manager)
    create_web_tools(mcp, web_manager)
    create_kb_tools(mcp)
    if matlab_manager:
        create_matlab_tools(mcp, matlab_manager)