# container-mcp © 2025 by Martin Bukowski is licensed under Apache 2.0

"""MATLAB tools for Container-MCP."""

from typing import Optional, Dict, Any
import logging
from pydantic import BaseModel, Field
from mcp.server.fastmcp import FastMCP
from cmcp.managers.matlab_manager import MatlabManager, MatlabResult, MatlabInput
from cmcp.utils.logging import get_logger

logger = get_logger(__name__)

def create_matlab_tools(mcp: FastMCP, matlab_manager: Optional[MatlabManager]) -> None:
    """Create and register MATLAB tools.
    
    Args:
        mcp: The MCP instance
        matlab_manager: The MATLAB manager instance (None if disabled/unavailable)
    """
    if not matlab_manager:
        logger.info("MatlabManager is not available or disabled. MATLAB tools will not be registered.")
        return

    @mcp.tool()
    async def matlab_code_interpreter(
        code: str,
        input_data: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """Execute MATLAB code in a sandboxed environment.
        
        Capabilities:
        - Run MATLAB scripts and perform calculations
        - Input Data: Pass Python variables to MATLAB using the input_data argument
        - Output Data: Return structured data from MATLAB by saving to 'output.mat'
        - Figure Saving: Automatically save generated figures as images
        - Standard I/O: Capture stdout and stderr
        
        Args:
            code: The MATLAB code to execute
            input_data: Dictionary of variables to pre-load into MATLAB workspace
            timeout: Optional timeout in seconds for execution
            
        Returns:
            Dictionary containing execution results, output data, and image paths
        """
        matlab_input_instance: Optional[MatlabInput] = None
        if input_data:
            matlab_input_instance = MatlabInput(variables=input_data)

        result: MatlabResult = await matlab_manager.execute(
            code=code, 
            input_vars=matlab_input_instance, 
            timeout=timeout
        )
        
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.exit_code,
            "output_data": result.output_data,
            "images": result.images
        }

    logger.info("MATLAB code interpreter tool registered.")

__all__ = ["create_matlab_tools"]