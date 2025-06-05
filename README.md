# Container-MCP

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A secure, container-based implementation of the Model Context Protocol (MCP) for executing tools on behalf of large language models.

## Overview

Container-MCP provides a sandboxed environment for safely executing code, running commands, accessing files, and performing web operations requested by large language models. It implements the MCP protocol to expose these capabilities as tools that can be discovered and called by AI systems in a secure manner.

The architecture uses a domain-specific manager pattern with multi-layered security to ensure tools execute in isolated environments with appropriate restrictions, protecting the host system from potentially harmful operations.

## Key Features

- **Multi-layered Security**
  - Container isolation using Podman/Docker
  - AppArmor profiles for restricting access
  - Firejail sandboxing for additional isolation
  - Resource limits (CPU, memory, execution time)
  - Path traversal prevention
  - Allowed extension restrictions

- **MCP Protocol Implementation**
  - Standardized tool discovery and execution
  - Resource management
  - Async execution support

- **Domain-Specific Managers**
  - `BashManager`: Secure command execution
  - `PythonManager`: Sandboxed Python code execution
  - `FileManager`: Safe file operations
  - `WebManager`: Secure web browsing and scraping
  - `MatlabManager`: Secure MATLAB code execution with figure saving

- **Configurable Environment**
  - Extensive configuration via environment variables
  - Custom environment support
  - Development and production modes

## Available Tools

### System Operations

#### `system_run_command`
Executes bash commands in a secure sandbox environment.

- **Parameters**:
  - `command` (string, required): The bash command to execute
  - `working_dir` (string, optional): Working directory (ignored in sandbox)
- **Returns**:
  - `stdout` (string): Command standard output
  - `stderr` (string): Command standard error
  - `exit_code` (integer): Command exit code
  - `success` (boolean): Whether command completed successfully

```json
{
  "stdout": "file1.txt\nfile2.txt\n",
  "stderr": "",
  "exit_code": 0,
  "success": true
}
```

#### `system_run_python`
Executes Python code in a secure sandbox environment.

- **Parameters**:
  - `code` (string, required): Python code to execute
  - `working_dir` (string, optional): Working directory (ignored in sandbox)
- **Returns**:
  - `output` (string): Print output from the code
  - `error` (string): Error output from the code
  - `result` (any): Optional return value (available if code sets `_` variable)
  - `success` (boolean): Whether code executed successfully

```json
{
  "output": "Hello, world!\n",
  "error": "",
  "result": 42,
  "success": true
}
```

#### `system_env_var`
Gets environment variable values.

- **Parameters**:
  - `var_name` (string, optional): Specific variable to retrieve
- **Returns**:
  - `variables` (object): Dictionary of environment variables
  - `requested_var` (string): Value of the requested variable (if var_name provided)

```json
{
  "variables": {
    "MCP_PORT": "8000",
    "SANDBOX_ROOT": "/app/sandbox"
  },
  "requested_var": "8000"
}
```

### File Operations

#### `file_read`
Reads file contents safely.

- **Parameters**:
  - `path` (string, required): Path to the file (relative to sandbox root)
  - `encoding` (string, optional): File encoding (default: "utf-8")
- **Returns**:
  - `content` (string): File content
  - `size` (integer): File size in bytes
  - `modified` (float): Last modified timestamp
  - `success` (boolean): Whether the read was successful

```json
{
  "content": "This is the content of the file.",
  "size": 31,
  "modified": 1673452800.0,
  "success": true
}
```

#### `file_write`
Writes content to a file safely.

- **Parameters**:
  - `path` (string, required): Path to the file (relative to sandbox root)
  - `content` (string, required): Content to write
  - `encoding` (string, optional): File encoding (default: "utf-8")
- **Returns**:
  - `success` (boolean): Whether the write was successful
  - `path` (string): Path to the written file

```json
{
  "success": true,
  "path": "data/myfile.txt"
}
```

#### `file_list`
Lists contents of a directory safely.

- **Parameters**:
  - `path` (string, optional): Path to the directory (default: "/")
  - `pattern` (string, optional): Glob pattern to filter files
- **Returns**:
  - `entries` (array): List of directory entries with metadata
  - `path` (string): The listed directory path
  - `success` (boolean): Whether the listing was successful

```json
{
  "entries": [
    {
      "name": "file1.txt",
      "path": "file1.txt",
      "is_directory": false,
      "size": 1024,
      "modified": 1673452800.0
    },
    {
      "name": "data",
      "path": "data",
      "is_directory": true,
      "size": null,
      "modified": 1673452500.0
    }
  ],
  "path": "/",
  "success": true
}
```

#### `file_delete`
Deletes a file safely.

- **Parameters**:
  - `path` (string, required): Path of the file to delete
- **Returns**:
  - `success` (boolean): Whether the deletion was successful
  - `path` (string): Path to the deleted file

```json
{
  "success": true,
  "path": "temp/old_file.txt"
}
```

#### `file_move`
Moves or renames a file safely.

- **Parameters**:
  - `source` (string, required): Source file path
  - `destination` (string, required): Destination file path
- **Returns**:
  - `success` (boolean): Whether the move was successful
  - `source` (string): Original file path
  - `destination` (string): New file path

```json
{
  "success": true,
  "source": "data/old_name.txt",
  "destination": "data/new_name.txt"
}
```

### Web Operations

#### `web_search`
Uses a search engine to find information on the web.

- **Parameters**:
  - `query` (string, required): The query to search for
- **Returns**:
  - `results` (array): List of search results
  - `query` (string): The original query

```json
{
  "results": [
    {
      "title": "Search Result Title",
      "url": "https://example.com/page1",
      "snippet": "Text snippet from the search result..."
    }
  ],
  "query": "example search query"
}
```

#### `web_scrape`
Scrapes a specific URL and returns the content.

- **Parameters**:
  - `url` (string, required): The URL to scrape
  - `selector` (string, optional): CSS selector to target specific content
- **Returns**:
  - `content` (string): Scraped content
  - `url` (string): The URL that was scraped
  - `title` (string): Page title
  - `success` (boolean): Whether the scrape was successful
  - `error` (string): Error message if scrape failed

```json
{
  "content": "This is the content of the web page...",
  "url": "https://example.com/page",
  "title": "Example Page",
  "success": true,
  "error": null
}
```

#### `web_browse`
Interactively browses a website using Playwright.

- **Parameters**:
  - `url` (string, required): Starting URL for browsing session
- **Returns**:
  - `content` (string): Page HTML content
  - `url` (string): The final URL after any redirects
  - `title` (string): Page title
  - `success` (boolean): Whether the browsing was successful
  - `error` (string): Error message if browsing failed

```json
{
  "content": "<!DOCTYPE html><html>...</html>",
  "url": "https://example.com/after_redirect",
  "title": "Example Page",
  "success": true,
  "error": null
}
```

### MATLAB Code Execution

#### `matlab_code_interpreter`
Executes MATLAB code in a sandboxed environment using `MatlabManager`. It can run scripts, perform calculations, and automatically save figures generated by the code (e.g., using `plot`, `figure`, `surf`). Saved figures are returned as a list of image paths.

- **Parameters**:
  - `code` (string, required): The MATLAB code to execute.
  - `input_data` (dictionary, optional): A dictionary of variables to be pre-loaded into the MATLAB workspace. Keys are variable names, and values are the corresponding data (e.g., numbers, strings, lists/arrays compatible with `scipy.io.savemat`).
    Example:
    ```json
    {
      "input_data": {
        "my_matrix": [[1, 2], [3, 4]],
        "my_string": "hello from Python",
        "my_scalar": 3.14
      }
    }
    ```
    These variables are saved into an `input.mat` file, which is then automatically loaded using `load('input.mat');` at the beginning of your MATLAB script.
  - `timeout` (integer, optional): Optional timeout in seconds for the execution.
- **Returns**:
  - `output` (string): Standard output from the MATLAB code (e.g., from `disp()` commands).
  - `error` (string): Standard error from the MATLAB code.
  - `images` (array of strings): A list of paths to image files generated by the MATLAB code (e.g., from `plot`, `figure`).
  - `output_data` (dictionary, optional): A dictionary containing variables returned from the MATLAB workspace. To make variables available here, your MATLAB script must save them to a file named `output.mat`. For example:
    ```matlab
    % Inside your MATLAB script:
    result_matrix = [10, 20; 30, 40];
    result_string = 'hello from MATLAB';
    save('output.mat', 'result_matrix', 'result_string');
    ```
    The `output_data` field will then contain these variables (e.g., `{"result_matrix": [[10,20],[30,40]], "result_string": "hello from MATLAB"}`).

**Example JSON Response**:
```json
{
  "output": "MATLAB script executed.\nResult processed.\nPlotting complete.\nSaved figure: sandbox/matlab_exec_uuid/figure_handle_1_uuid.png\n",
  "error": "",
  "images": ["sandbox/matlab_exec_uuid/figure_handle_1_uuid.png"],
  "output_data": {
    "result_matrix": [[10, 20], [30, 40]],
    "result_string": "hello from MATLAB"
  }
}
```

## Execution Environment

Container-MCP provides isolated execution environments for different types of operations, each with its own security measures and resource constraints.

### Container Environment

The main Container-MCP service runs inside a container (using Podman or Docker) providing the first layer of isolation:

- **Base Image**: `mathworks/matlab:r2025a`. This provides the MATLAB environment and is based on Ubuntu.
- **User**: A non-root `appuser` is created for running the application.
- **Python**: The environment aims to use Python 3.12, installed via `apt` if not already the default `python3` in the base image. The application runs in a virtual environment (`.venv`).
- **Network**: Limited to localhost binding only for the MCP server. MATLAB execution via Firejail is configured with no network access by default.
- **Filesystem**: Volume mounts for configuration, data, and logs
- **Security**: AppArmor, Seccomp, and capability restrictions

### Bash Execution Environment

The Bash execution environment is configured with multiple isolation layers:

- **Allowed Commands**: Restricted to safe commands configured in `BASH_ALLOWED_COMMANDS`
- **Firejail Sandbox**: Process isolation with restricted filesystem access
- **AppArmor Profile**: Fine-grained access control
- **Resource Limits**:
  - Execution timeout (default: 30s, max: 120s)
  - Limited directory access to sandbox only
- **Network**: No network access
- **File System**: Read-only access to data, read-write to sandbox

Example allowed commands:
```
ls, cat, grep, find, echo, pwd, mkdir, touch
```

### Python Execution Environment

The Python execution environment is designed for secure code execution:

- **Python Version**: The `Containerfile` installs `python3.12` and related development packages. The application's virtual environment is created using this version. The MATLAB base image itself contains a Python version, but CMCP prioritizes its own managed Python environment for consistency.
- **Memory Limit**: Configurable memory ceiling (default: 256MB)
- **Execution Timeout**: Configurable time limit (default: 30s, max: 120s)
- **AppArmor Profile**: Restricts access to system resources
- **Firejail Sandbox**: Process isolation
- **Capabilities**: All capabilities dropped
- **Network**: No network access
- **Available Libraries**: Only standard library
- **Output Capturing**: stdout/stderr redirected and sanitized
- **Resource Controls**: CPU and memory limits enforced

### MATLAB Execution Environment

The MATLAB execution environment is designed for secure code execution and plot generation:

- **MATLAB Version**: Depends on the version pointed to by `MATLAB_EXECUTABLE_PATH`.
- **Memory Limit**: Configurable memory ceiling (default: 512MB via Firejail's `rlimit-as`, though MATLAB's own usage can be complex).
- **Execution Timeout**: Configurable time limit (default: 60s, max: 300s).
- **AppArmor Profile**: Not specifically profiled by default (relies on Firejail and container).
- **Firejail Sandbox**: Process isolation, restricted filesystem access, network disabled by default.
- **Capabilities**: All capabilities dropped via Firejail.
- **Network**: No network access by default (via Firejail).
- **Figure Saving**: Automatically saves figures generated by commands like `plot`, `figure`, `imagesc`, etc., into the execution-specific sandbox directory. The format is configurable (default: `png`).
- **Input/Output Data**: Supports passing data to MATLAB scripts via `input.mat` and retrieving data from `output.mat`. This functionality relies on the `scipy` Python library, which is included in the container.
- **Output Capturing**: `stdout` and `stderr` are captured.
- **Resource Controls**: CPU and memory limits primarily enforced by Firejail.

### File System Environment

The file system environment controls access to files within the sandbox:

- **Base Directory**: All operations restricted to sandbox root
- **Path Validation**: All paths normalized and checked for traversal attempts
- **Size Limits**: Maximum file size enforced (default: 10MB)
- **Extension Control**: Only allowed extensions permitted (default: txt, md, csv, json, py)
- **Permission Control**: Appropriate read/write permissions enforced
- **Isolation**: No access to host file system

### Web Environment

The web environment provides controlled access to external resources:

- **Domain Control**: Optional whitelisting of allowed domains
- **Timeout Control**: Configurable timeouts for operations
- **Browser Control**: Headless browser via Playwright for full rendering
- **Scraping Control**: Simple scraping via requests/BeautifulSoup
- **Content Sanitization**: All content parsed and sanitized
- **Network Isolation**: Separate network namespace via container

## Architecture

The project follows a modular architecture:

```
container-mcp/
├── cmcp/                     # Main application code
│   ├── managers/             # Domain-specific managers
│   │   ├── bash_manager.py   # Secure bash execution
│   │   ├── python_manager.py # Secure python execution
│   │   ├── file_manager.py   # Secure file operations
│   │   └── web_manager.py    # Secure web operations
│   ├── utils/                # Utility functions
│   ├── config.py             # Configuration system
│   └── main.py               # MCP server setup
├── apparmor/                 # AppArmor profiles
├── config/                   # Configuration files
├── bin/                      # Build/run scripts
├── data/                     # Data directory
├── logs/                     # Log directory
├── sandbox/                  # Sandboxed execution space
│   ├── bash/                 # Bash sandbox
│   ├── python/               # Python sandbox
│   ├── files/                # File operation sandbox
│   └── browser/              # Web browser sandbox
├── temp/                     # Temporary storage
└── tests/                    # Test suites
```

Each manager follows consistent design patterns:
- `.from_env()` class method for environment-based initialization
- Async execution methods for non-blocking operations
- Strong input validation and error handling
- Security-first approach to all operations

## Security Measures

Container-MCP implements multiple layers of security:

1. **Container Isolation**: Uses Podman/Docker for container isolation
2. **AppArmor Profiles**: Fine-grained access control for bash and Python execution
3. **Firejail Sandboxing**: Additional process isolation
4. **Resource Limits**: Memory, CPU, and execution time limits
5. **Path Traversal Prevention**: Validates and normalizes all file paths
6. **Allowed Extension Restrictions**: Controls what file types can be accessed
7. **Network Restrictions**: Controls what domains can be accessed
8. **Least Privilege**: Components run with minimal necessary permissions

## Installation

### Prerequisites

- Linux system with Podman or Docker
- Firejail (`apt install firejail` or `dnf install firejail`)
- AppArmor (`apt install apparmor apparmor-utils` or `dnf install apparmor apparmor-utils`)
- An active MATLAB license if you intend to use MATLAB execution features.

**Note on Base Image and Build Process**: The `Containerfile` now uses `mathworks/matlab:r2025a` as its base. This means:
- The initial download of the base image will be significantly larger.
- Build times might increase.
- You must have appropriate access to pull the `mathworks/matlab` image (e.g., logged into a Docker Hub account that has access if it's a private or licensed image, or if MathWorks policies require it).

### Quick Start

The quickest way to get started is to use the all-in-one script:

```bash
git clone https://github.com/54rt1n/container-mcp.git
cd container-mcp
chmod +x bin/00-all-in-one.sh
./bin/00-all-in-one.sh
```

### Step-by-Step Installation

You can also perform the installation steps individually:

1. **Initialize the project**:
   ```bash
   ./bin/01-init.sh
   ```

2. **Build the container**:
   ```bash
   ./bin/02-build-container.sh
   ```

3. **Set up the environment**:
   ```bash
   ./bin/03-setup-environment.sh
   ```

4. **Run the container**:
   ```bash
   ./bin/04-run-container.sh
   ```

5. **Run tests** (optional):
   ```bash
   ./bin/05-run-tests.sh
   ```

## Usage

Once the container is running, you can connect to it using any MCP client implementation. The server will be available at `http://localhost:8000` or the port specified in your configuration.

**Important:** When configuring your MCP client, you must set the endpoint URL to `http://127.0.0.1:<port>/sse` (where `<port>` is 8000 by default or the port you've configured). The `/sse` path is required for proper server-sent events communication.

### Example Python Client

```python
from mcp.client.sse import sse_client
from mcp import ClientSession
import asyncio

async def main():
    # Connect to the Container-MCP server
    # Note the /sse endpoint suffix required for SSE communication
    sse_url = "http://127.0.0.1:8000/sse"  # Or your configured port
    
    # Connect to the SSE endpoint
    async with sse_client(sse_url) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()
            
            # Discover available tools
            result = await session.list_tools()
            print(f"Available tools: {[tool.name for tool in result.tools]}")
            
            # Execute a Python script
            python_result = await session.execute_tool(
                "system_run_python",
                {"code": "print('Hello, world!')\nresult = 42\n_ = result"}
            )
            print(f"Python result: {python_result}")
            
            # Execute a bash command
            bash_result = await session.execute_tool(
                "system_run_command",
                {"command": "ls -la"}
            )
            print(f"Command output: {bash_result['stdout']}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Configuration

Container-MCP can be configured through environment variables, which can be set in `volume/config/custom.env`:

### Server Configuration

```
# MCP Server Configuration
MCP_HOST=127.0.0.1
MCP_PORT=9001
DEBUG=true
LOG_LEVEL=INFO
```

### Bash Manager Configuration

```
# Bash Manager Configuration
BASH_ALLOWED_COMMANDS=ls,cat,grep,find,echo,pwd,mkdir,touch
BASH_TIMEOUT_DEFAULT=30
BASH_TIMEOUT_MAX=120
```

### Python Manager Configuration

```
# Python Manager Configuration
PYTHON_MEMORY_LIMIT=256
PYTHON_TIMEOUT_DEFAULT=30
PYTHON_TIMEOUT_MAX=120
```

### File Manager Configuration

```
# File Manager Configuration 
FILE_MAX_SIZE_MB=10
FILE_ALLOWED_EXTENSIONS=txt,md,csv,json,py
```

### Web Manager Configuration

```
# Web Manager Configuration
WEB_TIMEOUT_DEFAULT=30
WEB_ALLOWED_DOMAINS=*
```

### MATLAB Manager Configuration

```
# MATLAB Manager Configuration
MATLAB_ENABLED=true
MATLAB_SANDBOX_DIR=/app/sandbox/matlab  # Or your local equivalent like ./sandbox/matlab
MATLAB_EXECUTABLE_PATH=matlab           # Default, or specify full path e.g., /usr/local/MATLAB/R2023b/bin/matlab
MATLAB_MEMORY_LIMIT=512                 # In MB, approximate limit via Firejail
MATLAB_TIMEOUT_DEFAULT=60               # In seconds
MATLAB_TIMEOUT_MAX=300                  # In seconds
MATLAB_DEFAULT_IMAGE_FORMAT=png         # E.g., png, jpg, fig

#### MATLAB Licensing Requirements
The `mathworks/matlab:r2025a` base image used by Container-MCP requires a valid MATLAB license to run. When you run the Container-MCP container, you must configure it to access your MATLAB license. This typically involves setting an environment variable.

- **Network License Manager (FlexLM)**: If your organization uses a network license server, you'll likely need to set `MLM_LICENSE_FILE`.
  ```bash
  # Example when running the container:
  podman run -d -p 8000:8000 \
    -e MLM_LICENSE_FILE=your_port@your_license_server_address \
    your_container_mcp_image_name
  ```

- **Designated Computer / Individual License File**: If you have a license file, you might need to mount it into the container and set `LM_LICENSE_FILE` (or a similar variable) to its path within the container.
  ```bash
  # Example when running the container:
  podman run -d -p 8000:8000 \
    -v /path/to/your/license.lic:/opt/matlab_licenses/license.lic \
    -e LM_LICENSE_FILE=/opt/matlab_licenses/license.lic \
    your_container_mcp_image_name
  ```

**Important**: The specific environment variable name (`MLM_LICENSE_FILE`, `LM_LICENSE_FILE`, etc.) and the value format depend on your MATLAB license type and configuration. Consult your MATLAB license administrator or MathWorks documentation for the precise details applicable to your situation. Without proper licensing, MATLAB execution will fail.
```

## Development

### Setting Up a Development Environment

1. Create a Python virtual environment:
   ```bash
   python3.12 -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements-dev.txt
   ```

3. Install the package in development mode:
   ```bash
   pip install -e .
   ```

### Running Tests

```bash
# Run all tests
pytest

# Run only unit tests
pytest tests/unit

# Run only integration tests
pytest tests/integration

# Run with coverage report
pytest --cov=cmcp --cov-report=term --cov-report=html
```

### Development Server

To run the MCP server in development mode:

```bash
python -m cmcp.main --test-mode
```

## License

This project is licensed under the Apache License 2.0.

## Author

Martin Bukowski
