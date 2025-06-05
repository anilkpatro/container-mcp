# tests/unit/test_matlab_manager.py

import asyncio
import os
import shutil
import tempfile
import uuid
from unittest.mock import patch, AsyncMock, MagicMock, call, PropertyMock
import numpy as np # For creating sample data for .mat files

import pytest
# Conditional import for type hinting if scipy is part of the test environment
try:
    from scipy.io.matlab.miobase import MatReadError
except ImportError:
    MatReadError = Exception # Fallback if scipy not installed in test env

from cmcp.managers.matlab_manager import MatlabManager, MatlabResult, MatlabInput # Import MatlabInput
from cmcp.config import AppConfig, MatlabConfig


# Mark all tests in this file as asyncio
pytestmark = pytest.mark.asyncio


@pytest.fixture
def sandbox_dir():
    """Create a temporary sandbox directory for testing."""
    temp_dir = tempfile.mkdtemp(prefix="cmcp_test_matlab_sandbox_")
    # Ensure the 'matlab' subdirectory exists if config points there by default
    os.makedirs(os.path.join(temp_dir, "matlab"), exist_ok=True)
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_matlab_config_enabled(sandbox_dir):
    """Provides a mock MatlabConfig instance with MATLAB enabled."""
    return MatlabConfig(
        sandbox_dir=os.path.join(sandbox_dir, "matlab"), # Consistent with default in config.py
        matlab_executable="mock_matlab",
        memory_limit=512,
        timeout_default=30,
        timeout_max=120,
        default_image_format="png",
        enabled=True,
    )

@pytest.fixture
def mock_matlab_config_disabled(sandbox_dir):
    """Provides a mock MatlabConfig instance with MATLAB disabled."""
    return MatlabConfig(
        sandbox_dir=os.path.join(sandbox_dir, "matlab"),
        matlab_executable="mock_matlab",
        enabled=False,
    )

@pytest.fixture
def mock_app_config_matlab_enabled(mock_matlab_config_enabled):
    """Provides a mock AppConfig with MatlabConfig enabled."""
    app_cfg = AppConfig() # Loads default, we override matlab_config
    app_cfg.matlab_config = mock_matlab_config_enabled
    return app_cfg

@pytest.fixture
def mock_app_config_matlab_disabled(mock_matlab_config_disabled):
    """Provides a mock AppConfig with MatlabConfig disabled."""
    app_cfg = AppConfig()
    app_cfg.matlab_config = mock_matlab_config_disabled
    return app_cfg


# --- Test Cases ---

async def test_matlab_manager_initialization(sandbox_dir):
    """Test MatlabManager initializes correctly."""
    manager = MatlabManager(
        sandbox_dir=sandbox_dir,
        matlab_executable="test_matlab_exe",
        memory_limit=1024,
        timeout_default=60,
        timeout_max=300,
        default_image_format="jpg",
    )
    assert manager.sandbox_dir == sandbox_dir
    assert manager.matlab_executable == "test_matlab_exe"
    assert manager.memory_limit == 1024
    assert manager.timeout_default == 60
    assert manager.timeout_max == 300
    assert manager.default_image_format == "jpg"
    assert os.path.exists(sandbox_dir) # Check sandbox dir is created/accessible

async def test_matlab_manager_from_env_enabled(mock_app_config_matlab_enabled):
    """Test MatlabManager.from_env() when MATLAB is enabled."""
    with patch('cmcp.managers.matlab_manager.load_config', return_value=mock_app_config_matlab_enabled):
        manager = MatlabManager.from_env(app_config=mock_app_config_matlab_enabled)

    cfg = mock_app_config_matlab_enabled.matlab_config
    assert manager.sandbox_dir == cfg.sandbox_dir
    assert manager.matlab_executable == cfg.matlab_executable
    assert manager.memory_limit == cfg.memory_limit
    assert manager.timeout_default == cfg.timeout_default
    assert manager.timeout_max == cfg.timeout_max
    assert manager.default_image_format == cfg.default_image_format

async def test_matlab_manager_from_env_disabled(mock_app_config_matlab_disabled):
    """Test MatlabManager.from_env() when MATLAB is disabled in config."""
    with patch('cmcp.managers.matlab_manager.load_config', return_value=mock_app_config_matlab_disabled):
        with pytest.raises(EnvironmentError, match="MatlabManager is disabled"):
            MatlabManager.from_env(app_config=mock_app_config_matlab_disabled)


@patch('shutil.which', return_value="mock_firejail_path") # Assume firejail is available
async def test_execute_simple_code(mock_shutil_which, mock_app_config_matlab_enabled):
    """Test execution of a simple MATLAB command."""
    manager = MatlabManager.from_env(app_config=mock_app_config_matlab_enabled)
    mock_matlab_stdout = "Hello, MATLAB!"
    mock_matlab_stderr = ""

    mock_proc = AsyncMock()
    mock_proc.communicate = AsyncMock(return_value=(mock_matlab_stdout.encode(), mock_matlab_stderr.encode()))
    mock_proc.returncode = 0

    with patch('asyncio.create_subprocess_exec', return_value=mock_proc) as mock_create_subprocess_exec:
        result = await manager.execute("disp('Hello, MATLAB!');")

    assert result.output == mock_matlab_stdout
    assert result.error == mock_matlab_stderr
    assert not result.images

    # Check that the subprocess was called with expected arguments (basic check)
    # The actual command will be complex due to temp files and firejail
    assert mock_create_subprocess_exec.called
    call_args, _ = mock_create_subprocess_exec.call_args
    # The first element of call_args is usually the command itself if a list of strings
    # For *args, it's a tuple of all positional arguments.
    # Here, args[0] would be the first argument to create_subprocess_exec, which is the first part of the command.
    assert "mock_firejail_path" in call_args[0] # firejail should be part of the command
    assert manager.matlab_executable in call_args # matlab executable should be there
    assert "-batch" in call_args # batch mode for matlab
    assert result.output_data is None # No output.mat handling in this test


@patch('shutil.which', return_value="mock_firejail_path")
async def test_execute_code_with_error(mock_shutil_which, mock_app_config_matlab_enabled):
    """Test execution of MATLAB code that generates an error."""
    manager = MatlabManager.from_env(app_config=mock_app_config_matlab_enabled)
    mock_matlab_stdout = ""
    mock_matlab_stderr = "Error: Undefined function 'foo' for input arguments of type 'double'."

    mock_proc = AsyncMock()
    mock_proc.communicate = AsyncMock(return_value=(mock_matlab_stdout.encode(), mock_matlab_stderr.encode()))
    mock_proc.returncode = 1 # MATLAB typically exits with non-zero on error

    with patch('asyncio.create_subprocess_exec', return_value=mock_proc):
        result = await manager.execute("foo(123);")

    # Output might contain some MATLAB startup messages even on error, but stderr is key
    # For this test, assuming stderr is precisely what MATLAB produced.
    assert result.error.strip() == mock_matlab_stderr
    assert not result.images
    assert result.output_data is None


@patch('shutil.which', return_value="mock_firejail_path") # firejail available
async def test_execute_figure_saving(mock_shutil_which, mock_app_config_matlab_enabled, sandbox_dir):
    """Test execution of MATLAB code that generates a plot and saves it."""
    manager = MatlabManager.from_env(app_config=mock_app_config_matlab_enabled)

    # The manager creates an execution_dir like "matlab_exec_UUID"
    # We need to know this to simulate file creation by the mocked subprocess
    # and to have listdir find it.

    mock_matlab_stdout = "Plotting complete.\nSaved figure: /path/to/figure_handle_1_someuuid.png" # Example output
    mock_matlab_stderr = ""

    mock_proc = AsyncMock()
    mock_proc.communicate = AsyncMock(return_value=(mock_matlab_stdout.encode(), mock_matlab_stderr.encode()))
    mock_proc.returncode = 0

    # Simulate that the MATLAB script (run by the mocked subprocess) creates an image file
    # This requires knowing the execution_dir that the manager will create.
    # We can patch uuid.uuid4 to control the generated names for predictability.

    fixed_uuid_str_exec_dir = "fixedexecdiruuid"
    fixed_uuid_exec_dir = uuid.UUID(fixed_uuid_str_exec_dir)

    fixed_uuid_str_fig = "fixedfigureuuid0"

    # The manager's execute method will create a directory like:
    # sandbox_dir/matlab_exec_fixedexecdiruuid/
    # And the image file will be like:
    # sandbox_dir/matlab_exec_fixedexecdiruuid/figure_handle_1_fixedfigureuuid0.png

    expected_exec_dir_name = f"matlab_exec_{fixed_uuid_str_exec_dir}"
    # The config sandbox_dir is sandbox_dir/matlab, so the full path is:
    full_execution_dir = os.path.join(manager.sandbox_dir, expected_exec_dir_name)

    # Ensure this directory is made by the manager's execute method before listdir is called
    # The manager itself calls os.makedirs(execution_dir, exist_ok=True)

    # The figure saving logic in MatlabManager generates filenames like:
    # figure_handle_{fig.Number}_{uuid.uuid4().hex}.{self.default_image_format}
    # figure_current_{uuid.uuid4().hex}.{self.default_image_format}
    # We'll mock the second uuid.uuid4() call inside figure_saving_code

    # Mock os.listdir to return our simulated image file
    # Mock os.path.exists for the image file as well if the manager checks

    simulated_image_name = f"figure_handle_1_{fixed_uuid_str_fig}.{manager.default_image_format}"
    expected_image_path = os.path.join(full_execution_dir, simulated_image_name)

    def mock_listdir_side_effect(path):
        if path == full_execution_dir:
            return [simulated_image_name]
        return []

    # We need two different UUIDs: one for exec_dir, one for figure name
    mock_uuid_patcher = patch('uuid.uuid4', side_effect=[
        uuid.UUID(fixed_uuid_str_exec_dir), # For execution_dir_name
        uuid.UUID(fixed_uuid_str_fig),      # For the first figure name in figure_saving_code
        uuid.UUID("anotheruuid1"),          # For the second figure name (currentFig)
        uuid.UUID("anotheruuid2"),          # Potentially more if the user code also uses uuid.
        uuid.UUID("anotheruuid3"),
    ])


    with mock_uuid_patcher as mock_uuid_gen, \
         patch('asyncio.create_subprocess_exec', return_value=mock_proc) as mock_create_subprocess, \
         patch('os.listdir', side_effect=mock_listdir_side_effect) as mock_os_listdir:

        # Critical: The manager's execute method itself creates full_execution_dir.
        # We need to ensure our mock_os_listdir is prepared for that path.
        # We also need to simulate the image file being "created" by the MATLAB script.
        # The simplest way is to ensure os.listdir returns it for the correct path.

        result = await manager.execute("plot(1:10);")

    assert result.output == mock_matlab_stdout
    assert result.error == mock_matlab_stderr
    assert len(result.images) == 1
    assert result.images[0] == expected_image_path

    # Check that listdir was called on the correct execution directory
    mock_os_listdir.assert_any_call(full_execution_dir)

    # Check that the script was written into the execution dir (name is now fixed as run_script.m)
    script_path_pattern = os.path.join(full_execution_dir, "run_script.m")

    # Check that the subprocess was called with the script path from the correct exec_dir
    found_script_arg = False
    call_args_list = mock_create_subprocess.call_args[0] # Get the args tuple
    for arg in call_args_list:
        if isinstance(arg, str) and script_path_pattern in arg: # script_path_pattern is part of the -batch "run(...)"
            found_script_arg = True
            break
    assert found_script_arg, f"Script path {script_path_pattern} not found in subprocess args: {call_args_list}"
    assert result.output_data is None


@patch('shutil.which', return_value="mock_firejail_path")
async def test_execute_timeout(mock_shutil_which, mock_app_config_matlab_enabled):
    """Test if MATLAB execution times out correctly."""
    manager = MatlabManager.from_env(app_config=mock_app_config_matlab_enabled)

    # Mock create_subprocess_exec to simulate a timeout
    mock_create_subprocess_exec = AsyncMock(side_effect=asyncio.TimeoutError)
    # If create_subprocess_exec itself raises TimeoutError, communicate won't be called.
    # More realistically, proc.communicate() is what times out.

    mock_proc = AsyncMock()
    mock_proc.communicate = AsyncMock(side_effect=asyncio.TimeoutError)
    mock_proc.kill = MagicMock() # Ensure kill is called
    mock_proc.returncode = None # Process hasn't exited before kill

    with patch('asyncio.create_subprocess_exec', return_value=mock_proc) as mock_async_exec:
        result = await manager.execute("while true, end", timeout=1) # Short timeout for testing

    assert "Execution timed out after 1 seconds." in result.error
    assert result.output == ""
    assert not result.images
    assert result.output_data is None
    mock_proc.kill.assert_called_once()


# Conceptual test for sandbox command structure
@patch('shutil.which') # Mock shutil.which for firejail and matlab
async def test_sandbox_command_structure(mock_shutil_which_multiple, mock_app_config_matlab_enabled):
    """Check the structure of the firejail command."""

    # Side effect for shutil.which: first call for firejail, second for matlab
    def which_side_effect(cmd):
        if cmd == "firejail":
            return "/usr/bin/firejail"
        if cmd == mock_app_config_matlab_enabled.matlab_config.matlab_executable:
            return f"/opt/MATLAB/{mock_app_config_matlab_enabled.matlab_config.matlab_executable}"
        return None

    mock_shutil_which_multiple.side_effect = which_side_effect

    manager = MatlabManager.from_env(app_config=mock_app_config_matlab_enabled)

    # Path to the execution directory within the sandbox
    # Manager creates sandbox_dir/matlab_exec_UUID.
    # We need a predictable UUID for this test.
    fixed_uuid_str = "testexecdirforsandbox"
    execution_dir_name = f"matlab_exec_{fixed_uuid_str}"
    # The manager's sandbox_dir is from config: sandbox_dir/matlab
    full_execution_dir_sandbox_view = os.path.join(manager.sandbox_dir, execution_dir_name)

    # The manager writes the script to full_execution_dir_sandbox_view/run_script.m
    temp_script_name = "run_script.m"
    matlab_script_path_arg = os.path.join(full_execution_dir_sandbox_view, temp_script_name)

    # We need to control the UUID for the execution_dir_name
    with patch('uuid.uuid4', return_value=uuid.UUID(fixed_uuid_str)):
        # Call _get_sandbox_command indirectly by trying to execute
        # We need to mock asyncio.create_subprocess_exec to prevent actual execution
        # Also mock scipy to prevent issues if test environment doesn't have it
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"",b""))
        mock_proc.returncode = 0
        with patch('cmcp.managers.matlab_manager.scipy', MagicMock()) as mock_scipy_module, \
             patch('asyncio.create_subprocess_exec', return_value=mock_proc) as mock_create_subprocess:
            await manager.execute("disp(1);")

    assert mock_create_subprocess.called
    call_args_list = mock_create_subprocess.call_args[0]

    assert call_args_list[0] == "/usr/bin/firejail"
    assert f"--private={manager.sandbox_dir}" in call_args_list
    assert f"--whitelist={full_execution_dir_sandbox_view}" in call_args_list
    # The matlab executable itself is one of the last arguments, before -nodesktop, -nosplash, -batch
    assert f"/opt/MATLAB/{manager.matlab_executable}" in call_args_list

    # Check for the MATLAB batch command structure
    expected_batch_command_part = f"run('{matlab_script_path_arg}')"
    found_batch_command = False
    for arg in call_args_list: # Iterate through the actual command list passed to subprocess
        if expected_batch_command_part in arg:
            found_batch_command = True
            break
    assert found_batch_command, f"Expected batch command part '{expected_batch_command_part}' not found in {call_args_list}"

# Further tests could include:
# - test_matlab_not_found (mock shutil.which for matlab to return None)
# - test_firejail_not_available (mock shutil.which for firejail to return None)
# - test_execute_multiple_figures (more complex mocking of listdir and uuid)
# - test behavior when script writing fails (e.g. disk full, permissions)
# - test cleanup of execution directory (if implemented, currently it's kept)

async def test_firejail_not_available(mock_app_config_matlab_enabled):
    """Test that the command runs without firejail if it's not available."""

    def which_side_effect(cmd):
        if cmd == "firejail":
            return None # Firejail not found
        if cmd == mock_app_config_matlab_enabled.matlab_config.matlab_executable:
            return f"/mock/path/to/{mock_app_config_matlab_enabled.matlab_config.matlab_executable}"
        return None

    with patch('shutil.which', side_effect=which_side_effect) as mock_shutil_which_call:
        manager = MatlabManager.from_env(app_config=mock_app_config_matlab_enabled)

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"output", b""))
        mock_proc.returncode = 0
        with patch('asyncio.create_subprocess_exec', return_value=mock_proc) as mock_create_subprocess:
            await manager.execute("disp(1);")

    assert mock_create_subprocess.called
    cmd_args_list = mock_create_subprocess.call_args[0]
    assert "firejail" not in cmd_args_list[0] # First arg should be matlab executable
    assert f"/mock/path/to/{manager.matlab_executable}" == cmd_args_list[0]
    # Ensure it logged a warning
    # (Requires capturing logs or checking logger calls if using a mock logger)


async def test_matlab_executable_not_found(mock_app_config_matlab_enabled):
    """Test behavior if MATLAB executable is not found (conceptual via _get_sandbox_command)."""

    # This test is a bit tricky because shutil.which is usually for `firejail` in the current code.
    # The matlab_executable path is taken from config.
    # If this path itself is invalid, asyncio.create_subprocess_exec would raise FileNotFoundError.

    # To truly test this, we'd need create_subprocess_exec to raise FileNotFoundError
    # when trying to run the configured matlab_executable.

    manager = MatlabManager.from_env(app_config=mock_app_config_matlab_enabled)
    manager.matlab_executable = "/invalid/path/to/matlab" # Override with a clearly invalid path

    mock_create_subprocess_exec = AsyncMock(side_effect=FileNotFoundError(f"[Errno 2] No such file or directory: '{manager.matlab_executable}'"))

    with patch('shutil.which', return_value="firejail"), \
         patch('asyncio.create_subprocess_exec', mock_create_subprocess_exec):
        result = await manager.execute("disp(1)")

    assert f"No such file or directory: '{manager.matlab_executable}'" in result.error
    assert result.output == ""
    assert result.output_data is None

# Test for multiple figure saving
@patch('shutil.which', return_value="mock_firejail_path") # firejail available
async def test_execute_multiple_figures(mock_shutil_which, mock_app_config_matlab_enabled, sandbox_dir):
    manager = MatlabManager.from_env(app_config=mock_app_config_matlab_enabled)

    mock_matlab_stdout = (
        "Plotting complete.\n"
        "Saved figure: /path/to/figure_handle_1_uuid1.png\n"
        "Saved figure: /path/to/figure_handle_2_uuid2.png\n"
        "Saved current figure: /path/to/figure_current_uuid3.png"
    )
    mock_matlab_stderr = ""

    mock_proc = AsyncMock()
    mock_proc.communicate = AsyncMock(return_value=(mock_matlab_stdout.encode(), mock_matlab_stderr.encode()))
    mock_proc.returncode = 0

    # Setup for predictable UUIDs and paths
    exec_dir_uuid = uuid.uuid4()
    fig_uuid1 = uuid.uuid4()
    fig_uuid2 = uuid.uuid4()
    fig_uuid3 = uuid.uuid4() # For the 'current' figure

    expected_exec_dir_name = f"matlab_exec_{exec_dir_uuid.hex}"
    full_execution_dir = os.path.join(manager.sandbox_dir, expected_exec_dir_name)

    simulated_image_name1 = f"figure_handle_1_{fig_uuid1.hex}.{manager.default_image_format}"
    simulated_image_name2 = f"figure_handle_2_{fig_uuid2.hex}.{manager.default_image_format}"
    # The "current" figure name generation in matlab_manager.py is:
    # ['figure_current_{uuid.uuid4().hex}.{self.default_image_format}']
    # So it will use the *next* UUID from the side_effect list for this.
    simulated_image_name_current = f"figure_current_{fig_uuid3.hex}.{manager.default_image_format}"

    expected_image_path1 = os.path.join(full_execution_dir, simulated_image_name1)
    expected_image_path2 = os.path.join(full_execution_dir, simulated_image_name2)
    expected_image_path_current = os.path.join(full_execution_dir, simulated_image_name_current)

    def mock_listdir_side_effect(path):
        if path == full_execution_dir:
            return [simulated_image_name1, simulated_image_name2, simulated_image_name_current]
        return []

    # Order of UUID generation in MatlabManager:
    # 1. For execution_dir_name (in `execute`)
    # 2. For each handle-based figure name (in `figure_saving_code` loop) -> fig_uuid1, fig_uuid2
    # 3. For current figure name (in `figure_saving_code` try block for gcf) -> fig_uuid3
    # (Additional UUIDs might be consumed by user code if it also uses uuid.uuid4(), so provide extras)
    uuid_side_effects = [
        exec_dir_uuid,  # For execution_dir_name
        fig_uuid1,      # For figure_handle_1
        fig_uuid2,      # For figure_handle_2
        fig_uuid3,      # For figure_current
        uuid.uuid4(),   # Extra
        uuid.uuid4()    # Extra
    ]

    with patch('uuid.uuid4', side_effect=uuid_side_effects), \
         patch('asyncio.create_subprocess_exec', return_value=mock_proc), \
         patch('os.listdir', side_effect=mock_listdir_side_effect) as mock_os_listdir:

        result = await manager.execute("figure; plot(1:10); figure; plot(rand(5));")

    assert result.output == mock_matlab_stdout
    assert result.error == mock_matlab_stderr
    assert len(result.images) == 3
    assert expected_image_path1 in result.images
    assert expected_image_path2 in result.images
    assert expected_image_path_current in result.images

    mock_os_listdir.assert_any_call(full_execution_dir)

    # Check script path in subprocess call
    script_path_pattern = os.path.join(full_execution_dir, "run_script.m") # Fixed script name
    # mock_create_subprocess is the mock for asyncio.create_subprocess_exec
    args_list = mock_create_subprocess.call_args[0]
    found_script_arg = any(script_path_pattern in arg for arg in args_list if isinstance(arg, str))
    assert found_script_arg, f"Script path {script_path_pattern} not found in subprocess args: {args_list}"
    assert result.output_data is None

# To make the last assert work for script_path_pattern in the multiple figures test:
# The mock_proc is asyncio.create_subprocess_exec itself.
# So it should be mock_proc.call_args[0] if mock_proc is the direct mock of create_subprocess_exec.
# Let's adjust that:
@patch('shutil.which', return_value="mock_firejail_path") # firejail available
async def test_execute_multiple_figures_corrected(mock_shutil_which, mock_app_config_matlab_enabled, sandbox_dir):
    manager = MatlabManager.from_env(app_config=mock_app_config_matlab_enabled)

    mock_matlab_stdout = (
        "Plotting complete.\n"
        "Saved figure: /path/to/figure_handle_1_uuid1.png\n"
        "Saved figure: /path/to/figure_handle_2_uuid2.png\n"
        "Saved current figure: /path/to/figure_current_uuid3.png"
    )
    mock_matlab_stderr = ""

    mock_async_proc_instance = AsyncMock() # This is the Process object
    mock_async_proc_instance.communicate = AsyncMock(return_value=(mock_matlab_stdout.encode(), mock_matlab_stderr.encode()))
    mock_async_proc_instance.returncode = 0

    # Setup for predictable UUIDs and paths
    exec_dir_uuid = uuid.uuid4()
    fig_uuid1 = uuid.uuid4()
    fig_uuid2 = uuid.uuid4()
    fig_uuid3 = uuid.uuid4()

    expected_exec_dir_name = f"matlab_exec_{exec_dir_uuid.hex}"
    full_execution_dir = os.path.join(manager.sandbox_dir, expected_exec_dir_name)

    simulated_image_name1 = f"figure_handle_1_{fig_uuid1.hex}.{manager.default_image_format}"
    simulated_image_name2 = f"figure_handle_2_{fig_uuid2.hex}.{manager.default_image_format}"
    simulated_image_name_current = f"figure_current_{fig_uuid3.hex}.{manager.default_image_format}"

    expected_image_path1 = os.path.join(full_execution_dir, simulated_image_name1)
    expected_image_path2 = os.path.join(full_execution_dir, simulated_image_name2)
    expected_image_path_current = os.path.join(full_execution_dir, simulated_image_name_current)

    def mock_listdir_side_effect(path):
        if path == full_execution_dir:
            return [simulated_image_name1, simulated_image_name2, simulated_image_name_current]
        return []

    uuid_side_effects = [ exec_dir_uuid, fig_uuid1, fig_uuid2, fig_uuid3, uuid.uuid4(), uuid.uuid4()]

    with patch('uuid.uuid4', side_effect=uuid_side_effects), \
         patch('asyncio.create_subprocess_exec', return_value=mock_async_proc_instance) as mock_create_subprocess, \
         patch('os.listdir', side_effect=mock_listdir_side_effect) as mock_os_listdir:

        result = await manager.execute("figure; plot(1:10); figure; plot(rand(5));")

    assert result.output == mock_matlab_stdout
    assert result.error == mock_matlab_stderr
    assert len(result.images) == 3
    assert expected_image_path1 in result.images
    assert expected_image_path2 in result.images
    assert expected_image_path_current in result.images

    mock_os_listdir.assert_any_call(full_execution_dir)

    script_path_pattern = os.path.join(full_execution_dir, f"run_script_{exec_dir_uuid.hex}.m")
    # mock_create_subprocess is the mock for asyncio.create_subprocess_exec
    args_list = mock_create_subprocess.call_args[0]
    found_script_arg = any(script_path_pattern in arg for arg in args_list if isinstance(arg, str))
    assert found_script_arg, f"Script path {script_path_pattern} not found in subprocess args: {args_list}"
    assert result.output_data is None


# --- New Test Cases for Input/Output Data Handling ---

@patch('shutil.which', return_value="mock_firejail_path")
@patch('cmcp.managers.matlab_manager.scipy') # Mock the scipy module in the manager
async def test_execute_with_input_data(mock_scipy, mock_shutil_which, mock_app_config_matlab_enabled, sandbox_dir):
    """Test execution with input data being saved and loaded."""
    # Ensure mock_scipy.io.savemat exists
    mock_scipy.io = MagicMock()
    mock_scipy.io.savemat = MagicMock()

    manager = MatlabManager.from_env(app_config=mock_app_config_matlab_enabled)
    input_data = {'my_var': np.array([1, 2, 3]), 'my_string': 'test'}
    matlab_input = MatlabInput(variables=input_data)

    mock_proc = AsyncMock()
    mock_proc.communicate = AsyncMock(return_value=(b"Output from MATLAB", b""))
    mock_proc.returncode = 0

    # Need to control exec_dir uuid for path prediction
    fixed_uuid_exec_dir = uuid.uuid4()
    expected_exec_dir_name = f"matlab_exec_{fixed_uuid_exec_dir.hex}"
    full_execution_dir = os.path.join(manager.sandbox_dir, expected_exec_dir_name)
    expected_input_mat_path = os.path.join(full_execution_dir, "input.mat")

    # Mock open to check script content for "load('input.mat');"
    # This is tricky as open is a builtin. Use `unittest.mock.mock_open`.
    # However, it's easier to verify the call to savemat and assume the load command is prepended.
    # For more rigorous test of script content, one would need to capture what's written to temp_script_path.
    # Let's focus on savemat call and that the path is correct.

    with patch('uuid.uuid4', return_value=fixed_uuid_exec_dir), \
         patch('asyncio.create_subprocess_exec', return_value=mock_proc) as mock_create_subprocess, \
         patch('builtins.open', MagicMock()) as mock_open_builtin: # Mock open to not actually write file

        await manager.execute(code="disp(my_var);", input_vars=matlab_input)

    mock_scipy.io.savemat.assert_called_once()
    # Check path argument of savemat
    call_args_savemat, _ = mock_scipy.io.savemat.call_args
    assert call_args_savemat[0] == expected_input_mat_path
    # Check data argument of savemat (handles numpy array comparison)
    assert np.array_equal(call_args_savemat[1]['my_var'], input_data['my_var'])
    assert call_args_savemat[1]['my_string'] == input_data['my_string']

    # Verify that the script path used in MATLAB's run command is correct
    # and that the script content (which we can't easily intercept without more mocks)
    # would have included the load('input.mat')
    # The manager now creates a script named "run_script.m" in execution_dir
    expected_script_path_in_matlab_run = os.path.join(full_execution_dir, "run_script.m")

    found_batch_run_command = False
    subprocess_args_list = mock_create_subprocess.call_args[0]
    for arg in subprocess_args_list:
        if f"run('{expected_script_path_in_matlab_run}')" in arg:
            found_batch_run_command = True
            break
    assert found_batch_run_command

    # To verify `load('input.mat');` is in the script, we would need to mock `open()`
    # specifically for `temp_script_path` and inspect what's written.
    # For now, this level of detail for savemat call is good.


@patch('shutil.which', return_value="mock_firejail_path")
@patch('cmcp.managers.matlab_manager.scipy')
async def test_execute_generates_output_data(mock_scipy, mock_shutil_which, mock_app_config_matlab_enabled, sandbox_dir):
    """Test that output.mat is loaded and data is returned."""
    mock_scipy.io = MagicMock()
    mock_scipy.io.loadmat = MagicMock(return_value={'result_var': np.array([4,5,6]), '__header__': 'dummy'})

    manager = MatlabManager.from_env(app_config=mock_app_config_matlab_enabled)
    mock_proc = AsyncMock(returncode=0)
    mock_proc.communicate = AsyncMock(return_value=(b"Done", b""))

    fixed_uuid_exec_dir = uuid.uuid4()
    expected_exec_dir_name = f"matlab_exec_{fixed_uuid_exec_dir.hex}"
    full_execution_dir = os.path.join(manager.sandbox_dir, expected_exec_dir_name)
    expected_output_mat_path = os.path.join(full_execution_dir, "output.mat")

    with patch('uuid.uuid4', return_value=fixed_uuid_exec_dir), \
         patch('asyncio.create_subprocess_exec', return_value=mock_proc), \
         patch('os.path.exists', return_value=True) as mock_path_exists: # Simulate output.mat exists

        result = await manager.execute(code="save('output.mat', 'result_var');")

    mock_path_exists.assert_called_with(expected_output_mat_path)
    mock_scipy.io.loadmat.assert_called_with(expected_output_mat_path)
    assert result.output_data is not None
    assert 'result_var' in result.output_data
    assert np.array_equal(result.output_data['result_var'], np.array([4,5,6]))
    assert '__header__' not in result.output_data


@patch('shutil.which', return_value="mock_firejail_path")
@patch('cmcp.managers.matlab_manager.scipy')
async def test_execute_no_output_mat_file(mock_scipy, mock_shutil_which, mock_app_config_matlab_enabled, sandbox_dir):
    """Test behavior when output.mat is not created by the MATLAB script."""
    mock_scipy.io = MagicMock() # Ensure scipy.io.loadmat is not called if file doesn't exist
    mock_scipy.io.loadmat = MagicMock()

    manager = MatlabManager.from_env(app_config=mock_app_config_matlab_enabled)
    mock_proc = AsyncMock(returncode=0)
    mock_proc.communicate = AsyncMock(return_value=(b"Done", b""))

    fixed_uuid_exec_dir = uuid.uuid4()
    with patch('uuid.uuid4', return_value=fixed_uuid_exec_dir), \
         patch('asyncio.create_subprocess_exec', return_value=mock_proc), \
         patch('os.path.exists', return_value=False) as mock_path_exists: # Simulate output.mat does NOT exist

        result = await manager.execute(code="disp('No output.mat created');")

    assert result.output_data is None
    mock_scipy.io.loadmat.assert_not_called()


@patch('shutil.which', return_value="mock_firejail_path")
@patch('cmcp.managers.matlab_manager.scipy')
async def test_execute_input_data_save_error(mock_scipy, mock_shutil_which, mock_app_config_matlab_enabled):
    """Test behavior when scipy.io.savemat raises an error."""
    mock_scipy.io = MagicMock()
    mock_scipy.io.savemat = MagicMock(side_effect=IOError("Disk full"))

    manager = MatlabManager.from_env(app_config=mock_app_config_matlab_enabled)
    input_data = {'my_var': [1, 2, 3]}
    matlab_input = MatlabInput(variables=input_data)

    # No subprocess should be called if savemat fails early
    with patch('asyncio.create_subprocess_exec', new_callable=AsyncMock) as mock_create_subprocess:
        result = await manager.execute(code="disp(my_var);", input_vars=matlab_input)

    assert "Failed to serialize input variables to .mat file: Disk full" in result.error
    assert result.output_data is None
    mock_create_subprocess.assert_not_called() # MATLAB execution should be skipped


@patch('shutil.which', return_value="mock_firejail_path")
@patch('cmcp.managers.matlab_manager.scipy')
async def test_execute_output_data_load_error(mock_scipy, mock_shutil_which, mock_app_config_matlab_enabled, sandbox_dir):
    """Test behavior when scipy.io.loadmat raises an error."""
    mock_scipy.io = MagicMock()
    # Use a specific, but potentially general scipy error if MatReadError is not available
    loadmat_exception = MatReadError("Invalid MAT-file") if MatReadError is not Exception else IOError("Invalid MAT-file")
    mock_scipy.io.loadmat = MagicMock(side_effect=loadmat_exception)

    manager = MatlabManager.from_env(app_config=mock_app_config_matlab_enabled)
    mock_proc = AsyncMock(returncode=0)
    mock_proc.communicate = AsyncMock(return_value=(b"Done", b""))

    fixed_uuid_exec_dir = uuid.uuid4()
    with patch('uuid.uuid4', return_value=fixed_uuid_exec_dir), \
         patch('asyncio.create_subprocess_exec', return_value=mock_proc), \
         patch('os.path.exists', return_value=True): # Simulate output.mat exists

        result = await manager.execute(code="save('output.mat');") # Script creates output.mat

    assert result.output_data is None
    assert "Warning: Failed to load output data from output.mat" in result.error
    assert "Invalid MAT-file" in result.error


@patch('shutil.which', return_value="mock_firejail_path")
async def test_execute_with_input_scipy_unavailable(mock_shutil_which, mock_app_config_matlab_enabled):
    """Test execution with input_vars when scipy is not available."""
    manager = MatlabManager.from_env(app_config=mock_app_config_matlab_enabled)
    input_data = {'my_var': [1, 2, 3]}
    matlab_input = MatlabInput(variables=input_data)

    # Simulate scipy being unavailable by patching it to None in the manager's module scope
    with patch('cmcp.managers.matlab_manager.scipy', None):
        result = await manager.execute(code="disp(1);", input_vars=matlab_input)

    assert "scipy.io is required for input/output variable handling but is not installed." in result.error
    assert result.output_data is None

@patch('shutil.which', return_value="mock_firejail_path")
@patch('cmcp.managers.matlab_manager.scipy', None) # Scipy unavailable for this test
async def test_execute_no_input_scipy_unavailable(mock_scipy_none, mock_shutil_which, mock_app_config_matlab_enabled):
    """Test execution without input_vars when scipy is not available (should still run)."""
    manager = MatlabManager.from_env(app_config=mock_app_config_matlab_enabled)

    mock_proc = AsyncMock()
    mock_proc.communicate = AsyncMock(return_value=(b"Simple run", b""))
    mock_proc.returncode = 0

    with patch('asyncio.create_subprocess_exec', return_value=mock_proc) as mock_create_subprocess:
        result = await manager.execute(code="disp('OK');")

    assert result.output == "Simple run"
    assert result.error == ""
    assert result.output_data is None # No attempt to load output.mat if scipy is None
    mock_create_subprocess.assert_called_once()
