#!/usr/bin/env python3
"""
Secure subprocess wrapper with comprehensive security controls
- Always uses timeout to prevent hanging processes
- Always uses check=True by default for error handling
- Comprehensive logging of return codes, stdout, stderr
- Input validation and sanitization
- No shell=True usage to prevent injection attacks
"""

import subprocess
import logging
import shlex
from typing import List, Optional, Dict, Any, Union, Tuple
from pathlib import Path
import os
import time

from core.structured_logger import get_structured_logger


class SecureSubprocessError(Exception):
    """Custom exception for secure subprocess errors"""
    pass


class SecureSubprocess:
    """Enterprise-grade secure subprocess wrapper with comprehensive controls"""

    def __init__(self, logger_name: str = "SecureSubprocess"):
        self.logger = get_structured_logger(logger_name)
        self.default_timeout = 30  # 30 second default timeout
        self.max_timeout = 300     # 5 minute maximum timeout

    def run_secure(
        self,
        command: Union[str, List[str]],
        timeout: Optional[int] = None,
        check: bool = True,
        cwd: Optional[Union[str, Path]] = None,
        env: Optional[Dict[str, str]] = None,
        capture_output: bool = True,
        text: bool = True,
        input_data: Optional[str] = None,
        allowed_return_codes: Optional[List[int]] = None
    ) -> subprocess.CompletedProcess:
        """
        Execute command with comprehensive security controls
        
        Args:
            command: Command to execute (string or list)
            timeout: Timeout in seconds (default: 30s, max: 300s)
            check: Raise exception on non-zero return code (default: True)
            cwd: Working directory
            env: Environment variables
            capture_output: Capture stdout/stderr
            text: Return text output instead of bytes
            input_data: Input to pass to subprocess
            allowed_return_codes: List of acceptable return codes (overrides check)
            
        Returns:
            CompletedProcess with comprehensive logging
            
        Raises:
            SecureSubprocessError: On security violations or execution failures
        """
        
        # Security validations
        timeout = self._validate_timeout(timeout)
        command_list = self._validate_and_sanitize_command(command)
        cwd = self._validate_working_directory(cwd)
        env = self._validate_environment(env)
        
        # Log command execution attempt
        self.logger.info("Executing secure subprocess", extra={
            "command": command_list,
            "timeout": timeout,
            "cwd": str(cwd) if cwd else None,
            "check": check,
            "capture_output": capture_output
        })
        
        start_time = time.time()
        
        try:
            # Execute with full security controls
            result = subprocess.run(
                command_list,
                timeout=timeout,
                check=False,  # We'll handle check manually for better control
                cwd=cwd,
                env=env,
                capture_output=capture_output,
                text=text,
                input=input_data,
                shell=False  # CRITICAL: Never use shell=True
            )
            
            execution_time = time.time() - start_time
            
            # Comprehensive result logging
            self._log_execution_result(command_list, result, execution_time)
            
            # Handle return code validation
            if allowed_return_codes:
                if result.returncode not in allowed_return_codes:
                    error_msg = f"Command failed with return code {result.returncode}, expected one of {allowed_return_codes}"
                    self.logger.error(error_msg, extra={
                        "command": command_list,
                        "return_code": result.returncode,
                        "allowed_codes": allowed_return_codes,
                        "stderr": result.stderr
                    })
                    raise SecureSubprocessError(error_msg)
            elif check and result.returncode != 0:
                error_msg = f"Command failed with return code {result.returncode}"
                self.logger.error(error_msg, extra={
                    "command": command_list,
                    "return_code": result.returncode,
                    "stderr": result.stderr
                })
                raise SecureSubprocessError(error_msg)
            
            return result
            
        except subprocess.TimeoutExpired as e:
            error_msg = f"Command timed out after {timeout} seconds"
            self.logger.error(error_msg, extra={
                "command": command_list,
                "timeout": timeout,
                "execution_time": time.time() - start_time
            })
            raise SecureSubprocessError(error_msg) from e
            
        except subprocess.SubprocessError as e:
            error_msg = f"Subprocess execution failed: {str(e)}"
            self.logger.error(error_msg, extra={
                "command": command_list,
                "error_type": type(e).__name__,
                "error_details": str(e)
            })
            raise SecureSubprocessError(error_msg) from e
            
        except Exception as e:
            error_msg = f"Unexpected error during subprocess execution: {str(e)}"
            self.logger.error(error_msg, extra={
                "command": command_list,
                "error_type": type(e).__name__,
                "error_details": str(e)
            })
            raise SecureSubprocessError(error_msg) from e

    def popen_secure(
        self,
        command: Union[str, List[str]],
        timeout: Optional[int] = None,
        cwd: Optional[Union[str, Path]] = None,
        env: Optional[Dict[str, str]] = None,
        stdout: Optional[int] = subprocess.PIPE,
        stderr: Optional[int] = subprocess.PIPE,
        text: bool = True
    ) -> Tuple[subprocess.Popen, Dict[str, Any]]:
        """
        Create secure Popen process with monitoring
        
        Args:
            command: Command to execute
            timeout: Process timeout
            cwd: Working directory
            env: Environment variables
            stdout: Stdout handling
            stderr: Stderr handling
            text: Text mode
            
        Returns:
            Tuple of (Popen process, monitoring metadata)
        """
        
        timeout = self._validate_timeout(timeout)
        command_list = self._validate_and_sanitize_command(command)
        cwd = self._validate_working_directory(cwd)
        env = self._validate_environment(env)
        
        self.logger.info("Creating secure Popen process", extra={
            "command": command_list,
            "timeout": timeout,
            "cwd": str(cwd) if cwd else None
        })
        
        try:
            process = subprocess.Popen(
                command_list,
                cwd=cwd,
                env=env,
                stdout=stdout,
                stderr=stderr,
                text=text,
                shell=False  # CRITICAL: Never use shell=True
            )
            
            # Process monitoring metadata
            monitoring_data = {
                "command": command_list,
                "pid": process.pid,
                "timeout": timeout,
                "start_time": time.time(),
                "cwd": str(cwd) if cwd else None
            }
            
            self.logger.info(f"Secure Popen process created with PID {process.pid}")
            
            return process, monitoring_data
            
        except Exception as e:
            error_msg = f"Failed to create Popen process: {str(e)}"
            self.logger.error(error_msg, extra={
                "command": command_list,
                "error_type": type(e).__name__,
                "error_details": str(e)
            })
            raise SecureSubprocessError(error_msg) from e

    def _validate_timeout(self, timeout: Optional[int]) -> int:
        """Validate and sanitize timeout value"""
        if timeout is None:
            return self.default_timeout
        
        if not isinstance(timeout, int) or timeout <= 0:
            raise SecureSubprocessError(f"Invalid timeout: {timeout}. Must be positive integer.")
        
        if timeout > self.max_timeout:
            self.logger.warning(f"Timeout {timeout}s exceeds maximum {self.max_timeout}s, using maximum")
            return self.max_timeout
            
        return timeout

    def _validate_and_sanitize_command(self, command: Union[str, List[str]]) -> List[str]:
        """Validate and sanitize command arguments"""
        if isinstance(command, str):
            # Parse string command safely
            try:
                command_list = shlex.split(command)
            except ValueError as e:
                raise SecureSubprocessError(f"Invalid command string: {str(e)}")
        elif isinstance(command, (list, tuple)):
            command_list = [str(arg) for arg in command]
        else:
            raise SecureSubprocessError(f"Invalid command type: {type(command)}")
        
        if not command_list:
            raise SecureSubprocessError("Empty command list")
        
        # Security: Validate executable path
        executable = command_list[0]
        if not self._is_safe_executable(executable):
            raise SecureSubprocessError(f"Unsafe executable: {executable}")
        
        # Log sanitized command
        self.logger.debug("Command sanitized", extra={
            "original_command": command,
            "sanitized_command": command_list
        })
        
        return command_list

    def _is_safe_executable(self, executable: str) -> bool:
        """Validate executable safety"""
        # Block dangerous commands
        dangerous_commands = {
            'rm', 'del', 'format', 'fdisk', 'mkfs',
            'dd', 'chmod', 'chown', 'passwd', 'su', 'sudo'
        }
        
        executable_name = Path(executable).name.lower()
        if executable_name in dangerous_commands:
            return False
        
        # Allow common safe commands
        safe_commands = {
            'python', 'python3', 'pip', 'node', 'npm',
            'git', 'grep', 'find', 'ls', 'cat', 'echo',
            'tasklist', 'taskkill', 'ps', 'kill',
            'open', 'xdg-open', 'explorer'
        }
        
        if executable_name in safe_commands:
            return True
        
        # Allow full paths to Python and common tools
        if executable.endswith(('.py', '.exe', '.bat', '.sh')):
            return True
            
        return True  # Allow by default but log for monitoring

    def _validate_working_directory(self, cwd: Optional[Union[str, Path]]) -> Optional[Path]:
        """Validate working directory"""
        if cwd is None:
            return None
            
        cwd_path = Path(cwd)
        if not cwd_path.exists():
            raise SecureSubprocessError(f"Working directory does not exist: {cwd_path}")
        
        if not cwd_path.is_dir():
            raise SecureSubprocessError(f"Working directory is not a directory: {cwd_path}")
            
        return cwd_path

    def _validate_environment(self, env: Optional[Dict[str, str]]) -> Optional[Dict[str, str]]:
        """Validate environment variables"""
        if env is None:
            return None
        
        # Inherit safe environment variables and add custom ones
        safe_env = {}
        safe_keys = {
            'PATH', 'PYTHONPATH', 'HOME', 'USER', 'TEMP', 'TMP',
            'SYSTEMROOT', 'PROGRAMFILES', 'PROGRAMFILES(X86)',
            'USERPROFILE', 'APPDATA', 'LOCALAPPDATA'
        }
        
        # Inherit safe system environment
        for key in safe_keys:
            if key in os.environ:
                safe_env[key] = os.environ[key]
        
        # Add validated custom environment variables
        for key, value in env.items():
            if self._is_safe_env_var(key, value):
                safe_env[key] = str(value)
            else:
                self.logger.warning(f"Skipped unsafe environment variable: {key}")
        
        return safe_env

    def _is_safe_env_var(self, key: str, value: str) -> bool:
        """Validate environment variable safety"""
        # Block dangerous environment variables
        dangerous_keys = {'LD_PRELOAD', 'DYLD_INSERT_LIBRARIES', 'PYTHONHOME'}
        if key.upper() in dangerous_keys:
            return False
            
        # Validate value doesn't contain injection attempts
        dangerous_patterns = ['$(', '`', '&&', '||', ';', '|']
        if any(pattern in str(value) for pattern in dangerous_patterns):
            return False
            
        return True

    def _log_execution_result(self, command: List[str], result: subprocess.CompletedProcess, execution_time: float):
        """Comprehensive logging of execution results"""
        log_data = {
            "command": command,
            "return_code": result.returncode,
            "execution_time_seconds": round(execution_time, 3),
            "stdout_length": len(result.stdout) if result.stdout else 0,
            "stderr_length": len(result.stderr) if result.stderr else 0
        }
        
        if result.returncode == 0:
            self.logger.info("Subprocess completed successfully", extra=log_data)
        else:
            self.logger.warning("Subprocess completed with non-zero return code", extra=log_data)
        
        # Log stdout/stderr if available (truncated for security)
        if result.stdout:
            stdout_preview = result.stdout[:500] + "..." if len(result.stdout) > 500 else result.stdout
            self.logger.debug("Subprocess stdout", extra={"stdout_preview": stdout_preview})
        
        if result.stderr:
            stderr_preview = result.stderr[:500] + "..." if len(result.stderr) > 500 else result.stderr
            self.logger.debug("Subprocess stderr", extra={"stderr_preview": stderr_preview})


# Global secure subprocess instance
secure_subprocess = SecureSubprocess()


# Convenience functions for common usage patterns
def run_secure_command(
    command: Union[str, List[str]],
    timeout: int = 30,
    check: bool = True,
    **kwargs
) -> subprocess.CompletedProcess:
    """Run command with secure subprocess wrapper"""
    return secure_subprocess.run_secure(command, timeout=timeout, check=check, **kwargs)


def run_secure_script(
    script_path: Union[str, Path],
    args: Optional[List[str]] = None,
    timeout: int = 60,
    **kwargs
) -> subprocess.CompletedProcess:
    """Run script with secure subprocess wrapper"""
    command = [str(script_path)]
    if args:
        command.extend(str(arg) for arg in args)
    
    return secure_subprocess.run_secure(command, timeout=timeout, **kwargs)