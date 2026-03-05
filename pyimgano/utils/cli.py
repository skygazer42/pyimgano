"""
CLI and Engineering Tools

Features:
- Command-line argument parsing
- Configuration management (YAML/JSON)
- Logging utilities
- Progress bars
- Experiment tracking
- Plugin system
"""

from typing import Optional, Dict, Any, List
from pathlib import Path
import json
import yaml
import argparse
import logging
from datetime import datetime
import sys


class ArgumentParser:
    """Enhanced argument parser for CLI applications."""

    def __init__(self, description: str = ""):
        """
        Initialize argument parser.

        Parameters
        ----------
        description : str, default=""
            Program description
        """
        self.parser = argparse.ArgumentParser(description=description)
        self.subparsers = None

    def add_argument(
        self,
        name: str,
        type: type = str,
        default: Any = None,
        help: str = "",
        required: bool = False,
        choices: Optional[List] = None
    ):
        """
        Add argument.

        Parameters
        ----------
        name : str
            Argument name
        type : type, default=str
            Argument type
        default : Any, optional
            Default value
        help : str, default=""
            Help text
        required : bool, default=False
            Required flag
        choices : list, optional
            Valid choices
        """
        kwargs = {
            'type': type,
            'help': help,
        }

        if default is not None:
            kwargs['default'] = default

        if required:
            kwargs['required'] = required

        if choices:
            kwargs['choices'] = choices

        self.parser.add_argument(name, **kwargs)

    def add_subcommand(self, name: str, help: str = "") -> argparse.ArgumentParser:
        """
        Add subcommand.

        Parameters
        ----------
        name : str
            Subcommand name
        help : str, default=""
            Help text

        Returns
        -------
        subparser : ArgumentParser
            Subcommand parser
        """
        if self.subparsers is None:
            self.subparsers = self.parser.add_subparsers(dest='command')

        return self.subparsers.add_parser(name, help=help)

    def parse_args(self, args: Optional[List[str]] = None) -> argparse.Namespace:
        """
        Parse command-line arguments.

        Parameters
        ----------
        args : list, optional
            Arguments to parse (default: sys.argv)

        Returns
        -------
        args : Namespace
            Parsed arguments
        """
        return self.parser.parse_args(args)


class ConfigManager:
    """Configuration file management."""

    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize config manager.

        Parameters
        ----------
        config_file : str, optional
            Path to configuration file
        """
        self.config_file = config_file
        self.config = {}

        if config_file and Path(config_file).exists():
            self.load(config_file)

    def load(self, config_file: str):
        """
        Load configuration from file.

        Parameters
        ----------
        config_file : str
            Configuration file path
        """
        path = Path(config_file)

        if path.suffix in ['.yaml', '.yml']:
            with open(path, 'r') as f:
                self.config = yaml.safe_load(f) or {}
        elif path.suffix == '.json':
            with open(path, 'r') as f:
                self.config = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")

        self.config_file = config_file

    def save(self, config_file: Optional[str] = None):
        """
        Save configuration to file.

        Parameters
        ----------
        config_file : str, optional
            Configuration file path (default: current config file)
        """
        if config_file is None:
            config_file = self.config_file

        if config_file is None:
            raise ValueError("No config file specified")

        path = Path(config_file)

        if path.suffix in ['.yaml', '.yml']:
            with open(path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
        elif path.suffix == '.json':
            with open(path, 'w') as f:
                json.dump(self.config, f, indent=2)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.

        Parameters
        ----------
        key : str
            Configuration key (supports dot notation: 'section.subsection.key')
        default : Any, optional
            Default value if key not found

        Returns
        -------
        value : Any
            Configuration value
        """
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any):
        """
        Set configuration value.

        Parameters
        ----------
        key : str
            Configuration key (supports dot notation)
        value : Any
            Value to set
        """
        keys = key.split('.')
        config = self.config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def update(self, updates: Dict[str, Any]):
        """
        Update configuration with dictionary.

        Parameters
        ----------
        updates : dict
            Configuration updates
        """
        self.config.update(updates)


class Logger:
    """Enhanced logging utility."""

    def __init__(
        self,
        name: str,
        log_file: Optional[str] = None,
        level: str = 'INFO',
        format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ):
        """
        Initialize logger.

        Parameters
        ----------
        name : str
            Logger name
        log_file : str, optional
            Log file path
        level : str, default='INFO'
            Logging level: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
        format : str
            Log message format
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))

        # Remove existing handlers
        self.logger.handlers = []

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(format))
        self.logger.addHandler(console_handler)

        # File handler
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter(format))
            self.logger.addHandler(file_handler)

    def debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)

    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)

    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)

    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)

    def critical(self, message: str):
        """Log critical message."""
        self.logger.critical(message)


class ProgressBar:
    """Simple progress bar."""

    def __init__(self, total: int, desc: str = "", bar_length: int = 40):
        """
        Initialize progress bar.

        Parameters
        ----------
        total : int
            Total number of items
        desc : str, default=""
            Description
        bar_length : int, default=40
            Progress bar length
        """
        self.total = total
        self.desc = desc
        self.bar_length = bar_length
        self.current = 0
        self.start_time = None

    def __enter__(self):
        """Start progress bar."""
        self.start_time = datetime.now()
        self.update(0)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Finish progress bar."""
        self.update(self.total)
        print()  # New line

    def update(self, n: int = 1):
        """
        Update progress.

        Parameters
        ----------
        n : int, default=1
            Number of items completed
        """
        self.current += n
        self._print()

    def set(self, value: int):
        """
        Set current progress.

        Parameters
        ----------
        value : int
            Current progress value
        """
        self.current = value
        self._print()

    def _print(self):
        """Print progress bar."""
        if self.total == 0:
            return

        progress = self.current / self.total
        filled = int(self.bar_length * progress)
        bar = '=' * filled + '-' * (self.bar_length - filled)

        # Calculate ETA
        if self.start_time and self.current > 0:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            eta = elapsed / self.current * (self.total - self.current)
            eta_str = f"{eta:.1f}s"
        else:
            eta_str = "?"

        desc_str = f"{self.desc}: " if self.desc else ""
        print(f"\r{desc_str}[{bar}] {self.current}/{self.total} ({progress*100:.1f}%) ETA: {eta_str}",
              end='', flush=True)


class ExperimentTracker:
    """Track experiment runs and results."""

    def __init__(self, experiment_dir: str):
        """
        Initialize experiment tracker.

        Parameters
        ----------
        experiment_dir : str
            Directory to store experiment results
        """
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        self.current_run = None
        self.run_dir = None

    def start_run(self, run_name: Optional[str] = None, config: Optional[Dict] = None):
        """
        Start new experiment run.

        Parameters
        ----------
        run_name : str, optional
            Run name (default: timestamp)
        config : dict, optional
            Run configuration
        """
        if run_name is None:
            run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.current_run = run_name
        self.run_dir = self.experiment_dir / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        if config:
            with open(self.run_dir / 'config.json', 'w') as f:
                json.dump(config, f, indent=2)

        # Create metadata
        metadata = {
            'run_name': run_name,
            'start_time': datetime.now().isoformat(),
        }

        with open(self.run_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log metrics.

        Parameters
        ----------
        metrics : dict
            Metric values
        step : int, optional
            Training step
        """
        if self.run_dir is None:
            raise RuntimeError("No active run. Call start_run() first.")

        metrics_file = self.run_dir / 'metrics.jsonl'

        entry = {
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics
        }

        if step is not None:
            entry['step'] = step

        with open(metrics_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')

    def save_artifact(self, artifact_path: str, artifact_name: Optional[str] = None):
        """
        Save artifact to run directory.

        Parameters
        ----------
        artifact_path : str
            Path to artifact file
        artifact_name : str, optional
            Name to save artifact as
        """
        if self.run_dir is None:
            raise RuntimeError("No active run. Call start_run() first.")

        import shutil

        if artifact_name is None:
            artifact_name = Path(artifact_path).name

        dest = self.run_dir / artifact_name
        shutil.copy2(artifact_path, dest)

    def end_run(self):
        """End current run."""
        if self.run_dir:
            # Update metadata
            metadata_file = self.run_dir / 'metadata.json'
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)

                metadata['end_time'] = datetime.now().isoformat()

                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)

        self.current_run = None
        self.run_dir = None


class PluginManager:
    """Plugin system for extensibility."""

    def __init__(self):
        """Initialize plugin manager."""
        self.plugins = {}

    def register(self, name: str, plugin: Any):
        """
        Register plugin.

        Parameters
        ----------
        name : str
            Plugin name
        plugin : Any
            Plugin object
        """
        self.plugins[name] = plugin

    def get(self, name: str) -> Optional[Any]:
        """
        Get plugin by name.

        Parameters
        ----------
        name : str
            Plugin name

        Returns
        -------
        plugin : Any or None
            Plugin object
        """
        return self.plugins.get(name)

    def list_plugins(self) -> List[str]:
        """
        List registered plugins.

        Returns
        -------
        plugins : list
            List of plugin names
        """
        return list(self.plugins.keys())

    def unregister(self, name: str):
        """
        Unregister plugin.

        Parameters
        ----------
        name : str
            Plugin name
        """
        if name in self.plugins:
            del self.plugins[name]


# Convenience functions
def setup_logger(
    name: str = 'pyimgano',
    log_file: Optional[str] = None,
    level: str = 'INFO'
) -> Logger:
    """
    Set up logger.

    Parameters
    ----------
    name : str, default='pyimgano'
        Logger name
    log_file : str, optional
        Log file path
    level : str, default='INFO'
        Logging level

    Returns
    -------
    logger : Logger
        Configured logger
    """
    return Logger(name, log_file, level)


def load_config(config_file: str) -> ConfigManager:
    """
    Load configuration file.

    Parameters
    ----------
    config_file : str
        Configuration file path

    Returns
    -------
    config : ConfigManager
        Configuration manager
    """
    return ConfigManager(config_file)


def create_experiment_tracker(experiment_dir: str = './experiments') -> ExperimentTracker:
    """
    Create experiment tracker.

    Parameters
    ----------
    experiment_dir : str, default='./experiments'
        Experiment directory

    Returns
    -------
    tracker : ExperimentTracker
        Experiment tracker
    """
    return ExperimentTracker(experiment_dir)
