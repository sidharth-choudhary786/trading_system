# trading_system/utils/config_loader.py
import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
import logging

from ..core.exceptions import TradingSystemError

class ConfigLoader:
    """
    Configuration loader for trading system
    Handles YAML configuration files and environment variables
    """
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.logger = logging.getLogger(__name__)
        self._config_cache = {}
        
        # Ensure config directory exists
        if not self.config_dir.exists():
            raise TradingSystemError(f"Config directory not found: {config_dir}")
    
    def load_config(self, mode: str = "backtest") -> Dict[str, Any]:
        """
        Load configuration for specified mode
        
        Args:
            mode: "backtest" or "production"
            
        Returns:
            Combined configuration dictionary
        """
        if mode in self._config_cache:
            return self._config_cache[mode]
            
        try:
            # Load shared configuration first
            shared_config = self._load_yaml_file("shared.yaml")
            
            # Load mode-specific configuration
            if mode == "backtest":
                mode_config = self._load_yaml_file("backtest.yaml")
            elif mode == "production":
                mode_config = self._load_yaml_file("production.yaml")
            else:
                raise TradingSystemError(f"Unknown mode: {mode}")
            
            # Merge configurations (mode-specific overrides shared)
            config = self._deep_merge(shared_config, mode_config)
            
            # Resolve environment variables
            config = self._resolve_env_vars(config)
            
            # Validate configuration
            self._validate_config(config, mode)
            
            self._config_cache[mode] = config
            self.logger.info(f"Configuration loaded successfully for mode: {mode}")
            
            return config
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            raise TradingSystemError(f"Configuration loading failed: {e}")
    
    def _load_yaml_file(self, filename: str) -> Dict[str, Any]:
        """Load YAML configuration file"""
        file_path = self.config_dir / filename
        
        if not file_path.exists():
            raise TradingSystemError(f"Config file not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        
        if config is None:
            config = {}
            
        return config
    
    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """Deep merge two dictionaries"""
        result = base.copy()
        
        for key, value in override.items():
            if (key in result and isinstance(result[key], dict) 
                and isinstance(value, dict)):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
                
        return result
    
    def _resolve_env_vars(self, config: Dict) -> Dict:
        """Resolve environment variables in configuration"""
        if isinstance(config, dict):
            return {k: self._resolve_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._resolve_env_vars(item) for item in config]
        elif isinstance(config, str) and config.startswith("${") and config.endswith("}"):
            env_var = config[2:-1]
            return os.getenv(env_var, config)
        else:
            return config
    
    def _validate_config(self, config: Dict, mode: str):
        """Validate configuration"""
        required_sections = ['data', 'models', 'portfolio', 'risk']
        
        for section in required_sections:
            if section not in config:
                raise TradingSystemError(f"Missing required configuration section: {section}")
        
        # Mode-specific validations
        if mode == "production":
            if 'broker' not in config.get('execution', {}):
                raise TradingSystemError("Broker configuration required for production mode")
            
            broker = config['execution']['broker']
            if broker not in config:
                raise TradingSystemError(f"Broker configuration missing for: {broker}")
    
    def get_config_value(self, config: Dict, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation
        
        Args:
            config: Configuration dictionary
            key_path: Dot-separated key path (e.g., 'data.sources')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key_path.split('.')
        current = config
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
                
        return current
    
    def update_config(self, mode: str, updates: Dict[str, Any]):
        """
        Update configuration with new values
        
        Args:
            mode: Configuration mode
            updates: Dictionary of updates
        """
        if mode not in self._config_cache:
            self.load_config(mode)
            
        self._config_cache[mode] = self._deep_merge(self._config_cache[mode], updates)
        self.logger.info(f"Configuration updated for mode: {mode}")
    
    def save_config(self, mode: str, filepath: Optional[str] = None):
        """
        Save current configuration to file
        
        Args:
            mode: Configuration mode
            filepath: Optional custom filepath
        """
        if mode not in self._config_cache:
            raise TradingSystemError(f"No configuration loaded for mode: {mode}")
            
        if filepath is None:
            filepath = self.config_dir / f"{mode}_current.yaml"
        
        with open(filepath, 'w', encoding='utf-8') as file:
            yaml.dump(self._config_cache[mode], file, default_flow_style=False)
            
        self.logger.info(f"Configuration saved to: {filepath}")
