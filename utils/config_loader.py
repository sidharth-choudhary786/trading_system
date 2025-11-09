# trading_system/utils/config_loader.py
"""
Configuration management utilities for the trading system.
Handles YAML configuration files, environment variables, and validation.
"""

import yaml
import os
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging
from dotenv import load_dotenv
import json

logger = logging.getLogger(__name__)

class ConfigLoader:
    """
    Configuration loader for trading system
    Handles YAML configuration files and environment variables
    """
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.logger = logging.getLogger(__name__)
        self._config_cache = {}
        
        # Load environment variables
        load_dotenv()
        
        # Ensure config directory exists
        if not self.config_dir.exists():
            raise TradingSystemError(f"Config directory not found: {config_dir}")
        
        logger.info(f"ConfigLoader initialized with directory: {config_dir}")
    
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
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
            
            if config is None:
                config = {}
                
            self.logger.debug(f"Loaded config from {filename}")
            return config
            
        except yaml.YAMLError as e:
            raise TradingSystemError(f"Error parsing YAML file {filename}: {e}")
        except Exception as e:
            raise TradingSystemError(f"Error reading config file {filename}: {e}")
    
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
            env_value = os.getenv(env_var)
            if env_value is None:
                self.logger.warning(f"Environment variable {env_var} not found")
                return config
            return env_value
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
        
        try:
            with open(filepath, 'w', encoding='utf-8') as file:
                yaml.dump(self._config_cache[mode], file, default_flow_style=False, indent=2)
                
            self.logger.info(f"Configuration saved to: {filepath}")
            
        except Exception as e:
            raise TradingSystemError(f"Error saving configuration: {e}")
    
    def create_config_template(self, mode: str, filepath: str):
        """
        Create configuration template for given mode
        
        Args:
            mode: Configuration mode
            filepath: Output file path
        """
        template = self._get_config_template(mode)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as file:
                yaml.dump(template, file, default_flow_style=False, indent=2)
                
            self.logger.info(f"Configuration template created: {filepath}")
            
        except Exception as e:
            raise TradingSystemError(f"Error creating configuration template: {e}")
    
    def _get_config_template(self, mode: str) -> Dict[str, Any]:
        """Get configuration template for mode"""
        base_template = {
            'mode': mode,
            'description': f'Trading system configuration for {mode} mode',
            'version': '1.0.0'
        }
        
        if mode == 'backtest':
            base_template.update({
                'backtest': {
                    'initial_capital': 1000000,
                    'start_date': '2020-01-01',
                    'end_date': '2023-12-31',
                    'commission': 0.001,
                    'slippage': 0.001
                }
            })
        elif mode == 'production':
            base_template.update({
                'production': {
                    'live_trading': False,
                    'paper_trading': True,
                    'broker': 'zerodha'
                }
            })
        
        return base_template
    
    def validate_config_file(self, filepath: str) -> bool:
        """
        Validate configuration file
        
        Args:
            filepath: Path to configuration file
            
        Returns:
            True if valid
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
            
            # Basic structure validation
            if not isinstance(config, dict):
                self.logger.error("Configuration must be a dictionary")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return False
    
    def get_config_stats(self, mode: str) -> Dict[str, Any]:
        """
        Get configuration statistics
        
        Args:
            mode: Configuration mode
            
        Returns:
            Configuration statistics
        """
        if mode not in self._config_cache:
            self.load_config(mode)
        
        config = self._config_cache[mode]
        
        def count_keys(d):
            if isinstance(d, dict):
                return sum(count_keys(v) for v in d.values()) + len(d)
            elif isinstance(d, list):
                return sum(count_keys(item) for item in d)
            else:
                return 1
        
        total_keys = count_keys(config)
        
        return {
            'mode': mode,
            'total_keys': total_keys,
            'sections': list(config.keys()),
            'data_sources': self.get_config_value(config, 'data.sources', []),
            'active_models': self.get_config_value(config, 'models.active_models', [])
        }

class TradingSystemError(Exception):
    """Base exception for trading system configuration errors"""
    pass

# Utility functions
def load_config(mode: str = "backtest", config_dir: str = "config") -> Dict[str, Any]:
    """Convenience function to load configuration"""
    loader = ConfigLoader(config_dir)
    return loader.load_config(mode)

def get_config_value(config: Dict, key_path: str, default: Any = None) -> Any:
    """Convenience function to get config value"""
    return ConfigLoader.get_config_value(config, key_path, default)

# Example usage
if __name__ == "__main__":
    # Test configuration loading
    try:
        config_loader = ConfigLoader()
        config = config_loader.load_config("backtest")
        
        print("Configuration loaded successfully!")
        print(f"Mode: {config.get('mode', 'N/A')}")
        print(f"Initial Capital: {config_loader.get_config_value(config, 'backtest.initial_capital', 'N/A')}")
        
        # Get configuration statistics
        stats = config_loader.get_config_stats("backtest")
        print(f"Configuration stats: {stats}")
        
    except Exception as e:
        print(f"Configuration loading failed: {e}")
