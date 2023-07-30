from copy import deepcopy
from typing import Tuple, Optional, Any, Dict


class ConfigError(BaseException):
    pass


class Singleton(type):
    """
    Singleton class.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class ConfigParser(metaclass=Singleton):
    """
    Class for handling parsing of YAML config file.
    """

    def __init__(self, config: Optional[Dict] = None) -> None:
        self._config = deepcopy(config)

    @property
    def config(self) -> Optional[Dict]:
        return self._config

    @config.setter
    def config(self, new_config: Dict) -> None:
        self._config = deepcopy(new_config)

    def get_param(self, keys: Tuple[str, ...], type: type, default: Optional[Any] = None) -> Optional[Any]:
        """
        Getting value from config or using default.
        """

        if self._config is None:
            raise ConfigError(f"Cannot get param with keys {' '.join(keys)}, because config doesn't exist.")

        param = None
        try:
            for key in keys:
                if param is None:
                    param = self._config[key]
                else:
                    param = param[key]
        except KeyError:
            param = default
        finally:
            if param is None:
                raise ConfigError(f"No parameter in config under keys {' '.join(keys)}.")
            else:
                param = type(param)

        return param
