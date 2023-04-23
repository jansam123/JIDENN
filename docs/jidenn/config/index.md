Module jidenn.config
====================
This module to fully configure the data preparation, training and evaulation.
The configuration is done using the hydra package https://hydra.cc/docs/intro/.
The basic idea is that the user can specify a configuration file in yaml format
such that it follows the structure of the `jidenn.config.config.JIDENNConfig` dataclass. 
The hydra package reads the configuration file and creates a `jidenn.config.config.JIDENNConfig` 
object that is passed to the main function with the following decorator:

```python
@hydra.main(version_base="1.2", config_path="jidenn/config", config_name="config")
def main(args: config.JIDENNConfig) -> None:
    some_option = args.some_option
```
`config_path` is the path to the folder where the configuration file 
`config.yaml` (`config_name` option) is located. The configured variables
are then accessible via the `args` object's attributes.

To match the configuration file to the `jidenn.config.config.JIDENNConfig` class 
with the `.yaml` file the argument of the `main()` must be matched to the
`jidenn.config.config.JIDENNConfig` class. This is done by the following lines:
```python
from hydra.core.config_store import ConfigStore
from jidenn.config import config

cs = ConfigStore.instance()
cs.store(name="args", node=config.JIDENNConfig)
```

Dataclasses can be nested, each nested dataclass represents a sub-configuration,
i.e. an indentation in the yaml file. 
The following dataclasses
```python
from dataclasses import dataclass

@dataclass
class SubConfig1:
    sub_config1_arg1: str
    sub_config1_arg2: int

@dataclass
class SubConfig2:
    sub_config2_arg1: str
    sub_config2_arg2: int

@dataclass
class GeneralConfig:
    sub_config1: SubConfig1
    sub_config2: SubConfig2
    
@dataclass
class JIDENNConfig:
    general: GeneralConfig
    ...
```  
are represented in the yaml file as:
```yaml
general:
  sub_config1:
    sub_config1_arg1: "test"
    sub_config1_arg2: 1
  sub_config2:
    sub_config2_arg1: "train"
    sub_config2_arg2: 2
```

Optionally the `.yaml` files can be split into multiple files. This is done by
addign the `defaults` entry in the main `.yaml` file. The `defaults` entry is a list:
```yaml
defaults:
    - data: data2
    - _self_
```
The second `.yaml` file must be located in the `jidenn/config/data` folder (generaly in the `config_path/data`, 
where `config_path` is the path specified in the `@hydra.main` decorator)  and must
be named `data2.yaml`. The `_self_` entry tells hydra to overwrite the default,
so if you change some entry in the main `.yaml` file, it will overwrite the `defaults`.

.. warning:: 
 
    All dataclasses **must** have type annotation, so the user can see what options
    are available. The `Literal` type is used to specify concrete options for a string 
    variable that are available.

Some configuration options are `Optional`. In python, missing arguments are
assumed to be `None`. In the `.yaml` file, this can be achieved by omitting
the argument or by setting it to `null`:
```yaml
# Two ways to set an argument to None 
optional_arg: null
optional_arg: 
```

Hydra also provides a way to override the configuration file from the command line:
```bash
python3 train.py general.sub_config1.sub_config1_arg2=2
```
For more information on how to use hydra, see https://hydra.cc/docs/intro/

Sub-modules
-----------
* jidenn.config.config
* jidenn.config.eval_config
* jidenn.config.model_config