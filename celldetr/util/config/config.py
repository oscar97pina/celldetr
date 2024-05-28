import yaml
import os
import os.path as osp
from .attr import AttrDict

class ConfigDict(AttrDict):
    PRIV_ATTRS = ["_store_", "_file_", "__deepcopy__"]
    
    @classmethod
    def _parse_element(cls, value):
        if cls._is_dict(value) and '__file__' in value:
            cfg = cls.from_file(value['__file__'])
            del value['__file__']
            if cls._is_cls(cfg):
                cfg.update(cls._parse_dict(value))
            else:
                assert len(value) == 0,\
                    "Content from __file__ must be a dict if it needs to be extended"
            return cfg
        if cls._is_dict(value):
            return cls._parse_dict(value)
        elif cls._is_list(value):
            return cls._parse_list(value)
        else:
            return value
        
    @classmethod
    def from_file(cls, file):
        # if not absolute, make it absolute with dirname of __file__
        if not osp.isabs(file):
            file = osp.join(osp.realpath(os.getcwd()), file)

        # read the file
        with open(file, 'r') as f:
            content = yaml.load(f, Loader=yaml.FullLoader)

        # if is not a dict, parse and return
        # for instance, it may be a list
        if not cls._is_dict(content):
            return cls._parse_element(content)

        # create empty config
        cfg = ConfigDict(**{})

        # if __base__ in content, extend the files
        ls_bases = list()
        if "__base__" in content:
            # get bases
            ls_bases = content["__base__"]
            # it must be a list
            assert cls._is_list(ls_bases),\
                   f"__base__ must be a list, got {type(ls_bases)}"
            
        # for each base config file, load it and update current cfg
        for base in ls_bases:
            assert cls._is_str(base),\
                f"Elements of __base__ must contain paths, got {type(base)}"
            # load file
            base = ConfigDict.from_file(file=base)
            # udpate
            cfg.update( base )
        
        if "__base__" in content:
            del content["__base__"]

        # now, parse the elements of the content
        for k, v in content.items():
            # if contains __file__, read the file and use it as base
            # for the current attribute
            cfg.update(ConfigDict(**{k : cls._parse_element(v)}))
            """
            if cls._is_dict(v) and '__file__' in v:
                # get filename
                base = v['__file__']
                assert cls._is_str(base),\
                    f"__file__ must be a string, got {type(base)}"
                
                # create config dict from file
                new_v = ConfigDict.from_file(file=base)
                del v['__file__']

                # the content of __file__ is dict
                if cls._is_cls(new_v):
                    # update with possible new attrs from v
                    new_v.update(ConfigDict(**v))
                # the content of __file__ is sth else (ie list)
                else:
                    # ensure there are no more attrs on v, as we cannot extend them
                    assert len(v) == 0,\
                        "Content from __file__ must be a dict if it needs to be extended"
                v = new_v
                """ 
        return cfg
                        
    @classmethod
    def from_options(cls, options):
        # options is a list of path.to.key=value
        # parse the list and create a dict
        # it does not allow __base__ option
        if options is None:
            return ConfigDict(**dict())
        
        def _parse_str_value(value):
            # the value can be a string, a number, a boolean, or a list of strings, numbers, booleans
            # check if it is a list
            if value.startswith("[") and value.endswith("]"):
                # remove the brackets
                value = value[1:-1]
                # split the list
                value = value.split(",")
                # parse each element
                value = [_parse_str_value(v) for v in value]
                return value
            # check if it is a boolean
            if value.lower() in ["true", "false"]:
                return value.lower()=="true"
            # check if it is a number
            try:
                return int(value)
            except ValueError:
                pass
            try:
                return float(value)
            except ValueError:
                pass
            # otherwise it is a string
            return value
        
        cfg_dict = dict()
        for option in options:
            assert "=" in option, \
                f"Invalid option {option}, must be path.to.key=value"
            key, value = option.split("=")
            # parse the value
            value = _parse_str_value(value)
            # split the key in a list of keys
            key = key.split(".")
            # create the dict
            d = cfg_dict
            for k in key[:-1]:
                if k not in d:
                    d[k] = dict()
                d = d[k]
            d[key[-1]] = value
        return ConfigDict(**cfg_dict)