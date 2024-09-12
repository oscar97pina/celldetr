class AttrDict:
    PRIV_ATTRS = ["_store_"]

    def __init__(self, **kwargs):
        self._store_ = dict()
        for k, v in kwargs.items():
            self._store_[k] = self._parse_element(v)
        
    def __getattr__(self, key):
        """ if key in self.PRIV_ATTRS:
            return super().__getattribute__(key)
        return self._store_[key] """
        try:
            if key in self.PRIV_ATTRS:
                return super().__getattribute__(key)
            return self._store_[key]
        except KeyError:
            raise AttributeError(key)
    
    def __setattr__(self, key, value):
        if key in self.PRIV_ATTRS:
            super().__setattr__(key, value)
        else:
            self._store_[key] = value
    
    def __hasattr__(self, key):
        if key in self.PRIV_ATTRS:
            return super().__hasattr__(key)
        return key in self._store_
    
    def __delattr__(self, key):
        if key in self.PRIV_ATTRS:
            pass
        else:
            del self._store_[key]
    
    def __getitem__(self, key):
        return self._store_[key]
    
    def __setitem__(self, key, value):
        self._store_[key] = value
    
    def __delitem__(self, key):
        del self._store_[key]
    
    def __iter__(self):
        return iter(self._store_)
    
    def __len__(self):
        return len(self._store_)
    
    def __contains__(self, key):
        return key in self._store_

    def update(self, other):
        if self._is_dict(other):
            other = self.__class__(**other)
            
        assert self._is_cls(other),\
            f"Other must be a dict or an object of the class {self.__class__.__name__}"

        for k, v in other.items():
            if k in self:
                # if k not in self, just add the attribute
                if k not in self:
                    self[k] = v
                # if self[k] and v are attrdicts, update
                if self._is_cls(self[k]) and self._is_cls(v):
                    self[k].update(v)
                # else, replace
                # TODO: we can model distinct behaviors here
                else:
                    self[k] = v
            else:
                self[k] = v
                
    def keys(self):
        return self._store_.keys()
    
    def values(self):
        return self._store_.values()
    
    def items(self):
        return self._store_.items()

    @classmethod
    def _is_list(cls, value):
        return isinstance(value, list) or isinstance(value, tuple)

    @classmethod
    def _is_dict(cls, value):
        return isinstance(value, dict)

    @classmethod
    def _is_str(cls, value):
        return isinstance(value, str)

    @classmethod
    def _is_cls(cls, value):
        return isinstance(value, cls)

    @classmethod
    def _parse_element(cls, value):
        if cls._is_dict(value):
            return cls._parse_dict(value)
        elif cls._is_list(value):
            return cls._parse_list(value)
        else:
            return value

    @classmethod
    def _parse_list(cls, ls):
        for i, v in enumerate(ls):
            ls[i] = cls._parse_element(v)
        return ls

    @classmethod
    def _parse_dict(cls, d):
        for k, v in d.items():
            d[k] = cls._parse_element(v)
        return cls(**d)
    
    def as_dict(self):
        import copy
        d = copy.deepcopy(self._store_)
        for key, value in d.items():
            if self._is_cls(value):
                d[key] = value.as_dict()
            if self._is_list(value):
                for i, v in enumerate(value):
                    if self._is_cls(v):
                        d[key][i] = v.as_dict()
        return d

    def has(self, key):
        return key in self._store_ and self._store_[key] is not None
    
    def get(self, key, default=None):
        return self._store_.get(key, default)

    def __str__(self):
        return f"{self.__class__.__name__}({self._store_})"
    
    def __repr__(self):
        return f"{self.__class__.__name__}({self._store_})"