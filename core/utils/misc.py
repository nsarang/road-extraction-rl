import re
import torch
import numpy as np
from copy import deepcopy
from json import JSONEncoder
from collections import defaultdict
from sortedcontainers import SortedDict
from .clock import TimedIdentifiable


class DummyObject:
    def noOp(*args, **kwargs):
        pass

    def __getattr__(self, _):
        return self.noOp


class AutoMixinMeta(type):
    def __call__(cls, *args, **kwargs):
        try:
            mixin = kwargs.pop("mixin")
            name = f"{cls.__name__}{mixin.__name__}"
            cls = type(name, (mixin, cls), dict(cls.__dict__))
        except KeyError:
            pass
        return type.__call__(cls, *args, **kwargs)


class JSONSerializer(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.generic):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.tolist()
        return super().default(obj)


class IndexedOrderedDict(SortedDict):  # ith key == self.iloc[i]
    def __init__(self, *args, **kwargs):
        super().__init__((lambda key: 0), *args, **kwargs)


class DefaultIndexedOrderedDict(IndexedOrderedDict):
    def __init__(self, default_factory, *args, **kwargs):
        assert callable(default_factory)
        self.default_factory = default_factory
        super().__init__(*args, **kwargs)

    def __missing__(self, key):
        self[key] = value = self.default_factory()
        return value


class UndoMixin(TimedIdentifiable):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._history = DefaultIndexedOrderedDict(list)

    def add_revert(self, op, *args, __copy__=True, **kwargs):
        if __copy__:
            args = deepcopy(args)
            kwargs = deepcopy(kwargs)
        self._history[self.clock.step].append((op, args, kwargs))

    def undo(self, index=None, count=None, iloc=False):
        assert (index is not None) + (count is not None) <= 1
        if count:
            while count:
                key = self._history.iloc[-1]
                ops = self._history[key]
                if len(ops) == 0:
                    self._history.pop(key)
                else:
                    op, args, kwargs = ops.pop()
                    op(*args, **kwargs)
                    count -= 1
        else:
            index = self._cindex(index, iloc)
            if index in self._history:
                for (op, args, kwargs) in reversed(self._history.pop(index)):
                    op(*args, **kwargs)

    def _cindex(self, index, iloc=False):
        if iloc:
            if index is None:
                index = -1
            index = self._history.iloc[index]
        else:
            if index is None:
                index = self.clock.step
            elif index < 0:
                index += self.clock.step + 1
        return index

    def reset(self):
        for _ in range(len(self._history)):
            self.undo(index=-1, iloc=True)

    def clear(self):
        self._history.clear()


class StackU(UndoMixin, list):
    def append(self, item):
        super().append(item)
        self.add_revert(super().pop)

    def pop(self):
        item = super().pop()
        self.add_revert(super().append, item)


class TStackU(UndoMixin, SortedDict):
    def __getitem__(self, key):
        if isinstance(key, slice):
            lastStep = self.iloc[-1] if len(self) else -1
            indices = range(*key.indices(lastStep + 1))
            return [self[ind] for ind in indices if ind in self]
        key = self._cindex(key)
        return super().__getitem__(key)

    def __setitem__(self, key, value):
        key = self._cindex(key)
        return super().__setitem__(key, value)

    def append(self, item, index=None, iloc=False, replace=True):
        index = self._cindex(index, iloc)
        if index in self:
            if replace:
                self.add_revert(self.__setitem__, index, self[index])
            else:
                raise RuntimeError
        else:
            self.add_revert(super().pop, index)
        self[index] = item

    def pop(self, index=None, iloc=False):
        index = self._cindex(index, iloc)
        if index in self:
            self.add_revert(self.__setitem__, index, self[index])
            super().pop(index)

    def _cindex(self, index, iloc=False):
        if iloc:
            if index is None:
                index = -1
            index = self.iloc[index]
        else:
            if index is None:
                index = self.clock.step
            elif index < 0:
                index += self.clock.step + 1
        return index


def split(string, delimiters, maxsplit=0):
    regexPattern = "|".join(map(re.escape, delimiters))
    return re.split(regexPattern, string, maxsplit)
