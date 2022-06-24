from rtree import index
from core.utils.misc import UndoMixin


class SearchTree(UndoMixin):
    def __init__(self, iterable=None, dim=2):
        super().__init__()
        self.dim = dim
        self.tree = index.Index(properties=index.Property(dimension=self.dim))

        # FIXED POINTS (NOT IN HISTORY)
        if iterable is not None:
            for idx, v in enumerate(iterable):
                self.tree.add(idx, v)

    def __getattr__(self, name):
        return getattr(self.tree, name)

    def __len__(self):
        return self.tree.get_size()

    def insert(self, id, coordinates):
        self.tree.insert(id, coordinates)
        self.add_revert(self.tree.delete, id, coordinates)

    def delete(self, id, coordinates):
        self.tree.delete(id, coordinates)
        self.add_revert(self.tree.insert, id, coordinates)

    add = insert
    remove = delete