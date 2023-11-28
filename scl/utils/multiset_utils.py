from dataclasses import dataclass
from typing import Any
from collections import Counter

@dataclass
class MultiSetNode(object):
    left: Any = None
    right: Any = None
    value: Any = None
    size: int = 0

    @property
    def empty(self):
        return self.size == 0 or self.value is None

    @property
    def left_size(self):
        return self.left.size if self.left else 0

    @property
    def right_size(self):
        return self.right.size if self.right else 0

    @property
    def frequency(self):
        return self.size - self.left_size - self.right_size

    def clear(self):
        self.value = None
        self.left = None
        self.right = None
        self.size = 0

    def from_iterable(iterable):
        ret = MultiSetNode()
        for item in iterable:
            ret.insert(item)
        return ret

    def to_iterable(self):
        return list(self)

    def insert(self, item: Any):
        if self.empty:
            self.value = item

        if item < self.value:
            self.left = self.left or MultiSetNode()
            self.left.insert(item)
        elif item > self.value:
            self.right = self.right or MultiSetNode()
            self.right.insert(item)

        self.size += 1

    def remove(self, item: Any):
        if self.size == 1:
            self.value = None
            self.left = None
            self.right = None
            self.size = 0
            assert self.empty
            return

        if item < self.value:
            self.left.remove(item)
            if self.left.empty:
                self.left = None
        elif item > self.value:
            self.right.remove(item)
            if self.right.empty:
                self.right = None

        self.size -= 1

    '''
    Find cumulative (i.e. CDF) and non-cumulative (i.e. PDF) frequency of item
    '''
    def forward_lookup(self, item: Any):
        if self.empty:
            raise ValueError(f"Could not find {item} in empty multiset")

        if item < self.value:
            assert self.left
            return self.left.forward_lookup(item)
        elif item > self.value:
            assert self.right
            cumul_count, incidence = self.right.forward_lookup(item)
            return cumul_count + self.size - self.right_size, incidence
        else:
            cumul_count = self.left_size
            incidence = self.size - cumul_count - self.right_size
            return cumul_count, incidence

    '''
    Find symbol and its cumulative (i.e. CDF) and non-cumulative (i.e. PDF) frequency given its index
    '''
    def reverse_lookup(self, index: int):
        left_size = self.left_size
        current_frequency = self.frequency

        if index < left_size:
            return self.left.reverse_lookup(index)
        elif index >= left_size + current_frequency:
            non_right_size = self.size - self.right_size
            symbol, (cumul_count, incidence) = self.right.reverse_lookup(index - non_right_size)
            return symbol, (cumul_count + non_right_size, incidence)
        else:
            return self.value, (left_size, current_frequency)

    def map_values(self, mapper):
        if self.empty:
            return

        self.value = mapper(self.value)
        if self.left:
            self.left.map_values(mapper)
        if self.right:
            self.right.map_values(mapper)

    def clone(self):
        return MultiSetNode.from_iterable(self.to_iterable())

    def __repr__(self):
        return str(Counter(self.to_iterable()))

    def __iter__(self):
        if self.left:
            yield from self.left

        yield from [self.value] * self.frequency

        if self.right:
            yield from self.right

    def __len__(self):
        return self.size

    def __contains__(self, item):
        if self.empty:
            return False

        if item < self.value:
            if self.left is None:
                return False
            return item in self.left
        elif item > self.value:
            if self.right is None:
                return False
            return item in self.right
        else:
            return True

    def __eq__(self, other):
        return sorted(self.to_iterable()) == sorted(other.to_iterable())
