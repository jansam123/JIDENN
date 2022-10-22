from __future__ import annotations
from typing import Callable
import tensorflow as tf


def bracketed_split(string, delimiter, strip_brackets=False):
    """ Split a string by the delimiter unless it is inside brackets.
    e.g.
        list(bracketed_split('abc,(def,ghi),jkl', delimiter=',')) == ['abc', '(def,ghi)', 'jkl']
    """

    openers = '('
    closers = ')'
    opener_to_closer = dict(zip(openers, closers))
    opening_bracket = dict()
    current_string = ''
    depth = 0
    for c in string:
        if c in openers:
            depth += 1
            opening_bracket[depth] = c
            if strip_brackets and depth == 1:
                continue
        elif c in closers:
            assert depth > 0, f"You exited more brackets that we have entered in string {string}"
            assert c == opener_to_closer[opening_bracket[depth]
                                         ], f"Closing bracket {c} did not match opening bracket {opening_bracket[depth]} in string {string}"
            depth -= 1
            if strip_brackets and depth == 0:
                continue
        if depth == 0 and c == delimiter:
            yield current_string
            current_string = ''
        else:
            current_string += c
    assert depth == 0, f'You did not close all brackets in string {string}'
    yield current_string


class SimpleCut:
    def __init__(self, cut_repr: str) -> None:
        if cut_repr[0] == '(' and cut_repr[-1] == ')':
            self._cut_repr = cut_repr[1:-1]
        else:
            self._cut_repr = cut_repr

    def __str__(self) -> str:
        return self._cut_repr

    def __call__(self, sample: dict[str, tf.Tensor]) -> tf.Tensor:
        if '==' in self._cut_repr:
            key, value = self._cut_repr.split('==')
            return sample[key] == tf.cast(float(value), sample[key].dtype)
        elif '!=' in self._cut_repr:
            key, value = self._cut_repr.split('!=')
            return sample[key] != tf.cast(float(value), sample[key].dtype)
        elif '<' in self._cut_repr:
            key, value = self._cut_repr.split('<')
            return sample[key] < tf.cast(float(value), sample[key].dtype)
        elif '>' in self._cut_repr:
            key, value = self._cut_repr.split('>')
            return sample[key] > tf.cast(float(value), sample[key].dtype)
        elif '<=' in self._cut_repr:
            key, value = self._cut_repr.split('<=')
            return sample[key] <= tf.cast(float(value), sample[key].dtype)
        elif '>=' in self._cut_repr:
            key, value = self._cut_repr.split('>=')
            return sample[key] >= tf.cast(float(value), sample[key].dtype)
        else:
            raise ValueError(f"Cut {self._cut_repr} is not valid")


class Cut:
    def __init__(self, repr: str) -> None:
        self._repr = repr

    def __call__(self, sample: dict[str, tf.Tensor]) -> bool:
        return self._evaluate(self, sample)

    def _evaluate(self, cut: Cut, sample: dict[str, tf.Tensor]) -> bool:
        split = list(bracketed_split(cut._repr, delimiter=' '))
        if '&&' in split and '||' in split:
            raise ValueError(f"Cut {cut._repr} is not valid, use brackets to separate && and ||")

        if len(split) == 1:
            return SimpleCut(split[0])(sample)
        if '&&' in split:
            split[:] = (x for x in split if x != '&&')
            return tf.reduce_all([self._evaluate(Cut(subcut[1:-1]), sample) for subcut in split])
        elif '||' in split:
            split[:] = (x for x in split if x != '||')
            return tf.reduce_any([self._evaluate(Cut(subcut[1:-1]), sample) for subcut in split])

    def __str__(self) -> str:
        return self._repr

    def __and__(self, other: Cut) -> Cut:
        return Cut(f'({self._repr}) && ({other._repr})')

    def __or__(self, other: Cut) -> Cut:
        return Cut(f'({self._repr}) || ({other._repr})')

    def get_filter_function(self, dict_mapping: Callable[[tf.Tensor, tf.Tensor, tf.Tensor], dict[str, tf.Tensor]]) -> Callable[[tf.Tensor, tf.Tensor, tf.Tensor], bool]:
        
        @tf.function
        def dataset_filter(x: tf.Tensor, y: tf.Tensor, z: tf.Tensor) -> bool:
            return self(dict_mapping(x, y, z))
        
        return dataset_filter
