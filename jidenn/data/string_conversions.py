"""
This module contains functions to convert strings to objects that can be evaluated on a sample, **without** using `eval`.
The two commonly operations are cuts and construction of variables from strings given some prior knowledge of the variables.
Data samples are represented as `Dict[str, tf.Tensor]` (alternatively `tf.Raggedtensor`) where the keys are the variable names and the values are the variable values.

The main advantage of this approach is that it is **safe** and can be **easily configured**.
"""
from __future__ import annotations
import tensorflow as tf
from typing import Dict, Iterable


def bracketed_split(string: str, delimiter: str = ',', strip_brackets: bool = True) -> Iterable[str]:
    """ Split a string by the delimiter unless it is inside brackets.
    original code from https://stackoverflow.com/questions/21662474/splitting-a-string-with-brackets-using-regular-expression-in-python

    Example:
    ```python
    for s in bracketed_split('abc,(def,ghi),jkl', delimiter=','):
        print(s)
    #'abc'
    #'(def,ghi)'
    #'jkl'
    ``` 

    Args:
        string(str): The string to split.
        delimiter(str): The delimiter to split by.
        strip_brackets(bool): If `True`, the brackets are stripped from the output strings.

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
    """A simple cut that can be evaluated on a sample. 
    It is initialized with a string representation of the cut, eg. 'jets_pt > 50_000'.
    The **allowed operators** are `==`, `!=`, `>`, `<`, `>=`, `<=`.
    There **must not be any spaces** in the string, next to the operators.

    When the class instance is called on a sample `Dict[str, tf.Tensor]`, it returns a boolean tensor that is `True` 
    if the given variable cut is passed and `False` otherwise.
    The boolean is returned as a `tf.Tensor`.

    Example:
    ```python
    cut = SimpleCut('jets_pt > 50_000')
    data_sample = {'jets_pt': tf.constant([100_000]), 'jets_eta': tf.constant([2.0])}
    data_sample2 = {'jets_pt': tf.constant([10_000]), 'jets_eta': tf.constant([0.1])}
    cut(data_sample) # True
    cut(data_sample2) # False
    ```

    Args:
        cut_repr (str): The string representation of the cut.
    """
    _ordered_signs = ['<=', '>=', '==', '!=', '<', '>']
    _sign_mapping = {'==': tf.equal, '!=': tf.not_equal, '>': tf.greater,
                     '<': tf.less, '>=': tf.greater_equal, '<=': tf.less_equal}

    def __init__(self, cut_repr: str) -> None:
        if cut_repr[0] == '(' and cut_repr[-1] == ')':
            self._cut_repr = cut_repr[1:-1]
        else:
            self._cut_repr = cut_repr

    def __str__(self) -> str:
        return self._cut_repr

    def __call__(self, sample: Dict[str, tf.Tensor]) -> tf.Tensor:
        """Evaluate the cut on a sample. If the cut is passed, the returned tensor is `True`, otherwise it is `False`.
        The behaviour is undefined if the cut is not valid, ie. if the variable name is not in the sample.

        Args:
            sample (Dict[str, tf.Tensor]): The sample to evaluate the cut on.

        Returns:
            tf.Tensor: A boolean tensor that is `True` if the cut is passed and `False` otherwise.
        """
        for sign in self._ordered_signs:
            if sign in self._cut_repr:
                left, right = self._cut_repr.split(sign)
                left = left.strip()
                right = right.strip()
                if left in sample:
                    left = sample[left]
                else:
                    left = tf.constant(float(left))
                if right in sample:
                    right = sample[right]
                else:
                    right = tf.cast(float(right), left.dtype)
                return tf.reduce_all(self._sign_mapping[sign](left, right))


class Cut:
    """Class representing a composite cut that can be evaluated on a sample `Dict[str, tf.Tensor]`.
    The cut is initialized with a string representation of the cut, eg. '(jets_pt > 50_000) && (jets_eta < 2.0)'.
    They are a composition of `SimpleCut`s and the **allowed operators** are `&&` and `||` .
    `SimpleCut` **must be enclosed in brackets** and composition operators must be **separated by spaces** from the `SimpleCut`s.
    There **must not be any space** next to the `==`, `!=`, `>`, `<`, `>=`, `<=` operators.

    Example:
    ```python
    cut = Cut('(jets_pt > 50_000) && (jets_eta < 1.5)')
    data_sample = {'jets_pt': tf.constant([100_000]), 'jets_eta': tf.constant([2.0])}
    data_sample2 = {'jets_pt': tf.constant([10_000]), 'jets_eta': tf.constant([0.1])}
    cut(data_sample) # False
    cut(data_sample2) # False
    ```

    Two `Cut`s can be combined with the `&` and `|` operators. The `&` operator is equivalent to `&&` and the `|` operator is equivalent to `||`.

    Example:
    ```python
    cut = Cut('(jets_pt > 50_000) && (jets_eta < 1.5)')
    cut2 = cut & Cut('(jets_pt < 100_000)')
    cut3 = cut | Cut('(jets_eta > -1.5)')
    ```

    Args:
        cut_repr (str): The string representation of the cut.
    """

    def __init__(self, cut_repr: str) -> None:
        self._cut_repr = cut_repr

    def __call__(self, sample: Dict[str, tf.Tensor]) -> tf.Tensor:
        """Evaluate the cut on a sample. If the cut is passed, the returned tensor is `True`, otherwise it is `False`.
        The behaviour is undefined if the cut is not valid, ie. if the variable name is not in the sample.

        Args:
            sample (Dict[str, tf.Tensor]): The sample to evaluate the cut. 

        Returns:
            bool: `True` if the cut is passed and `False` otherwise as a `tf.Tensor`.
        """
        return self._evaluate(self, sample)

    def _evaluate(self, cut: Cut, sample: Dict[str, tf.Tensor]) -> tf.Tensor:
        split = list(bracketed_split(cut._cut_repr, delimiter=' ', strip_brackets=True))
        if '&&' in split and '||' in split:
            raise ValueError(f"Cut {cut._cut_repr} is not valid, use brackets to separate && and ||")

        if len(split) == 1:
            return SimpleCut(split[0])(sample)
        if '&&' in split:
            split[:] = (x for x in split if x != '&&')
            return tf.reduce_all([self._evaluate(Cut(subcut), sample) for subcut in split])
        elif '||' in split:
            split[:] = (x for x in split if x != '||')
            return tf.reduce_any([self._evaluate(Cut(subcut), sample) for subcut in split])

    def __str__(self) -> str:
        return self._cut_repr

    def __and__(self, other: Cut) -> Cut:
        return Cut(f'({self._cut_repr}) && ({other._cut_repr})')

    def __or__(self, other: Cut) -> Cut:
        return Cut(f'({self._cut_repr}) || ({other._cut_repr})')


class SimpleExpression:
    """
    .. caution::
        Might not work properly!
    Class representing a simple expression that can be evaluated on a sample `Dict[str, tf.Tensor]`.
    Simple expression is a slice of a tensor in the sample, eg. `jets_pt[0]`.
    The class instance can be called on a sample to evaluate the expression.

    Example:
    ```python
    data_sample = {'jets_pt': tf.constant([100_000, 50_000])}
    expression = SimpleExpression('jets_pt[0]')
    expression(data_sample) # 100_000
    ```
    Args:
        expression (str): The string representation of the expression.
    """

    def __init__(self, expression: str):
        expression = expression.strip()
        self._expression = expression
        if '[' in expression and ']' in expression:
            self._var = expression.split('[')[0]
            self._slice = self._parse_slice(expression.split('[')[1][:-1])
        else:
            self._var = expression
            self._slice = None

    def __call__(self, sample: Dict[str, tf.Tensor]) -> tf.Tensor:
        """Evaluate the expression on a sample. The behaviour is undefined if the variable name is not in the sample.

        Args:
            sample (Dict[str, tf.Tensor]): The sample to evaluate the expression.

        Raises:
            ValueError: If the expression is not found in the sample and cannot be cast to float.

        Returns:
            tf.Tensor: The sliced tensor.
        """
        if self._var not in sample:
            try:
                return float(self._var)
            except ValueError:
                raise ValueError(f'Expression {self._expression} not found in sample and cannot be cast to float')
        if self._slice is None:
            return sample[self._var]
        else:
            tensor = sample[self._var]
            old_rank = tensor.shape.rank
            sliced = tensor[self._slice]
            new_rank = sliced.shape.rank
            if old_rank == new_rank:
                return tf.squeeze(sliced, axis=-1)
            else:
                return sliced

    def __str__(self):
        return self._expression

    def _pick_var(self, sample, var):
        var_split = var.split('[')
        if len(var_split) == 1:
            return sample[var]
        var_name = var_split[0]
        var_slice = self._parse_slice(var_split[1][:-1]) if '[' in var else None
        if sample[var_name].shape.rank == 1:
            return sample[var_name][var_slice]
        return tf.squeeze(sample[var_name][var_slice], axis=-1)

    def _parse_slice(self, slice_str: str):
        return tuple((slice(*(int(i) if i else None for i in part.strip().split(':'))) if ':' in part else int(part.strip())) for part in slice_str.split(','))


def _add(a, b):
    """Add two numbers"""
    return a + b


def _sub(a, b):
    """Subtract two numbers"""
    return a - b


def _multiply(a, b):
    """Multiply two numbers"""
    return a * b


def _divide(a, b):
    """Divide two numbers"""
    return a / b


class Expression:
    """
    .. caution::
        Might not work properly!
    An expression that can be evaluated on a sample `Dict[str, tf.Tensor]`.
    It can be used to create new variables in the sample, eg. `jets_pt[0] + jets_pt[1]`.
    The allowed operations are `+`, `-`, `*`, `/`. The opeartions can involve numbers, eg. `jets_pt[0]/1000`. 

    It is mainly use to use only some entries of a multidimensional tensor, eg. `jets_pt[0]` to use only the leading jet. 

    The name of the new variable is the same as the string represtation of the expression.

    Example:
    ``` python
    data_sample = {'jets_pt': tf.constant([100_000, 50_000])}
    expression = Expression('jets_pt[0] + jets_pt[1]')
    expression(data_sample) # 150_000
    ```
    Args:
        expression (str): The string representation of the expression.
    """

    _operation_mapping = {'+': _add, '-': _sub, '*': _multiply, '/': _divide}

    def __init__(self, expression):
        self._expression = expression

    @tf.function
    def __call__(self, sample: Dict[str, tf.Tensor]) -> tf.Tensor:
        """Evaluate the expression on a sample. The behaviour is undefined if the variable name is not in the sample.
        The method is decorated with `tf.function` to speed up the evaluation during training.

        Args:
            sample (Dict[str, tf.Tensor]): The sample to evaluate the expression.

        Returns:
            tf.Tensor: The sliced tensor.
        """
        return self._evaluate(self._expression, sample)

    def __str__(self):
        return self._expression

    def _evaluate(self, expression: str, sample):
        for operation_rep, operation in self._operation_mapping.items():
            if operation_rep in expression:
                left = self._evaluate(expression.split(operation_rep, 1)[0], sample)
                right = self._evaluate(expression.split(operation_rep, 1)[1], sample)
                try:
                    return operation(left, right)
                except TypeError:
                    return operation(float(left), float(right))

        return SimpleExpression(expression)(sample)


__pdoc__ = {'SimpleCut.__call__': True, 'Cut.__call__': True,
            'Expression.__call__': True, 'SimpleExpression.__call__': True}
