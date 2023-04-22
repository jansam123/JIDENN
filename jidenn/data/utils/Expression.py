import tensorflow as tf
from typing import Dict, Union


class SimpleExpression:

    def __init__(self, expression: str):
        expression = expression.strip()
        self._expression = expression
        if '[' in expression and ']' in expression:
            self._var = expression.split('[')[0]
            self._slice = self.parse_slice(expression.split('[')[1][:-1])
        else:
            self._var = expression
            self._slice = None
        
        

    def __call__(self, sample: Dict[str, tf.Tensor]) -> tf.Tensor:
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
        var_slice = self.parse_slice(var_split[1][:-1]) if '[' in var else None
        if sample[var_name].shape.rank == 1:
            return sample[var_name][var_slice]
        return tf.squeeze(sample[var_name][var_slice], axis=-1)

    def parse_slice(self, slice_str: str):
        return tuple((slice(*(int(i) if i else None for i in part.strip().split(':'))) if ':' in part else int(part.strip())) for part in slice_str.split(','))

def add(a, b):
    return a + b
def sub(a, b):
    return a - b
def multiply(a, b):
    return a * b
def divide(a, b):
    return a / b

class Expression:
    
    operation_mapping = {'+': add, '-': sub, '*': multiply, '/': divide}
        
    def __init__(self, expression):
        self._expression = expression

    @tf.function
    def __call__(self, sample: Dict[str, tf.Tensor]) -> tf.Tensor:
        return self._evaluate(self._expression, sample)

    def __str__(self):
        return self._expression

    def _evaluate(self, expression: str, sample):
        for operation_rep, operation in self.operation_mapping.items():
            if operation_rep in expression:
                left = self._evaluate(expression.split(operation_rep, 1)[0], sample)
                right = self._evaluate(expression.split(operation_rep, 1)[1], sample)
                try:
                    return operation(left, right)
                except TypeError:
                    return operation(float(left), float(right))
                    
        return SimpleExpression(expression)(sample)


def main():
    sample = {'a': tf.ragged.constant([[10., 2., 3], [2., 3]]), 'b': tf.constant([2.,2.]), 'c': tf.constant([3,3])}
    print(float(tf.constant([2])))
    expr = Expression('c*0.001-b')
    print(expr(sample))


if __name__ == '__main__':
    main()
