"""Three-number expression task, exact oracle, and binary reward.

No eval/exec, external data, model, or network call. The solver and reward parser
use different constructions: enumerated binary trees versus a restricted AST.
This is a deliberately narrow task, not a general mathematical parser.
"""
from __future__ import annotations

import ast
from collections import Counter
from dataclasses import asdict, dataclass
from fractions import Fraction
from itertools import permutations, product
import re
from typing import Sequence

TASK_VERSION = 'countdown3-v1'
MAX_RESPONSE_CHARS = 256
_ALLOWED_CHARS = re.compile(r'[0-9()+*/\- \t]+\Z', re.ASCII)


@dataclass(frozen=True)
class Check:
    format_valid: bool
    numbers_valid: bool
    target_equal: bool | None
    correct: bool
    reward: float
    reason: str
    numerator: int | None = None
    denominator: int | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def validate_problem(numbers: Sequence[int], target: int) -> None:
    if len(numbers) != 3 or any(type(x) is not int or not 1 <= x <= 20 for x in numbers):
        raise ValueError('Exactly three input integers in [1,20] are required')
    if type(target) is not int or not 1 <= target <= 100:
        raise ValueError('Target must be an integer in [1,100]')


def check_expression(text: str, numbers: Sequence[int], target: int) -> Check:
    """A single arithmetic expression is the ENTIRE answer.

    Only +,-,*,/, parentheses and the three supplied integer literals are allowed.
    Repeated supplied numbers must be used with the correct multiplicity.
    Intermediate negative/fractional values are allowed. Unary operators,
    powers, decimals, scientific notation, equations, fences and prose are not.
    Only outer whitespace is removed; no model answer is extracted or repaired.
    """
    validate_problem(numbers, target)
    if not isinstance(text, str):
        raise TypeError('text must be str')
    if len(text) > MAX_RESPONSE_CHARS:
        return Check(False, False, None, False, 0.0, 'too_long')
    expression = text.strip()
    if not expression or _ALLOWED_CHARS.fullmatch(expression) is None:
        return Check(False, False, None, False, 0.0, 'not_one_expression')
    try:
        tree = ast.parse(expression, mode='eval')
    except (SyntaxError, ValueError, RecursionError):
        return Check(False, False, None, False, 0.0, 'syntax_error')
    nodes = list(ast.walk(tree))
    if len(nodes) > 40:
        return Check(False, False, None, False, 0.0, 'expression_too_complex')
    leaves: list[int] = []

    def visit(node: ast.AST, depth: int = 0) -> Fraction:
        if depth > 10:
            raise ValueError('expression_too_complex')
        if isinstance(node, ast.Constant) and type(node.value) is int:
            literal = ast.get_source_segment(expression, node)
            if literal is None or re.fullmatch(r'[1-9][0-9]*|0', literal, re.ASCII) is None:
                raise ValueError('unsupported_literal')
            if not 0 <= node.value <= 100:
                raise ValueError('literal_out_of_range')
            leaves.append(node.value)
            return Fraction(node.value)
        if not isinstance(node, ast.BinOp) or type(node.op) not in (ast.Add, ast.Sub, ast.Mult, ast.Div):
            raise ValueError('unsupported_operation')
        left, right = visit(node.left, depth + 1), visit(node.right, depth + 1)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if right == 0:
            raise ZeroDivisionError
        return left / right

    try:
        value = visit(tree.body)
    except ZeroDivisionError:
        return Check(True, Counter(leaves) == Counter(numbers), None, False, 0.0, 'division_by_zero')
    except (ValueError, RecursionError) as exc:
        return Check(False, False, None, False, 0.0, str(exc))
    numbers_valid = Counter(leaves) == Counter(numbers)
    target_equal = value == Fraction(target)
    correct = numbers_valid and target_equal
    reason = 'correct' if correct else ('wrong_numbers' if not numbers_valid else 'wrong_value')
    return Check(True, numbers_valid, target_equal, correct, float(correct), reason,
                 value.numerator, value.denominator)


def _oracle_apply(a: Fraction, b: Fraction, operator: str) -> Fraction | None:
    if operator == '+':
        return a + b
    if operator == '-':
        return a - b
    if operator == '*':
        return a * b
    return a / b if b else None


def reachable_targets(numbers: Sequence[int], operators: str = '+-*/') -> dict[int, str]:
    """Enumerate both three-leaf binary-tree shapes and all leaf permutations.

    Return one deterministic witness for each integer target in [1,100]. This
    enumerates the exact task grammar, not an inference-model search or a dataset.
    """
    validate_problem(numbers, 1)
    if not operators or any(op not in '+-*/' for op in operators):
        raise ValueError('Unknown oracle operator')
    result: dict[int, str] = {}
    for a, b, c in sorted(set(permutations(numbers))):
        for op1, op2 in product(operators, repeat=2):
            av, bv, cv = Fraction(a), Fraction(b), Fraction(c)
            left = _oracle_apply(av, bv, op1)
            right = _oracle_apply(bv, cv, op2)
            candidates = []
            if left is not None:
                candidates.append((_oracle_apply(left, cv, op2), f'(({a}{op1}{b}){op2}{c})'))
            if right is not None:
                candidates.append((_oracle_apply(av, right, op1), f'({a}{op1}({b}{op2}{c}))'))
            for value, expression in candidates:
                if value is not None and value.denominator == 1 and 1 <= value <= 100:
                    target = int(value)
                    if target not in result or expression < result[target]:
                        result[target] = expression
    return dict(sorted(result.items()))


def make_messages(numbers: Sequence[int], target: int) -> list[dict[str, str]]:
    validate_problem(numbers, target)
    return [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': (
            f'Use the numbers {", ".join(map(str, numbers))} to make {target}. '
            'Use each supplied number exactly once, including repeated numbers. '
            'You may use only +, -, *, / and parentheses. Do not add other numbers. '
            'Negative or fractional intermediate results are allowed. '
            'Return only one arithmetic expression: no equals sign, explanation, or code fence.'
        )},
    ]
