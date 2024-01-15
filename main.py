# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=line-too-long
# pylint: disable=trailing-whitespace

from frozendict import frozendict
from proposition import Proposition
from predicate import Predicate, Model

#es = Proposition('&', Proposition('p'), Proposition('q'))
#print(es.string_repr()) #(p & q)
#((p & q) | (~q -> b)); ((a | ~b) -> c)
my_proposition = Proposition.parse_proposition('(~(a & ~b) & c)')
print(f"L'espressione inserita è: {my_proposition.string_repr()}")

#espressione equivalente ridotta in forma normale
eq_ex = my_proposition.get_equivalent_expression()
is_cnf, is_dnf = Proposition.parse_proposition(eq_ex).check_dnf_cnf()
print(f"L'espressione equivalente ridotta in {'cnf' if is_cnf else 'dnf' if is_dnf else 'forma normale' } è: {eq_ex}")

#doppio check se effettivamente sono equivalenti
print(my_proposition.check_equivalence(eq_ex))

#print della tabella di verità
my_proposition.print_truth_table()

#tautologia, contraddizione, soddisfacibile
t, c, s = my_proposition.is_tautology_contradiction_statisfiable()
R = 'tautologia' if t else 'contraddizione' if c else 'soddifacibile'
print('Questa proposizione è ' + R)

my_proposition.print_tree()

universe = {0, 1}
constant_interpretations={'a': 1, 'b': 0}
function_interpretations={
        'f': {
            (0, 0): 1,
            (0, 1): 1,
            (1, 0): 0,
            (1, 1): 0
        },
        'g': {
            (0, 0): 0,
            (0, 1): 1,
            (1, 0): 0,
            (1, 1): 1
        }
    }
relation_interpretations={
    'R': {
        (0, 0): False,
        (0, 1): True,
        (1, 0): True,
        (1, 1): False
    }
}
my_model = Model(universe, constant_interpretations, relation_interpretations, function_interpretations)

assignment = frozendict({
    'x': 1,
    'y': 0
})

f1 = Predicate.parse_predicate('Ex[(x = b & Ey[(y = g(x, 1) & f(x, y) = a)])]')
f2 = Predicate.parse_predicate('f(1, x)')
f3 = Predicate.parse_predicate('R(y, 0)')

formulas = [f1, f2, f3]
my_predicate = Predicate.parse_predicate('Ex[(x = b & Ey[(y = g(x, 1) & f(x, y) = a)])]')
print(my_predicate.string_repr())

result = my_model.evaluate_predicate(my_predicate, assignment)
print(result)
result = my_model.is_model_of(formulas, assignment)
print(result)
