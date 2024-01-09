# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=line-too-long
# pylint: disable=trailing-whitespace

from frozendict import frozendict
from proposition import Proposition
from predicate import Predicate, Model

#es = Proposition('&', Proposition('p'), Proposition('d'))
#print(es.string_repr()) #(p & q)
#((p | q) & (r | s))
my_proposition = Proposition.parse_proposition('((p & q) | ~(q -> b))')

#print della tabella di verità
my_proposition.print_truth_table()

result = my_proposition.check_dnf_cnf()
if result[0]:
    print("L'espressione è un CNF")
elif result[1]:
    print("L'espressione è un DNF")
else:
    print("L'espressione non è un CNF o un DNF")

#tautologia, contraddizione, soddisfacibile
t, c, s = my_proposition.is_tautology_contradiction_statisfiable()
R = 'tautologia' if t else 'contraddizione' if c else 'soddifacibile'
print('Questa proposizione è ' + R)

#equivalenza con un'altra espressione
E = '(p | ~ p)'
print(f"L'espressione {E} {'è' if my_proposition.check_equivalence(E) else 'non è'} equivalente")

my_proposition.print_tree()

universe = {0, 1}
constant_interpretations={'a': 1, 'b': 0}
function_interpretations={
        'f': {
            (0, 0): 0,
            (0, 1): 1,
            (1, 0): 1,
            (1, 1): 0
        },
        'g': {
            (0, 0): 0,
            (0, 1): 0,
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
