# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=line-too-long
# pylint: disable=trailing-whitespace

from frozendict import frozendict
from proposition import Proposition
from predicate import Term, Predicate, Model

#es = Proposition('&', Proposition('p'), Proposition('d'))
#print(es.string_repr()) #(p & q)

my_proposition = Proposition.parse_proposition('(( p & q ) | ~(q -> b))')

#print della tabella di verità
my_proposition.print_truth_table()

#tautologia, contraddizione, soddisfacibile
t, c, s = my_proposition.is_tautology_contradiction_statisfiable()
R = 'tautologia' if t else 'contraddizione' if c else 'soddifacibile'
print('Questa proposizione è ' + R)

#equivalenza con un'altra espressione
E = '(p | ~ p)'
print(f"L'espressione {E} {'è' if my_proposition.check_equivalence(E) else 'non è'} equivalente")

my_proposition.print_tree()

my_model = Model(
    universe={0, 1},
    constant_interpretations={
        'a': 1,
        'b': 0
    },
    function_interpretations={
        'f': {
            (0, 0): 0,
            (0, 1): 1,
            (1, 0): 1,
            (1, 1): 0
        },
        'plus': {
            (0, 0): 1,
            (0, 1): 0,
            (1, 0): 0,
            (1, 1): 1
        },
        'g': {
            (0, 0): 1,
            (0, 1): 0,
            (1, 0): 0,
            (1, 1): 1
        },
        'times': {
            (0, 0): 0,
            (0, 1): 0,
            (1, 0): 0,
            (1, 1): 1
        }
    },
    relation_interpretations={
        'F': {
            (0, 0): False,
            (0, 1): True,
            (1, 0): True,
            (1, 1): False
        }
    }
)

my_predicate = Predicate.parse_predicate('Ex[(x = a & Ey[(y = times(x, 1) & f(x, y) = b)])]')

assignment = frozendict({
    'x': 1,
    'y': 0
})

#conversione da albero a stringa
print(my_predicate.string_repr())

my_term = Term.parse_term('f(g(b, 1), times(1, b))')
result = my_model.evaluate_term(my_term, assignment)
print(result)

result = my_model.evaluate_predicate(my_predicate, assignment)
print(result)
