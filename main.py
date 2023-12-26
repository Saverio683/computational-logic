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

my_term = Term.parse_term('times(a, plus(a, 1))')
print(my_term.string_repr())

print()

my_predicate = Predicate.parse_predicate('Ex[(x = b & Ey[(y = plus(x, 1) & times(x, y) = a)])]')
print(my_predicate.string_repr())

print()

my_model = Model(
    universe={0, 1},
    constant_interpretations={
        'a': 0,
        '1': 1,
        'b': 0
    },
    function_interpretations={
        'f': {
            (0, 0): 0,
            (0, 1): 1,
            (1, 0): 1,
            (1, 1): 0,
        },
        'plus': {
            (0,0): 1,
            (0,1): 0,
            (1,0): 0,
            (1,1): 1
        },
        'g': {
            (0,0): 1,
            (0,1): 0,
            (1,0): 0,
            (1,1): 1
        },
        'times': {
            (0, 0): 0,
            (0, 1): 0,
            (1, 0): 0,
            (1, 1): 1,
        },
    },
    relation_interpretations={}
)

assignment = frozendict()  # Nessuna variabile, quindi l'assegnazione è vuota
result = my_model.evaluate_term(my_term, assignment)
print(result)

my_term = Term.parse_term('f(g(b, 1), times(1, b))')
result = my_model.evaluate_term(my_term, assignment)
print(result)

print(my_model.evaluate_predicate(Predicate.parse_predicate('Ex[(F(x) -> G(y, 4))]')))
