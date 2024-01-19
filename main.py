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
my_proposition = Proposition.parse_proposition('(~(a & ~b) & c)')
print(f"L'espressione inserita è: {my_proposition.string_repr()}")

#espressione equivalente ridotta in forma normale
eq_ex = my_proposition.get_equivalent_expression()
is_cnf, is_dnf = Proposition.parse_proposition(eq_ex).check_dnf_cnf()
print(f"L'espressione equivalente ridotta in {'cnf' if is_cnf else 'dnf' if is_dnf else 'forma normale' } è: {eq_ex}")

#print della tabella di verità
my_proposition.print_truth_table()

#tautologia, contraddizione, soddisfacibile
t, c, s = my_proposition.is_tautology_contradiction_statisfiable()
R = 'tautologia' if t else 'contraddizione' if c else 'soddifacibile'
print('Questa fbf è ' + R)

my_proposition.print_tree()

#dominio
universe = {0, 1}
#assegnazioni
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
        (0, 0): 0,
        (0, 1): 1,
        (1, 0): 1,
        (1, 1): 0
    }
}
#struttura
my_model = Model(universe, constant_interpretations, relation_interpretations, function_interpretations)

#ambiente
assignment = frozendict({
    'x': 1,
    'y': 0,
    'z': 0
})

my_predicate = Predicate.parse_predicate('Ex[(x = b & Ey[(y = g(x, z) & f(x, y) = a)])]')
print(my_predicate.string_repr())

#interpretazione
result = my_model.evaluate_predicate(my_predicate, assignment)
print(result)

f1 = Predicate.parse_predicate('f(1, x)')
f2 = Predicate.parse_predicate('R(y, 0)')

result = my_model.is_model_of([f1, f2], assignment)
print(result)
