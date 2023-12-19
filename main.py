# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=line-too-long
# pylint: disable=trailing-whitespace

from proposition import Proposition

#es = Proposition('&', Proposition('p'), Proposition('d'))
#print(es.string_repr()) #(p & q)

my_proposition = Proposition.render_tree('(( p & q ) | ~(q -> b))')

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