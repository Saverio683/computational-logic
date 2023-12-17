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
print(t,c,s)

#equivalenza con un'altra espressione
print(my_proposition.check_equivalence('(p | ~ p)'))
