# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=line-too-long
# pylint: disable=trailing-whitespace
# pylint: disable=pointless-string-statement

from typing import AbstractSet, FrozenSet, Generic, List, Mapping, Optional, Sequence, Set, Tuple, TypeVar, Union
from dataclasses import dataclass
import re
from frozendict import frozendict

def is_constant(string: str) -> bool:
    #le costanti sono lettere minuscole che vanno dalla 'a' alla 'e' o numeri
    return  string >= 'a' and string <= 'e' or string.isdigit()

def is_variable(string: str) -> bool:
    #le variabili sono rappresentate da lettere minuscole dalla 'u' alla 'z' 
    return string >= 'u' and string <= 'z'

def is_function(string: str) -> bool:
    #una funzione ha come prima lettera una minuscola che va dalla 'f' alla 't'
    return string[0] >= 'f' and string[0] <= 't' 

def is_equality(string: str) -> bool:
    #verifica se la stringa è l'operatore di uguaglianza (=)
    return string == '='

def is_relation(string: str) -> bool:
    #verifica se la stringa è il nome di una relazione
    #la stringa inizia con una maiuscola compresa tra 'F' e 'T'
    return bool(string) and string[0] >= 'F' and string[0] <= 'T'

def is_unary(string: str) -> bool:
    #verifico se è un operatore unario (NOT)
    return string == '~'

def is_binary(string: str) -> bool:
    #in caso di AND e OR e IMPLICA
    return string == '&' or string == '|' or string == '->'

def is_quantifier(string: str) -> bool:
    #'A' rappresenta il 'per ogni', mentre 'E' indica 'esiste'
    return string == 'A' or string == 'E'

@dataclass
class Term:
    #Un albero i cui nodi possono essere variabili, costanti e funzioni
    root: str
    #Gli argomenti sono presenti SOLO SE il nodo è una funzione
    arguments: Optional[Tuple['Term', ...]] 

    def __init__(self, root: str, arguments: Optional[Tuple['Term']] = None):
        if is_function(root): #verifico se il nodo è una funzione e nel caso verifico se possiede gli argomenti
            assert arguments is not None and len(arguments) > 0, 'Function requires arguments.'
            self.root = root
            self.arguments = tuple(arguments) if arguments else None
        elif is_constant(root) or is_variable(root): #invece variabili e costanti non devono avere argomenti
            assert arguments is None, 'Constants and variables cannot have arguments.'
            self.root = root
            self.arguments = None   
        else:
            raise ValueError(f'Invalid string: {root}')
        
    #conversione da albero a stringa
    def __get_string__(self, node: 'Term') -> str:
        if is_function(node.root):
            #se è una funzione, si chiama il metodo in maniera ricorsiva per ogni suo argomento, separandoli da virgole
            args = ', '.join(self.__get_string__(arg) for arg in node.arguments)
            return f'{node.root}({args})'
        #se invece è una variabile/costante, si ritorna il suo valore
        return node.root

    def string_repr(self, argument: Optional['Term'] = None) -> str:
        if argument:
            return self.__get_string__(argument)
        return self.__get_string__(self)
    
    #converto una stringa nel corrispettivo albero/oggetto
    @staticmethod
    def parse_term(string: str) -> 'Term':
        string = string.replace(' ', '')
        if is_function(string):
            '''
                nel caso di funzione, ad es "f(x,y)", root è "f" che è l'elemento a sinistra della parentesi "("
                poi si verifica che le parentesi siano chiuse correttamente
                poi si invoca la funzione ricorsiva sugli argomenti che sono separati da virgole
            '''
            root, _, arguments = string.partition('(')
            assert arguments[-1] == ')', f'Missing closing bracked in string: {arguments}'
            arguments = re.split(r',(?![^()]*\))', arguments[:-1])
            args = []
            for arg in tuple(arguments):
                args.append(Term.parse_term(arg))
            return Term(root, tuple(args))
        elif is_constant(string[0]) or is_variable(string[0]):            
            if len(string) == 1:
                return Term(string)
            raise ValueError(f'Invalid string after constant/variable: {string[1:]}')
        else:
            raise ValueError(f'Invalid term: {string}')

    def get_constants(self) -> Set[str]:
        constants_set = set()

        def collect_constants(node: Term):
            if is_constant(node.root):
                constants_set.add(node.root)
            elif is_function(node.root):
                for arg in node.arguments:
                    collect_constants(arg)

        collect_constants(self)
        return constants_set

    def get_variables(self) -> Set[str]:
        variables_set = set()

        def collect_variables(node: Term):
            if is_variable(node.root):
                variables_set.add(node.root)
            elif is_function(node.root):
                for arg in node.arguments:
                    collect_variables(arg)

        collect_variables(self)
        return variables_set

    def get_functions(self) -> Set[Tuple[str, int]]:
        functions_set = set()

        def collect_functions(node: Term):
            if is_function(node.root):
                #l'arietà di una funzione è il numero dei suoi argomenti, questo valore servirà per i modelli
                arity = len(node.arguments) if node.arguments else 0
                functions_set.add((node.root, arity))
                for arg in node.arguments:
                    collect_functions(arg)

        collect_functions(self)
        return functions_set
@dataclass
class Predicate:
    '''
        'root' è il nodo dell'albero può essere una relazione/uguaglianza/operatore/quantificatore
        'arguments' indica gli argomenti di root se è una relazione o un'uguaglianza, sennò è None
        'left' e 'right' sono i sottonodi, nel caso se root è un operatore unario/binario, sennò sono None
        'variable' e 'statement' vengono assegnati se root è un quantificatore, sennò sono None
    '''
    root: str
    arguments: Optional[Tuple[Term, ...]]
    left: Optional['Predicate']
    right: Optional['Predicate']
    variable: Optional[str]
    statement: Optional['Predicate']

    def __init__(self, root: str, arguments_or_left_or_variable: Union[Sequence[Term], 'Predicate', str], 
                right_or_statement: Optional['Predicate'] = None):
        if root == '':
            self.root, self.arguments = '', None
        elif is_equality(root) or is_relation(root):
            #verifico se arguments_or_left_or_variable È una sequenza e NON una stringa
            assert isinstance(arguments_or_left_or_variable, Sequence) and \
            not isinstance(arguments_or_left_or_variable, str)
            if is_equality(root):
                #Verifico se gli argomenti sono esattamente 2, in caso di operatore di uguaglianza =
                assert len(arguments_or_left_or_variable) == 2
            assert right_or_statement is None
            self.root, self.arguments = root, tuple(arguments_or_left_or_variable)
        elif is_unary(root):
            # Verifico che solo il figlio sinistro esiste in caso di ~
            assert isinstance(arguments_or_left_or_variable, Predicate)
            assert right_or_statement is None
            self.root, self.first = root, arguments_or_left_or_variable
        elif is_binary(root):
            # Verifico che sia left che right esistano
            assert isinstance(arguments_or_left_or_variable, Predicate)
            assert right_or_statement is not None
            self.root, self.first, self.second = \
            root, arguments_or_left_or_variable, right_or_statement
        else:
            assert is_quantifier(root)
            # Verifico che 'arguments' sia una variabile
            assert isinstance(arguments_or_left_or_variable, str) and \
            is_variable(arguments_or_left_or_variable)
            assert right_or_statement is not None
            self.root, self.variable, self.statement = \
            root, arguments_or_left_or_variable, right_or_statement

    #conversione dell'albero in stringa
    @staticmethod
    def __parse_predicate_tree__(node: 'Predicate') -> str:
        if is_binary(node.root):
            '''
                in caso di operatore binario, so per certezza che node.first è di tipo Predicate,
                invece per node.second so solo che non è None.
                Quindi devo vedere se è di tipo Term, Predicate o stringa
            '''
            second = ''
            if isinstance(node.second, Term):
                second = Term.string_repr(node.second)
            elif isinstance(node.second, Predicate):
                second = node.__parse_predicate_tree__(node.second)
            elif isinstance(node.second, str):
                second = node.second
            return f'({node.__parse_predicate_tree__(node.first)} {node.root} {second})'
        elif is_unary(node.root):
            return f'{node.root} {node.__parse_predicate_tree__(node.first)}'
        elif is_quantifier(node.root):
            #stesso procedimento fatto nel caso di operatore binario con node.second
            statement = ''
            if isinstance(node.statement, Term):
                statement = Term.string_repr(node.statement)
            elif isinstance(node.statement, Predicate):
                statement = node.__parse_predicate_tree__(node.statement)
            elif isinstance(node.statement, str):
                statement = node.statement
            return f'{node.root}{node.variable}[{statement}]'
        elif is_relation(node.root):
            args = ', '.join(Term.string_repr(arg) for arg in node.arguments)
            return f'{node.root}({args})'
        else:
            #caso di operatore di uguaglianza, dove gli argomenti sono esattamente 2
            return f'{Term.string_repr(node.arguments[0], argument=node.arguments[0])} {node.root} {Term.string_repr(node.arguments[1], argument=node.arguments[1])}'

    def string_repr(self) -> str:
        return f'{Predicate.__parse_predicate_tree__(self)}'
    
    #da stringa ad albero
    @staticmethod
    def parse_predicate(string: str) -> 'Predicate':  
        #rimuovo gli spazi bianchi
        string = string.replace(' ', '')

        def find_outer_operator(string: str) -> list[int | None, bool]:
            """
                Questo metodo trova l'operatore binario più esterno nella stringa, 
                tenendo conto delle precedenze delle parentesi.
                Ritorna l'indice dell'operatore e True se esso è '->', sennò false.
                Serve sapere se l'operatore è in particolare '->' perché è composto da due caratteri
                ad es. "(a & b) | c" ritorna l'indice dell'OR e false
            """
            open_count = 0
            stack = []
            for i, char in enumerate(string):
                if char == '(':
                    open_count += 1
                elif char == ')':
                    open_count -= 1
                elif char in ['&', '|']:
                    stack.append([i, open_count, False])
                elif i + 1 < len(string) and char + string[i + 1] == '->':
                    stack.append([i, open_count, True])

            if not stack: #nessun operatore trovato
                return None, False

            min_value = float('inf')# Inizializza con un valore molto grande
            min_element = None

            for el in stack:
                if el[1] < min_value:
                    min_value = el[1]
                    min_element = el

            if min_element[2]:
                return min_element[0], True
            return min_element[0], False

        if string == '':
            raise ValueError('Invalid empty string')

        if string.endswith(']'): #se finisce con ], allora si tratta di quantificatore
            for quantifier in ['A', 'E']:
                if quantifier in string:
                    _, root, right = string.partition(quantifier)
                    #verifico che le parentesi sono state aperte correttamente
                    assert right[1] == '[' and right.endswith(']') 
                    variable = right[0]
                    assert is_variable(variable), f'Invalid variable afer {root}: {right[0]}'
                    return Predicate(root, variable, Predicate.parse_predicate(right[2:-1]))

        left, root, right = '', '', ''

        #caso di operatore binario
        if string.startswith('(') and string.endswith(')') and re.search(r'&|\||->', string):
            string = string[1:-1] #rimuovo le parentesi esterne
            index, is_implica  = find_outer_operator(string)
            assert index is not None
            left = string[0:index]
            root_length = 2 if is_implica else 1
            root = string[index:index + root_length]
            right = string[index + root_length:]
            return Predicate(root, Predicate.parse_predicate(left), Predicate.parse_predicate(right))

        if '~' in string:
            _, root, left = string.partition('~')
            return Predicate(root, Predicate.parse_predicate(left))
        if '=' in string:
            left, root, right = string.partition('=')
            return Predicate(root, [Term.parse_term(left), Term.parse_term(right)])
        if is_relation(string):
            root, _, right = string.partition('(')
            args = []
            for arg in re.split(r',(?![^()]*\))', right[:-1]):
                args.append(Term.parse_term(arg))
            return Predicate(root, args)
        return Term.parse_term(string)

    def get_all(self) -> Tuple[Set, Set, Set, Set, Set]:
        '''
            Ritorna:
            -costanti
            -variabili
            -variabili legate
            -funzioni
            -relazioni
        '''
        constants = set()
        functions = set()
        relations = set()
        variables = set()
        bound_variables = set()

        def find_everything(node: Predicate | Term) -> Set:
            if isinstance(node, Predicate):
                if is_relation(node.root) or is_equality(node.root):
                    if is_relation(node.root):
                        relations.add(node.root)
                    for arg in node.arguments:
                        find_everything(arg)
                elif is_quantifier(node.root):
                    bound_variables.add(node.variable)
                    find_everything(node.statement)
                elif is_unary(node.root):
                    find_everything(node.first)
                else:
                    find_everything(node.first)
                    find_everything(node.second)
            else:
                if is_function(node.root):
                    arity = len(node.arguments) if node.arguments else 0
                    functions.add((node.root, arity))
                    for arg in node.arguments:
                        find_everything(arg)
                elif is_constant(node.root):
                    constants.add(node.root)
                else:
                    variables.add(node.root)
        find_everything(self)

        return (
            constants,
            variables,
            bound_variables,
            functions,
            relations
        )

#f(g(x),3)
#my_term = Term('sum', [Term('g', [Term('x')]), Term('3')])

#‘(Ex[f(g(x),3)=y]->GT(y,4))’
""" 
my_predicate = Predicate('->',
    Predicate('E', 'x',
        Predicate('=', [my_term, Term('y')])
    ),
    Predicate('GT', [Term('y'), Term('4')])
) """


T = TypeVar('T')
class Model(Generic[T]):
    """
    Classe del modello per i costrutti della logica dei predicati
    ATTRIBUTI:
    universe: il set di elementi in cui i termini possono essere valutati
    e su cui sono definite le quantificazioni
    constant_interpretations: mappatura da ogni nome di costante al suo valore
    relation_arities: mappatura da ogni nome di relazione alla sua arità, se la relazione
    è vuota allora l'arietà sarà -1.
    relation_interpretations: mappatura da ogni nome di relazione ai suoi rispettivi output.
    Lo stesso viene ripetuto per function_arities function_interpretations.
    """
    universe: FrozenSet[T]
    constant_interpretations: Mapping[str, T]
    relation_arities: Mapping[str, int]
    relation_interpretations: Mapping[str, AbstractSet[Tuple[T, ...]]]
    function_arities: Mapping[str, int]
    function_interpretations: Mapping[str, Mapping[Tuple[T, ...], T]]
    def __init__(self, universe: AbstractSet[T],
        constant_interpretations: Mapping[str, T],
        relation_interpretations: Mapping[str, AbstractSet[Tuple[T, ...]]],
        function_interpretations: Mapping[str, Mapping[Tuple[T, ...], T]] = frozendict()):
        for el in universe:
            '''
            per ogni elemento dell'universo, associo a esso la corrispettiva costante
            '''
            constant_interpretations[str(el)] = el
        for constant in constant_interpretations:
            '''
            verifico il corretto formato delle costanti e controllo se il loro valore 
            appartiene all'universo
            '''
            assert is_constant(constant)
            assert constant_interpretations[constant] in universe
        relation_arities = {}
        for relation in relation_interpretations:
            assert is_relation(relation)
            relation_interpretation = relation_interpretations[relation]
            if len(relation_interpretation) == 0:
                arity = -1
            else:
                '''
                some_arguments contiene il primo elemento di relation_interpretation,
                some_arguments è la tupla contenente i valori dei parametri della relazione
                questa variabile mi serve per calcolare l'arietà della relazione
                '''
                some_arguments = next(iter(relation_interpretation)) 
                arity = len(some_arguments) #se ad esempio some_arguments è (0,0), l'arietà è 2
                for arguments in relation_interpretation:
                    assert len(arguments) == arity
                    for argument in arguments:
                        assert argument in universe #check se l'argomento esiste nell'insieme universo
            relation_arities[relation] = arity #salvo l'arietà
        function_arities = {}
        for function in function_interpretations:
            '''
            il funzionamento è analogo per relation_interpretations
            '''
            assert is_function(function)
            function_interpretation = function_interpretations[function]
            assert len(function_interpretation) > 0
            some_argument = next(iter(function_interpretation))
            arity = len(some_argument)
            assert arity > 0
            assert len(function_interpretation) == len(universe)**arity
            for arguments in function_interpretation:
                assert len(arguments) == arity
                for argument in arguments:
                    assert argument in universe
                assert function_interpretation[arguments] in universe
            function_arities[function] = arity
        '''
        Uso i tipi frozenset e frozendict per rendere gli attributi del modello immutabili
        '''
        self.universe = frozenset(universe)
        self.constant_interpretations = frozendict(constant_interpretations)
        self.relation_arities = frozendict(relation_arities)
        self.relation_interpretations = \
            frozendict({relation: frozenset(relation_interpretations[relation])
            for relation in relation_interpretations})
        self.function_arities = frozendict(function_arities)
        self.function_interpretations = \
            frozendict({function: frozendict(function_interpretations[function])
            for function in function_interpretations})

    def evaluate_term(self, term: Term, assignment: Mapping[str, T] = frozendict()) -> T:
        """
        Calcolo i valori di verità del termine con l'assegnazione dei nomi delle variabili
        """
        # Verifico se le costanti nel termine sono nelle interpretazioni del modello
        assert term.get_constants().issubset(self.constant_interpretations.keys())
        # Verifico se le variabili del termine sono nell'assegnazione
        assert term.get_variables().issubset(assignment.keys())
        
        def evaluate_node(node: Term) -> T:
            if is_constant(node.root):
                return self.constant_interpretations[node.root]
            elif is_variable(node.root):
                return assignment[node.root]
            elif is_function(node.root):
                function_args = tuple(evaluate_node(arg) for arg in node.arguments)
                return self.function_interpretations[node.root][function_args]
            else:
                raise ValueError(f"Invalid node root: {node.root}")
        
        return evaluate_node(term)
    
    def evaluate_predicate(self, predicate: Predicate, assignment: Mapping[str, T] = frozendict()) -> bool:
        """
        Calcolo il valore di verità del predicato dato nel modello attuale sotto l'assegnazione data.
        """
        if is_equality(predicate.root):
            value1 = self.evaluate_term(predicate.arguments[0], assignment)
            value2 = self.evaluate_term(predicate.arguments[1], assignment)
            return value1 == value2

        elif is_relation(predicate.root):
            tuple_values = [self.evaluate_term(arg, assignment) for arg in predicate.arguments]
            return tuple(tuple_values) in self.relation_interpretations[predicate.root]

        elif is_unary(predicate.root):
            value = self.evaluate_predicate(predicate.first, assignment)
            return value

        elif is_binary(predicate.root):
            left_value = self.evaluate_predicate(predicate.first, assignment)
            right_value = self.evaluate_predicate(predicate.second, assignment)
            if predicate.root == '&':
                return left_value and right_value
            elif predicate.root == '|':
                return left_value or right_value
            else:  # implica
                return not left_value or right_value

        elif is_quantifier(predicate.root):
            variable = predicate.variable
            for element in self.universe:
                new_assignment = dict(assignment)
                new_assignment[variable] = element
                if self.evaluate_predicate(predicate.statement, new_assignment):
                    return True
            return False

        else:
            raise ValueError(f'Invalid predicate: {predicate.root}')
        
    def is_model_of(self, formulas: List[Predicate], assignment: frozendict[str, int]) -> bool:
        """
        Verifica se il modello è un modello di ciascuna delle formule fornite.

        Parametri:
        - formulas: un set di formule da verificare

        Ritorna:
        - True se ogni formula è vera nel modello, sennò False
        """
        for formula in formulas:
            #Check delle costanti, così come per funzioni e relazioni
            if isinstance(formula, Term):
                constants = formula.get_constants()
                functions = formula.get_functions()
                relations = ()
            else:
                (constants, _, __, functions, relations) = formula.get_all()
            assert constants.issubset(self.constant_interpretations.keys())
            
            for function, arity in functions:
                assert function in self.function_interpretations and self.function_arities[function] == arity
            
            for relation, arity in relations:
                assert relation in self.relation_interpretations and self.relation_arities[relation] in {-1, arity}
            
            # Check della verità della formula nel modello
            if isinstance(formula, Term):
                result = self.evaluate_term(formula, assignment)
            else:
                result = self.evaluate_predicate(formula)
            if not result:
                return False
        
        return True

#Ex[(x = b & Ey[(y = plus(x, 1) & times(x, y) = a)])]
#Ex[((x=y & x=z) -> ~P(x,z))]
#(Ex[(G(y) & H(z))] -> I(w))
#Ex[(F(x) -> G(y, 4))]
#(Ex[(Q(x) & R(x))] -> (S(y) | T(z)))
