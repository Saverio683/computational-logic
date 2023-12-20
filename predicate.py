# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=line-too-long
# pylint: disable=trailing-whitespace

from typing import Optional, Sequence, Tuple, Union
from dataclasses import dataclass

def is_constant(string: str) -> bool:
    #le costanti sono lettere minuscole che vanno dalla 'a' alla 'e' o numeri
    return  string >= 'a' and string <= 'e' or string.isalnum()

def is_variable(string: str) -> bool:
    #le variabili sono rappresentate da lettere minuscole dalla 'u' alla 'z' 
    return string >= 'u' and string <= 'z'

def is_function(string: str) -> bool:
    #una funzione ha come prima lettera una minuscola che va dalla 'f' alla 't', è alfanumerica
    return string[0] >= 'f' and string[0] <= 't' 

def is_equality(string: str) -> bool:
    #verifica se la stringa è l'operatore di uguaglianza (=)
    return string == '='

def is_relation(string: str) -> bool:
    #verifica se la stringa è il nome di una relazione
    #la stringa inizia con una maiuscola compresa tra 'F' e 'T' ed è alfanumerica
    return string[0] >= 'F' and string[0] <= 'T' and string.isalnum()

def is_unary(string: str) -> bool:
    #verifico se è un operatore unario (NOT)
    return string == '~'

def is_binary(string: str) -> bool:
    #in caso di AND e OR
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
        if is_function(root):
            assert arguments is not None and len(arguments) > 0, 'Function requires arguments.'
            self.root = root
            self.arguments = tuple(arguments) if arguments else None
        elif is_constant(root) or is_variable(root):
            assert arguments is None, 'Constants and variables cannot have arguments.'
            self.root = root
            self.arguments = None   
        else:
            assert False, f'Invalid root string: {root}'

    def string_repr(self, argument: Optional['Term'] = None) -> str:
        '''
            converto l'albero in stringa.
            la funzione parse_term è quella che costruisce la stringa lavorando nodo per nodo.
            se il nodo è una funzione (quindi possiege gli argomenti), si concatena il suo valore 
            con quello degli argomentim, iterandoli in maniera ricorsiva.
            dopo che viene 'convertita' una funzione, si aggiunge una ',' che serve a dividere gli argomenti
            se il nodo è una variabile/costante, si ritorna semplicemente il valore del nodo.
        '''
        def parse_term(node: Term) -> str:
            if is_function(node.root):
                result = ''
                for arg in node.arguments:
                    result += node.string_repr(arg)
                return f'{node.root}({result})'+','
            return node.root
        if argument:
            return parse_term(argument)
        #rimuovo l'ultimo carattere, che è una virgola in più
        return parse_term(self)[:-1]
@dataclass
class Predicate:
    '''
        'root' è il nodo dell'albero può essere una relazione/uguaglianza/operatore/quantificatore
        'arguments' indica gli argomenti di root se è una relazione o un'uguaglianza
        'left' e 'right' sono i sottonodi, nel caso se root è un operatore unario/binario
        'variable' e 'statement' vengono assegnati se root è un quantificatore
    '''
    root: str
    arguments: Optional[Tuple[Term, ...]]
    left: Optional['Predicate']
    right: Optional['Predicate']
    variable: Optional[str]
    statement: Optional['Predicate']

    def __init__(self, root: str, arguments_or_left_or_variable: Union[Sequence[Term], 'Predicate', str], 
                right_or_statement: Optional['Predicate'] = None):
        if is_equality(root) or is_relation(root):
            #verifico se arguments_or_left_or_variable È una sequenze e NON una stringa
            assert isinstance(arguments_or_left_or_variable, Sequence) and \
            not isinstance(arguments_or_left_or_variable, str)
            if is_equality(root):
                #Verifico se gli argomenti per l'uguale sono 2
                assert len(arguments_or_left_or_variable) == 2
            assert right_or_statement is None
            self.root, self.arguments = root, tuple(arguments_or_left_or_variable)
        elif is_unary(root):
            # Verifico che solo il figlio sinistro esiste
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

    @staticmethod
    def __parse_predicate__(node: 'Predicate') -> str:
        if is_binary(node.root):
            second = ''
            if isinstance(node.second, Term):
                second = Term.string_repr(node.second)
            elif isinstance(node.second, Predicate):
                second = node.__parse_predicate__(node.second)
            elif isinstance(node.second, str):
                second = node.second
            return f'{node.__parse_predicate__(node.first)}{node.root}{second}'
        elif is_unary(node.root):
            return f'{node.root}{node.__parse_predicate__(node.first)}'
        elif is_quantifier(node.root):
            statement = ''
            if isinstance(node.statement, Term):
                statement = Term.string_repr(node.statement)
            elif isinstance(node.statement, Predicate):
                statement = node.__parse_predicate__(node.statement)
            elif isinstance(node.statement, str):
                statement = node.statement
            return f'{node.root}{node.variable}[{statement}]'
        elif is_relation(node.root):
            args = ''
            for arg in node.arguments:
                args += Term.string_repr(arg, argument=arg)+','
            return f'{node.root}({args[:-1]})'
        else:
            return f'{Term.string_repr(node.arguments[0])}{node.root}{Term.string_repr(node.arguments[1], argument=node.arguments[1])}'

    def string_repr(self) -> str:
        return '('+Predicate.__parse_predicate__(self)+')'


#f(g(x),3)
my_term = Term('f', [Term('g', [Term('x')]), Term('3')])
print(my_term.string_repr())
#‘(Ex[f(g(x),3)=y]->GT(y,4))’
my_predicate = Predicate('->',
    Predicate('E', 'x',
        Predicate('=', [my_term, Term('y')])
    ),
    Predicate('GT', [Term('y'), Term('4')])
)

print(my_predicate.string_repr())
