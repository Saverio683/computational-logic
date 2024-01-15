# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=line-too-long
# pylint: disable=trailing-whitespace

from typing import List, NewType, Optional, Tuple, Set, AbstractSet, Sequence, Iterable
from dataclasses import dataclass
from itertools import product
from tabulate import tabulate

@dataclass
class Proposition:
    '''
        Classe della fbf.
        Parametri:
            -root: il valore del nodo, può essere una variabile o un connettivo.
            -left, right: altri nodi della classe Proposition, se root è un operatore unario, sarà presente
            solo il parametro left, in caso di operatore binario sarà presente anche right.
            Se root è una variabile, left e right saranno None.
    '''
    root: str #valore del nodo
    left: Optional['Proposition']
    right: Optional['Proposition']

    #tipo che definisce un'interpretazione
    Model = NewType('Model', dict[str, bool]) 

    def __init__(self, root: str, left: Optional['Proposition'] = None, right: Optional['Proposition'] = None):
        self.root = root
        self.left = left
        self.right = right

    @staticmethod
    def __is_variable__(string: str) -> bool:
        '''
            Check se il valore del nodo è una variabile.
            Le variabili sono lettere minuscole che vanno dalla 'a' alla 'z', sono le foglie dell'albero.
        '''        
        return len(string) == 1 and 'a' <= string <= 'z'

    @staticmethod
    def __is_unary__(string: str) -> bool:
        '''
            Check se il valore del nodo è un operatore unario (NOT), indicato con il simbolo '~'.
        '''  
        return string == '~'

    @staticmethod
    def __is_binary__(string: str) -> bool:
        '''
            Check se il valore del nodo è un operatore binario (AND, OR, IMPLICA), indicati
            con i rispettivi simboli '&', '|' e '->'.
        ''' 
        return string in ('&', '|', '->')

    def __get_variables__(self) -> Set[str]:
        '''
            Metodo che ritorna tutte le variabili della fbf.
        '''
        if self.__is_variable__(self.root):
            #il return è avvolto da graffe perché la funzione deve ritornare un set
            return {self.root} 
        if self.__is_unary__(self.root):
            return self.left.__get_variables__()
        if self.__is_binary__(self.root):
            #unisco le variabili di entrambi i rami, '|' è l'operatore UNIONE
            return self.left.__get_variables__() | self.right.__get_variables__()
        return set() #nessuna variabile, ritorno un set vuoto

    def __get_operators__(self) -> Set[str]:
        '''
            Metodo che ritorna tutti gli operatori della fbf.
        '''
        operators = set()
        if self.__is_unary__(self.root) or self.__is_binary__(self.root):
            operators.add(self.root)
        if self.left:
            operators |= self.left.__get_operators__()
        if self.right:
            operators |= self.right.__get_operators__()
        return operators

    def string_repr(self) -> str: #inorder traversal
        '''
            Metodo che si occupa della conversione da albero in stringa (inorder traversal).
        '''
        if self.__is_variable__(self.root):
            #ritorno il valore di root
            return self.root
        if self.__is_unary__(self.root):
            #concateno il valore di root col ramo sinistro
            return f'{self.root}{self.left.string_repr()}'
        if self.__is_binary__(self.root):
            #concateno il valore del nodo root e i due rami, con root al centro
            return f'({self.left.string_repr()} {self.root} {self.right.string_repr()})'
        raise ValueError(f'invalid node: {self.root}')

    @staticmethod
    def __tokenize__(expression: str) -> list[str]:
        '''
            Questa funzione prende l'espressione e la divide in 'token'.
            Un token è ogni elemento della stringa della fbf. Può essere una variabile, 
            una costante, un operatore o una parentesi.
            Questa lista di token servirà poi per costruire l'albero.
            Ad es. passando in input '(p & q)', la funzione ritorna ['(', 'p', '&', 'q', ')'].
        '''
        tokens = []
        current_token = ''
        for char in expression:
            if char in ['(', ')']:
                if current_token:
                    tokens.append(current_token)
                    current_token = ''
                tokens.append(char)
            #NB: considero anche il carattere precedente, poichè l'operatore IMPLICA è composto da due caratteri
            elif char.isspace() or (current_token and current_token[-1] in ['~', '&', '|', '->']):
                if current_token:
                    tokens.append(current_token)
                    current_token = ''
                if not char.isspace():
                    current_token += char
            else:
                current_token += char
        if current_token:
            tokens.append(current_token)
        return tokens

    @classmethod
    def __build_tree__(cls, tokens: list) -> 'Proposition':
        '''
            Questo è il metodo che, preso l'output della funzione tokenize, interpreta i token passati per
            derivarne l'albero, ritornando l'istanza della classe Proposition.
        '''
        if not tokens:
            return Proposition('')

        #prendo il primo token della lista
        token = tokens.pop(0)

        if token == '(': #se l'espressione iniza con '(', sicuramente ci sarà un operatore binario, considerando che 
            #si sta lavorando con le fbf, quindi va rispettato la corretta chiusura delle parentesi
            #quindi prendo le sotto-espressioni da destra e sinistra in maniera ricorsiva
            left_sub_formula = cls.__build_tree__(tokens)
            operator = tokens.pop(0)
            right_sub_formula = cls.__build_tree__(tokens)
            #Verifico che le parentesi sono chiuse correttamente
            assert tokens.pop(0) == ')', 'Missing closing parenthesis'
            return cls(root=operator, left=left_sub_formula, right=right_sub_formula)
        #se non ci sono parentesi, allora ci può essere una variabile o un NOT
        if cls.__is_variable__(token):
            return cls(root=token)
        if cls.__is_unary__(token):
            sub_formula = cls.__build_tree__(tokens)
            return cls(root=token, left=sub_formula)
        raise ValueError(f'Invalid token: {token}')

    @staticmethod
    def parse_proposition(expression: str) -> 'Proposition':
        '''
            Metodo che converte la stringa di una fbf nel suo corrispettivo albero.
            Così facendo, si può istanziare la classe Proposition senza invocare esplicitamente il costruttore.
        '''
        tokens = Proposition.__tokenize__(expression)
        return Proposition.__build_tree__(tokens)

    def __is_model__(self, model: Model) -> bool:
        '''
            Metodo che controlla se il dictionary passato è un modello valido.
            Il controllo avviene verificando la sintasi delle sue variabili.
        '''
        for key in model:
            if not self.__is_variable__(key):
                return False
        return True

    def __get_model_variables__(self, model: Model) -> AbstractSet[str]:
        '''
            Metodo che ritorna le variabili di un modello.
        '''
        assert self.__is_model__(model), 'Invalid model'
        return model.keys()

    def __get_all_models__(self, variables: Sequence[str]) -> Iterable[Model]:
        '''
            Metodo che ritorna tutte le combinazioni di verità.
        '''        
        for v in variables: #check delle variabili
            assert self.__is_variable__(v), f'Invalid variable: {v}'

        values = [False, True]
        #product genera il prodotto cartesiano, ossia tutte le possibili combinazioni
        combinations = product(values, repeat=len(variables))

        for combination in combinations:
            #con zip 'associo' le variabili alle combinazioni booleane
            #con dict converto le coppie variabile-booleano in dictionary
            #yield restituisce il dizionario appena creato, pronto per essere iterato
            yield dict(zip(variables, combination))

    def __evaluate_model__(self, model: Model) -> bool:
        '''
            Associo i valori di verità alle variabili della fbf, iterando ogni nodo dell'albero.
        '''
        if self.__is_variable__(self.root):
            #Se il nodo attuale è una variabile, ritorna il suo valore dal modello
            return model[self.root]
        if self.__is_unary__(self.root):
            #In caso di operatore unario, cioè NOT, ritorna il not del suo nodo sottostante
            return not self.left.__evaluate_model__(model)
        if self.__is_binary__(self.root):
            left_value = self.left.__evaluate_model__(model)
            right_value = self.right.__evaluate_model__(model)
        if self.root == '&':
            return left_value and right_value
        if self.root == '|':
            return left_value or right_value
        if self.root == '->':
            return (not left_value) or right_value

        raise ValueError(f"Invalid binary operator: {self.root}")
        
    def __get_truth_values__(self) -> Iterable[bool]:
        '''
            Metodo che ritorna i valori di verità della fbf.
            Ad es. se l'espressione è (p & q), ritorna [False, False, False, True].
        '''
        variables = list(self.__get_variables__()) 
        models = self.__get_all_models__(variables)

        for model in models:
            yield self.__evaluate_model__(model)

    def is_tautology_contradiction_statisfiable(self) -> Tuple[bool, bool, bool]:
        '''
            Metodo che verifica le proprietà della fbf, ritorna 3 booleani che indicano rispettivamente
            se la fbf è una tautologia, una contraddizione o se è soddisfacibile.
        '''
        truth_values = self.__get_truth_values__()
        truth_values = list(truth_values)
        counter = truth_values.count(True)
        if counter == 0: #contraddizione, nessun modello
            result = (False, True, False)
        elif counter == len(truth_values): #la fbf è valida, quindi una tautologia
            result = (True, False, False)
        else:
            result = (False, False, True) #esiste almeno un modello, la fbf è soddisfacibile
        return result
    
    def get_subexpressions(self) -> List[str]:
        '''
            Metodo che ritorna le sottoformule della fbf.
        '''
        subexprs = []

        if self.__is_variable__(self.root):
            return [self.root]
        if self.__is_unary__(self.root):
            subexprs.extend(self.left.get_subexpressions())
        elif self.__is_binary__(self.root):
            subexprs.extend(self.left.get_subexpressions())
            subexprs.extend(self.right.get_subexpressions())
        
        subexprs.append(self.string_repr())
        return subexprs

    def print_truth_table(self) -> None:
        '''
            Metodo che mostra a video la tabella di verità, valutando la fbf e le sue sottoformule.
        '''
        table = []
        variables = list(self.__get_variables__())
        subexpressions = self.get_subexpressions()
        
        # Rimuovo le variabili dalla lista delle sottoespressioni
        subexpressions = [expr for expr in subexpressions if expr not in variables]
        
        headers = variables + subexpressions
        colalign = ('center',) * len(headers)
        
        models = self.__get_all_models__(variables)
        
        for model in models:
            row = [model[var] for var in variables]
            subexpr_values = {}
            for subexpr in subexpressions:
                subexpr_formula = Proposition.parse_proposition(subexpr)
                subexpr_values[subexpr] = subexpr_formula.__evaluate_model__({var: model[var] for var in variables})
            row.extend([subexpr_values[subexpr] for subexpr in subexpressions])
            table.append(row)
        
        print()
        print(tabulate(table, headers, tablefmt='heavy_outline', colalign=colalign))
        print()

    def check_equivalence(self, string: str) -> bool:
        '''
            Metodo che verifica se la fbf è equivalente all'espressione passata in input.
        '''
        other_proposition = Proposition.parse_proposition(string)

        #ricavo le variabili comuni tra le 2 formule, se hanno variabili diverse, non possono essere equivalenti
        common_variables = self.__get_variables__() & other_proposition.__get_variables__()
        if common_variables != self.__get_variables__() or common_variables != other_proposition.__get_variables__():
            return False

        #ricavo le possibili combinazioni e ne confronto le tabelle di verità
        models = self.__get_all_models__(list(common_variables))
        for model in models:
            if self.__evaluate_model__(model) != other_proposition.__evaluate_model__(model):
                return False
        return True

    def print_tree(self, node: Optional['Proposition'] = None, indent: int = 0) -> None:
        '''
            Metodo che mostra a video l'albero della fbf.
        '''
        c = 3 #valore iniziale dell'indentazione, ad ogni nodo figlio questo valore aumenta
        if not node:
            print()
            print(self.root)
            if self.left:
                self.print_tree(self.left)
            if self.right:
                self.print_tree(self.right)
            print()
        else:
            indent += c
            print(f"{' '*(indent-c)}╚{'═'*(c-1)}{node.root}")
            if node.right and node.left:
                self.print_tree(node.right, indent)
                self.print_tree(node.left, indent)
            elif node.left:
                self.print_tree(node.left, indent)
            else:
                return
            
    def check_dnf_cnf(self) -> Tuple[bool, bool]:
        '''
            Metodo che verifica se la fbf è un CNF o DNF.
            Ritorna i valori booleani rispettivamente di: CNF, DNF.
        '''

        def check_clauses(node: 'Proposition', is_cnf: bool) -> bool:
            '''
                Metodo dove avviene effettivamente la verifica.
            '''
            if self.__is_binary__(node.root):
                invalid_cnf_ops = ('&', '->')
                invalid_dnf_ops = ('|', '->')
                
                if (is_cnf and node.root in invalid_cnf_ops) or (not is_cnf and node.root in invalid_dnf_ops):
                    return False
                
                return check_clauses(node.left, is_cnf) and check_clauses(node.right, is_cnf)
            
            if self.__is_unary__(node.root):
                if (is_cnf and node.root in ('&', '->')) or (not is_cnf and node.root in ('|', '->')):
                    return False
                return check_clauses(node.left, is_cnf)
            
            return True

        is_cnf, is_dnf = False, False

        if self.__is_binary__(self.root):
            if self.root == '&':
                is_cnf = check_clauses(self.left, True) and check_clauses(self.right, True)
            elif self.root == '|':
                is_dnf = check_clauses(self.left, False) and check_clauses(self.right, False)

        return is_cnf, is_dnf

    def get_equivalent_expression(self) -> str: 
        '''
            Questo metodo ritorna una fbf ridotta in forma normale, equivalente alla fbf 
            di partenza.
        '''
        #duplico il self, così non modifico perennemente la proposizione di partenza ma ne genero una nuova
        equivalent_expression = Proposition.parse_proposition(self.string_repr())
        def replace_derivable_operators(node: Proposition) -> None:
            '''
                Questa funzione si occupa del primo step per ricavare una fbf equivalente in forma normale,
                cioè rimuovere tutti gli operatori 'non-principali'.
                \nIn questo caso l'unico operatore derivabile presente è l'operatore IMPLICA.
                \nSe la fbf contiene questo operatore, viene modificata nella sua fbf equivalente.
                Ad es. (q -> b) diventa (~q | b).
            '''
            if self.__is_binary__(node.root):
                replace_derivable_operators(node.left)
                replace_derivable_operators(node.right)
                if node.root in ('->'):
                    node.root = '|'
                    node.left = Proposition.parse_proposition(f'~{node.left.string_repr()}')
            elif self.__is_unary__(node.root):
                replace_derivable_operators(node.left)
        replace_derivable_operators(equivalent_expression)

        def check_for_de_morgan_laws(node: Proposition) -> None:
            '''
                Questa funzione applica le leggi di de Morgan alla fbf.
                \nOssia: 
                    \n'~(P | Q)' è equivalente a '~P & ~Q'
                    \n'~(P & Q)' è equivalente a '~P | ~Q'
            '''
            if self.__is_binary__(node.root):
                check_for_de_morgan_laws(node.left)
                check_for_de_morgan_laws(node.right)
            elif self.__is_unary__(node.root):
                if self.__is_binary__(node.left.root):
                    #non si tiene conto dell'IMPLICA, poiché questa funzione viene eseguita dopo la precedente
                    #che avrà già rimosso quel tipo di operatore
                    new_root = '|' if node.left.root == '&' else '&'
                    node.root = new_root
                    node.right = Proposition.parse_proposition(f'~{node.left.right.string_repr()}')
                    node.left = Proposition.parse_proposition(f'~{node.left.left.string_repr()}')

        def check_for_double_negation(node: Proposition) -> None:
            '''
                Verifica se ci sono due NOT in successione nell'albero, annullandoli.
            '''
            if self.__is_binary__(node.root):
                check_for_double_negation(node.left)
                check_for_double_negation(node.right)
            elif self.__is_unary__(node.root):
                if self.__is_unary__(node.left.root):
                    copy = node.left.left
                    node.root = copy.root
                    node.left = copy.left
                    node.right = copy.right
                    check_for_double_negation(node)

        def check_for_distributivity(node: Proposition) -> None:
            '''
                Questo metodo applica la proprietà distributiva alla fbf per convertirla
                in CNF o DNF.
                Ad es. 'P & (Q | R)' diventa '(P & Q) | (Q & R)'.
                Le variabili p rappresenterà
            '''
            if not self.__is_binary__(node.root):
                return
            
            if self.__is_binary__(node.left.root):
                p, q_and_r = node.left, node.right
            elif self.__is_binary__(node.right.root):
                p, q_and_r = node.right, node.left
            else:
                return
            
            if node.root == '&':
                if q_and_r.root == '|':
                    node.root = '|'
                    node.left = Proposition.parse_proposition(f'({p.string_repr()} & {q_and_r.left.string_repr()})')
                    node.right = Proposition.parse_proposition(f'({p.string_repr()} & {q_and_r.right.string_repr()})')
            else:
                if q_and_r.root == '&':
                    node.root = '&'
                    node.left = Proposition.parse_proposition(f'({p.string_repr()} | {q_and_r.left.string_repr()})')
                    node.right = Proposition.parse_proposition(f'({p.string_repr()} | {q_and_r.right.string_repr()})')

        check_for_de_morgan_laws(equivalent_expression)
        check_for_double_negation(equivalent_expression)
        check_for_distributivity(equivalent_expression)

        return equivalent_expression.string_repr()
