# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=line-too-long
# pylint: disable=trailing-whitespace

from typing import NewType, Optional, Tuple, Set, AbstractSet, Sequence, Iterable
from dataclasses import dataclass
from itertools import product
from tabulate import tabulate

@dataclass
class Proposition:
    root: str
    left: Optional['Proposition']
    right: Optional['Proposition']

    Model = NewType('Model', dict[str, bool])

    def __init__(self, root: str, left: Optional['Proposition'] = None, right: Optional['Proposition'] = None):
        self.root = root
        self.left = left
        self.right = right

    @staticmethod
    def __is_variable__(string: str) -> bool:
        #le variabili sono lettere minuscole che vanno dalla 'a' alla 'z', sono le foglie dell'albero
        return len(string) == 1 and 'a' <= string <= 'z'

    @staticmethod
    def __is_unary__(string: str) -> bool:
        #operatori che hanno un solo ramo che sarà il sinistro, cioè il NOT
        return string == '~'

    @staticmethod
    def __is_binary__(string: str) -> bool:
        #operatori come AND, OR e IMPLICA
        return string == '&' or string == '|' or string == '->'

    def __get_variables__(self) -> Set[str]:
        if self.__is_variable__(self.root):
            #il return è avvolto da graffe perché la funzione deve ritornare un set
            return {self.root} 
        elif self.__is_unary__(self.root):
            return self.left.__get_variables__()
        elif self.__is_binary__(self.root):
            #unisco le variabili di entrambi i rami, '|' è l'operatore UNIONE
            return self.left.__get_variables__() | self.right.__get_variables__()
        else:
            return set()

    def __get_operators__(self) -> Set[str]:
        #stessa cosa del metodo getVariables, ma con gli operatori
        operators = set()

        if self.__is_unary__(self.root) or self.__is_binary__(self.root):
            operators.add(self.root)

        if self.left:
            operators |= self.left.__get_operators__()
        if self.right:
            operators |= self.right.__get_operators__()

        return operators

    #conversione dell'albero in stringa
    def string_repr(self) -> str:
        if self.__is_variable__(self.root):
            #ritorno il valore di root
            return self.root
        elif self.__is_unary__(self.root):
            #concateno il valore di root col ramo sinistro
            return f'{self.root}{self.left.string_repr()}'
        elif self.__is_binary__(self.root):
            #concateno il valore del nodo root e i due rami, con root al centro
            return f'({self.left.string_repr()} {self.root} {self.right.string_repr()})'

    @staticmethod
    def __tokenize__(expression: str) -> list[str]:
        '''
        Questa funzione prende l'espressione e la divide in "token".
        Un token può essere una variabile, una costante, un operatore o una parentesi.
        Questa lista di token servirà poi per costruire l'albero
        Ad es. (p & q) diventa ['(', 'p', '&', 'q', ')']
        '''
        tokens = []
        current_token = ''
        for char in expression:
            if char in ['(', ')']:
                if current_token:
                    tokens.append(current_token)
                    current_token = ''
                tokens.append(char)
            elif char.isspace():
                if current_token:
                    tokens.append(current_token)
                    current_token = ''
            else:
                current_token += char
        if current_token:
            tokens.append(current_token)
        return tokens

    @classmethod
    def __build_tree__(cls, tokens: list) -> 'Proposition':
        if not tokens:
            return Proposition('')

        #prendo il primo token della lista
        token = tokens.pop(0)

        if token == '(': #se l'espressione iniza con '(', sicuramente c'è un operatore binario
            #quindi prendo le sotto-espressioni da destra e sinistra in maniera ricorsiva
            left_sub_formula = cls.__build_tree__(tokens)
            operator = tokens.pop(0)
            right_sub_formula = cls.__build_tree__(tokens)
            assert tokens.pop(0) == ')' #Verifico che le parentesi sono chiuse correttamente
            return cls(root=operator, left=left_sub_formula, right=right_sub_formula)
        #se non ci sono parentesi, allora ci può essere una variabile o un NOT
        elif cls.__is_variable__(token):
            return cls(root=token)
        elif cls.__is_unary__(token):
            sub_formula = cls.__build_tree__(tokens)
            return cls(root=token, left=sub_formula)
        else:
            raise ValueError(f"Invalid token: {token}")

    @staticmethod
    def parse_proposition(expression: str) -> 'Proposition':
        tokens = Proposition.__tokenize__(expression)
        return Proposition.__build_tree__(tokens)

    def __is_model__(self, model: Model) -> bool:
        #Controlla se il dictionary passato è un modello logico su un set di variabili
        for key in model:
            if not self.__is_variable__(key):
                return False
        return True

    def __get_model_variables__(self, model: Model) -> AbstractSet[str]:
        #Ritorna tutte le variabili del modello
        assert self.__is_model__(model) #se is_model ritorna false, l'assert emette un errore
        return model.keys()

    def __get_all_models__(self, variables: Sequence[str]) -> Iterable[Model]:
        #ritorna tutte le combinazioni booleane delle variabili
        for v in variables: #check delle variabili
            assert self.__is_variable__(v)

        values = [False, True]
        #product genera il prodotto cartesiano, aka tutte le possibili combinazioni
        combinations = product(values, repeat=len(variables))

        for combination in combinations:
            #con zip associo le variabili alle combinazioni booleane
            #con dict converto le coppie variabile-booleano in dictionary
            #yield restituisce il dizionario appena creato
            yield dict(zip(variables, combination))

    def __evaluate_model__(self, model: Model) -> bool:
        if self.__is_variable__(self.root):
            #Se il nodo attuale è una variabile, ritorna il suo valore dal modello
            return model[self.root]
        elif self.__is_unary__(self.root):
            #In caso di operatore unario, cioè NOT, ritorna il not del suo nodo sottostante
            return not self.left.__evaluate_model__(model)
        elif self.__is_binary__(self.root):
            left_value = self.left.__evaluate_model__(model)
            right_value = self.right.__evaluate_model__(model)

        if self.root == '&':
            return left_value and right_value
        elif self.root == '|':
            return left_value or right_value
        elif self.root == '->':
            return (not left_value) or right_value
        else:
            raise ValueError(f"Invalid binary operator: {self.root}")
        
    def __get_truth_values__(self) -> Iterable[bool]:
        #ritorna i valori di verità della formula
        #ad es. se l'espressione è (p & q), ritorna [False, False, False, True]
        variables = list(self.__get_variables__()) 
        models = self.__get_all_models__(variables)

        for model in models:
            yield self.__evaluate_model__(model)

    def is_tautology_contradiction_statisfiable(self) -> Tuple[bool, bool, bool]:
        truth_values = self.__get_truth_values__()
        truth_values = list(truth_values)
        counter = truth_values.count(True)
        if counter == 0:
            result = (False, True, False)
        elif counter == len(truth_values):
            result = (True, False, False)
        else:
            result = (False, False, True)
        return result

    def print_truth_table(self):
        #mostra a video la tabella di verità
        table = []
        variables = self.__get_variables__()
        headers = list(variables)
        headers.append(self.string_repr())
        colalign = ('center',) * len(headers)
        models = self.__get_all_models__(variables)
        truth_values = self.__get_truth_values__()
        for (model, truth_value) in zip(models, truth_values):
            row = list(model.values())
            row.append(truth_value)
            table.append(row)
        print()
        print(tabulate(table, headers, tablefmt='heavy_outline', colalign=colalign))
        print()

    def check_equivalence(self, string: str) -> bool:
        #controlla se la formula è equivalente alla formula passata come parametro
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
    
    def print_tree(self, node: Optional['Proposition'] = None, indent: int = 0):
        c = 3
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
