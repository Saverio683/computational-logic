# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=line-too-long
# pylint: disable=trailing-whitespace

from typing import Optional, Tuple
from dataclasses import dataclass

def is_constant(string: str) -> bool:
    #le costanti sono lettere minuscole che vanno dalla 'a' alla 'e'
    return string.isalnum() or string >= 'a' and string <= 'e'
    
def is_variable(string: str) -> bool:
    #le variabili sono rappresentate da lettere minuscole dalla 'u' alla 'z' 
    return string >= 'u' and string <= 'z'
def is_function(string: str) -> bool:
    print(string[0])
    #una funzione ha come prima lettera una minuscola che va dalla 'f' alla 't', è alfanumerica
    return string[0] >= 'f' and string[0] <= 't' 

@dataclass
class Term:
    #Un albero i cui nodi possono essere variabili, costanti e funzioni
    root: str
    #Gli argomenti sono presenti se "root" è una funzione
    arguments: Optional[Tuple['Term', ...]]

    def __init__(self, root: str, arguments: Optional[Tuple['Term']] = None):
        if is_constant(root) or is_variable(root):
            #se il nodo NON è una funzione, non gli si devono passare argomenti, se sì si ritorna un errore
            assert arguments is None
            self.root = root
        else:
            #viceversa, se il nodo è una funzione, DEVONO esserci gli argomenti (almeno 1)
            assert is_function(root)
            assert arguments is not None and len(arguments) > 0
            self.root = root
            self.arguments = arguments

#f(g(x),3)
my_term = Term('f', (Term('x')))
