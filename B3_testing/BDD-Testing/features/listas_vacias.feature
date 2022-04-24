# language: es
# encoding: utf-8

Característica: Gestion de diccionario vacio

Antecedentes: Lista vacia
  Dado una lista vacia

Escenario: Verificar el estado de la lista
  Entonces la lista tiene 0 elementos
  Y si busco la clave "{clave}" no encuentro ningun valor

Escenario: Agregar un elemento a una lista vacía
    Cuando agrego la clave "{clave}" con el valor "{valor}"    
    Entonces la lista tiene 1 elemento almacenado
    Y si busco la clave "{clave}" obtengo el valor "{valor}"

Esquema del escenario: Recuperar lista ordenada de las claves almacenadas
    Cuando se le agregan los pares <pares> de claves y valores
    Entonces se puede recuperar una lista ordenada <lista_ordenada> de las claves
    Ejemplos: 
    |pares                                             |lista_ordenada|
    |{'Brasil':5,'Alemania':4,'Italia':4,'Argentina':2}|['Alemania','Argentina','Brasil','Italia']|
    |{'Barco':45,'Avion':952,'Auto':200}               |['Auto','Avion','Barco']|