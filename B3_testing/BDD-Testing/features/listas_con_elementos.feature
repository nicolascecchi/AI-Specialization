# language: es

Característica: Gestion de diccionario con elementos

Antecedentes: Pruebas sobre diccionarios con elementos
  Dado un diccionario que contiene solo una clave "{clave}" con el valor "{valor}" 

Escenario: Actualizar el valor de una clave
  Cuando reasigno un nuevo valor "{nuevo_valor}" a una clave "{clave}" existente
  Entonces al buscar la clave "{clave}" obtengo el valor "{nuevo_valor}"

Escenario: Eliminar una pareja clave-valor
  Cuando elimino la clave "{clave}"
  Entonces al buscar la clave "{clave}" no obtengo ningun valor

Esquema del escenario: Contar elementos de un diccionario no vacio
    Cuando agrego los pares clave-valor <nuevos_pares> 
    Entonces el diccionario tiene <N> elementos
    Ejemplos: 
    |nuevos_pares             |N|
    |{'a':'1'}                |2|
    |{'a':'1','b':'2'}        |3|
    |{'a':'1','b':'2','c':'3'}|4|

