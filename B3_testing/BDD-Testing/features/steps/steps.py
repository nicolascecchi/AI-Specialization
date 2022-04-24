
from behave import *
from code import *
from math import isnan

#Diccionarios sin elementos
@given(u'una lista vacia')
def step_impl(context):
    context.dic = diccionario()

@then(u'la lista tiene 0 elementos')
def step_impl(context):
    assert context.dic.count() == 0
@then(u'si busco la clave "{clave}" no encuentro ningun valor')
def step_impl(context,clave):
    assert context.dic.buscarClave(clave) == None

@when(u'agrego la clave "{clave}" con el valor "{valor}"')
def step_impl(context,clave,valor):
    context.dic.agregarPar(clave,valor)
@then(u'la lista tiene 1 elemento almacenado')
def step_impl(context):
    assert context.dic.count() == 1
@then(u'si busco la clave "{clave}" obtengo el valor "{valor}"')
def step_impl(context,clave,valor):
    assert context.dic.buscarClave(clave) == valor
@when(u'se le agregan los pares {pares} de claves y valores')
def step_impl(context,pares):
    pares = eval(pares)
    for k in pares:
        context.dic.agregarPar(k,pares[k])
@then(u'se puede recuperar una lista ordenada {lista_ordenada} de las claves')
def step_impl(context,lista_ordenada):
    context.dic.verClaves == eval(lista_ordenada)


# Diccionarios con elementos
@given(u'un diccionario que contiene solo una clave "{clave}" con el valor "{valor}"')
def step_impl(context,clave,valor):
    context.dic = diccionario()
    context.dic.agregarPar(clave,valor)

@when(u'reasigno un nuevo valor "{nuevo_valor}" a una clave "{clave}" existente')
def step_impl(context,nuevo_valor,clave):
    context.dic.actualizarClave(clave,nuevo_valor)
@then(u'al buscar la clave "{clave}" obtengo el valor "{nuevo_valor}"')
def step_impl(context,clave,nuevo_valor):
    assert context.dic.buscarClave(clave) == nuevo_valor

@when(u'elimino la clave "{clave}"')
def step_impl(context,clave):
    context.dic.eliminarClave(clave)
@then(u'al buscar la clave "{clave}" no obtengo ningun valor')
def step_impl(context,clave):
    assert context.dic.buscarClave(clave) == None

@when(u'agrego los pares clave-valor {nuevos_pares}')
def step_impl(context,nuevos_pares):
    nuevos_pares = eval(nuevos_pares)
    for k in nuevos_pares:
        context.dic.agregarPar(k,nuevos_pares[k])
@then(u'el diccionario tiene {N} elementos')
def step_impl(context,N):
    N = int(N)
    assert context.dic.count() == N
