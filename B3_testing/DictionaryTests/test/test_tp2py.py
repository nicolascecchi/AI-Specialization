import pytest
from utils.clase_diccionario import Diccionario


# Un Diccionario vacío tiene 0 pares clave-valor
# Un Diccionario vacío tiene 0 claves
def test_empty_dict():
    dic = Diccionario()
    assert len(dic.data) == 0,"En una dict vacío hay 0 elementos"
    assert len(dic.data.keys()) == 0, "Un dict vacio no tiene claves"


# Requisitos asociados: 1,3,4,5
# Al agregar un par clave:valor a un dic vacío, vacía hay un elemento
# Al agregar un par clave:valor a un dic vacío, se puede recuperar el valor con la clave 
# Se puede actualizar el valor de una clave
def test_dic_agregar_elemento():
    dic = Diccionario()
    dic.add_pair("clave","valor")
    assert len(dic.data) == 1, "Al agregar un elemento, hay un elemento"
    assert dic.search_key("clave") == "valor", "Al agregar un elemento, se puede recuperar por su clave"
    dic.add_pair("clave","nuevo_valor")
    assert dic.search_key("clave") == "nuevo_valor","Cuando se agrega una clave que ya existe, se sobreescribe el valor"

def test_recuperar_cantidad_elementos():
    dic = Diccionario()
    assert dic.cantidad_elementos() == 0, "No se recupera la cantidad de elementos"
    dic.add_pair("clave","valor")
    assert dic.cantidad_elementos() == 1, "No se recupera la cantidad de elementos"

#Se debe poder eliminar un elemento por la clave
def test_eliminar_elemento():
    dic = Diccionario()
    dic.add_pair("clave","valor")
    dic.pop('clave')
    assert dic.cantidad_elementos() == 0, "No se eliminó el elemento por su clave"

def test_orden():
    dic = Diccionario()
    dic.add_pair("A",1)
    dic.add_pair("C",3)
    dic.add_pair("B",2)
    assert dic.get_claves_ordenadas() == ["A","B","C"], "El Diccionario no se muestra ordenado"

# Requisitos asociados: 2,3 claves string y únicas
def test_requisitos_claves():
    dic = Diccionario()
    try:
        dic.add_pair(5,'value')
        assert dic.cantidad_elementos() == 0, "El diccionario admitió una clave string"
    except: 
        assert True