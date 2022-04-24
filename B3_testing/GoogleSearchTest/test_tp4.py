from buscarGoogle import buscarGoogle

def test_busqueda1(palabra='Computadora'):
    url = buscarGoogle(palabra)
    assert url == 'https://www.garbarino.com/q/computadora/srch'

def test_busqueda2(palabra='Telefonos'):
    url = buscarGoogle(palabra)
    assert url == 'https://tienda.personal.com.ar/samsung'

def test_busqueda3(palabra='Montañas'):
    url = buscarGoogle(palabra)
    assert url == 'https://www.geoenciclopedia.com/montanas/'
