

class Diccionario(object):
    def __init__(self):
        self.data = dict()
    def add_pair(self,key,value):
        assert isinstance(key,str), "Diccionario solo admite claves de tipo string"
        self.data[key] = value
    def search_key(self,key):
        return self.data[key]
    def cantidad_elementos(self):
        return len(self.data)
    def get_claves_ordenadas(self):
        return sorted(self.data)
    def keys(self):
        return self.data.keys()
    def pop(self,key):
        self.data.pop(key)