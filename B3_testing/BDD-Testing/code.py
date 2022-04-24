class Diccionario():
    def __init__(self):
        self.valores = {}

    def count(self):
        return len(self.valores)
        
    def buscarClave(self,clave):
        if clave in self.valores.keys():
            return self.valores[clave]
        else:
            return None 
    def agregarPar(self,clave,valor):
        self.valores.update({clave:valor})
    
    def actualizarClave(self,clave,valor):
        self.valores.update({clave:valor})
    
    def eliminarClave(self,clave):
        self.valores.pop(clave)
    
    def verClaves(self):
        return [k for k in sorted(self.valores)]