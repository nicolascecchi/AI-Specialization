from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import os

def __init__():
    pass

def buscarGoogle(busqueda):
    """
    Inicia un driver headless con que consulta a Google el input de la función. 
    Obtiene la url del 3er resultado no patrocinado.  

    Input:
    busqueda <str> Palabra a buscar en google.
    Returns:
    url <str> Url del tercer resultado no patrocinado de Google al buscar la palabra.
    """
    #Set del webdriver
    options = webdriver.FirefoxOptions()
    options.headless = True
    driver = webdriver.Firefox(executable_path=os.getcwd()+'/geckodriver',
                            options=options)
    
    #Se consulta a Google directo desde el get
    driver.get("http://www.google.com/search?q="+ busqueda + "&start=" + '1')
    
    #Asegura que el Driver no busque los elementos antes de que se cargue bien la página
    #La clase yuRUbf de Google son resultados no patrocinados
    timeout = 1
    element_present = EC.presence_of_element_located((By.CLASS_NAME, 'yuRUbf'))
    WebDriverWait(driver, timeout).until(element_present)

    #Lista de resultados resultados no patrocinados
    res_no_patrocinados = driver.find_elements_by_class_name('yuRUbf')

    #toma la url del 3er resultados no patrocinado
    url = res_no_patrocinados[2].find_element(By.TAG_NAME,'a').get_attribute('href')

    #Cierra el driver
    driver.close()
    
    return url
