from setuptools import setup  # Importamos la función setup

setup(name="DemoSeminario",   #Definimos el nombre del modulo
      packages=["orangedemo"],      #Ponemos el paquete donde se encuentran los widgets
      package_data={"orangedemo": ["icons/*.svg", "icons/*.png"]},      # Definimos donde se encuentran las imagenes y el tipo que son
      entry_points={"orange.widgets": "DemoSeminario = orangedemo"},    # Ponemos el paquete donde encuentran los widgets
      )