# Proyecto-Mape

Instalar pipenv
	* pip install pipenv

### Preparación del repositorio

1. Clonar el repositorio
	* git clone -b **branch-name** https://github.com/agui197/Proyecto-Mape.git
2. Entrar al directorio del repositorio
	* cd Proyecto-Mape
3. Ejecutar una shell de pipenv
	* pipenv shell
4. Instalar las dependencias (la bandera -d instala las dependencias de desarrollo en un entorno productivo no hace falta)
	* pipenv sync -d
5. Iniciar el editor de texto desde el shell de pipenv (para Visual Studio Code)
	* code .


### Instalación de paquetes adicionales

Para instalar paquetes debes asegurarte que estás dentro de la carpeta del 
proyecto y de la shell de tu entorno virtual

* pipenv shell

Una vez seguro que estás dentro los paquetes se instalan con pipenv install

* pipenv install **paquete**

Despues de haber instalado el paquete debes generar el archivo lock para 
mantener las versiones de tus dependencias (puede hacer despues de instalar 
todos los paquetes necesarios)
	* pipenv lock

### Salir del entorno virtual

Solamente ejecuta el comando **exit** y estarás en tu shell global.
