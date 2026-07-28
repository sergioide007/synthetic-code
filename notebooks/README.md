# Notebooks

Ambos notebooks están regenerados para instalar el paquete real del
repositorio (`pip install -e ".[all]"` tras `git clone`) e importar desde
`sintetico`/`trazabilidad`, en vez de pegar código o rutas de ficheros
sueltas como en versiones anteriores.

- `codigo_sintetico_colab.ipynb`: instalación, suite de tests, los 4
  pilares, demo de ahorro con API real (opcional), y cómo arrancar el
  dashboard de observabilidad desde Colab.
- `codigo_sintetico_trazabilidad.ipynb`: logger estructurado + agente
  ReAct, con proveedor real si configuras una API key o un simulador
  determinista si no.

**Antes de usarlos**, edita la celda con `REPO_URL = "https://github.com/<tu-usuario>/<tu-repo>.git"`
al inicio de cada uno y pon la URL real de tu fork/repo.
