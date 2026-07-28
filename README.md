# Código Sintético — código de referencia

Implementación de referencia para el libro **Código Sintético**: patrones de
producción para sistemas multiagente (orquestación eficiente, calidad
autónoma en CI/CD, código autolimpiable y arquitecturas resilientes),
más una **API de observabilidad con dashboard** al estilo Datadog/CloudWatch
para ver esos patrones funcionando en vivo.

## Estructura del repositorio

```
src/
  sintetico/       Librería de patrones (budget, orquestación, calidad,
                    resiliencia, gobernanza, seguridad, economía, ...)
  trazabilidad/     Logger estructurado + agente ReAct con trazabilidad
  sintetico_api/    API FastAPI + dashboard de observabilidad (Swagger incluido)
demos/              Scripts ejecutables por consola, uno por caso de uso
tests/              Suite de pytest (72+ tests; ver "Testing" abajo)
notebooks/          Notebooks de Colab
```

## Instalación

```bash
# Mínima: sólo los patrones, sin APIs reales ni dashboard
pip install -e .

# Con soporte para APIs reales (Anthropic/OpenAI/local) + Redis + MCP
pip install -e ".[all]"

# Sólo lo necesario para el dashboard/API
pip install -e ".[api]"

# Para desarrollo (tests)
pip install -e ".[dev]"
```

Copia `.env.example` a `.env` (o exporta las variables directamente) para
configurar tus API keys reales:

```bash
cp .env.example .env
# edita .env y añade tu ANTHROPIC_API_KEY / OPENAI_API_KEY
export $(cat .env | grep -v '^#' | xargs)
```

## Quick start

### 1. Demos por consola (sin necesidad de API key)

```bash
python demos/demo_pilares.py          # los 4 pilares, con costes simulados
python demos/demo_trazabilidad.py     # agente ReAct con logging estructurado
python demos/demo_redis_budget.py     # presupuesto distribuido (requiere Redis)
```

### 2. Demo de ahorro de costes con API real

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
python demos/demo_ahorro_tokens.py
```

Sin ninguna API key configurada, este mismo script cae automáticamente a
un proveedor simulado y lo indica explícitamente — la demo nunca falla
silenciosamente ni "finge" un ahorro que no es real.

### 3. Dashboard de observabilidad (API + Swagger + panel visual)

```bash
pip install -e ".[api]"
uvicorn sintetico_api.main:app --reload
```

- **Dashboard:** http://localhost:8000/ — ejecuta los 4 pilares con un
  clic, mira sus trazas en vivo (live tail vía Server-Sent Events),
  inspecciona la "waterfall" de cada ejecución, y habla con el agente
  ReAct en el panel "Agente en vivo".
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc
- **OpenAPI JSON:** http://localhost:8000/openapi.json

Sin `ANTHROPIC_API_KEY`/`OPENAI_API_KEY` en el entorno del servidor, el
panel "Agente en vivo" usa un simulador determinista que respeta el mismo
protocolo ReAct (Thought/Action/Observation): la demo completa —
dashboard incluido— funciona de principio a fin sin coste. Con una key
real configurada, ese mismo panel invoca el modelo real.

## Testing

```bash
pip install -e ".[dev,api]"
pytest tests/ -v
```

La suite cubre: presupuesto (local y Redis), orquestación, calidad
(auto-limpieza de código, TrustScore), resiliencia (circuit breakers),
gobernanza (RACI, autonomía progresiva, overrides), seguridad
(incluyendo el punto de extensión `secondary_classifier`), proveedores
reales (con un SDK de Anthropic simulado para no requerir red ni una key
real en CI), autenticación, y los endpoints HTTP de la API — incluyendo
tests de contrato compartidos entre el backend SQLite y el de Postgres
(`tests/test_trace_store_contract.py`). Cada test que corrige un bug de
una versión anterior lo documenta en su docstring como regresión.

CI (`.github/workflows/ci.yml`) corre esta suite en cada push/PR sobre
Python 3.10–3.12, más un smoke test que arranca `uvicorn` de verdad.

## Despliegue con más de una instancia (Postgres + autenticación)

Por defecto, la API usa SQLite local y no exige autenticación — perfecto
para la demo del libro. Para un despliegue real con varias instancias
del servicio:

```bash
pip install -e ".[api,postgres]"
export SINTETICO_API_DB_URL="postgresql://user:pass@host:5432/sintetico"
export SINTETICO_API_KEY="una-key-larga-y-aleatoria"
uvicorn sintetico_api.main:app --host 0.0.0.0 --port 8000
```

- Con `SINTETICO_API_DB_URL` apuntando a Postgres, el live-tail (SSE)
  funciona correctamente entre instancias vía `LISTEN`/`NOTIFY` nativo
  de Postgres — ver `sintetico_api/postgres_store.py`.
- Con `SINTETICO_API_KEY` configurada, todos los endpoints `/api/v1/*`
  (salvo `/api/v1/health`) exigen la cabecera `X-API-Key`. El dashboard
  pedirá la key mediante un modal (icono de candado) y la recordará en
  el navegador.

## Variables de entorno

Ver [`.env.example`](.env.example). Las más relevantes:

| Variable | Uso |
|---|---|
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` | Habilitan proveedores reales en demos y en la API |
| `SINTETICO_MODEL_HAIKU/SONNET/OPUS` | Sobreescriben el id de modelo real sin tocar código |
| `SINTETICO_API_DB_PATH` | Ruta del SQLite de trazas de la API (por defecto `sintetico_traces.db`) |
| `REDIS_HOST` / `REDIS_PORT` | Backend distribuido de presupuesto |

## Licencia

MIT
