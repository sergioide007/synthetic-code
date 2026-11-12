# Notebooks

These two notebooks belong to the Spanish edition of the book (*Código sintético*): their prose is in Spanish, the code
is the same `sintetico` package. English-speaking readers should start with [`examples/`](../examples), which covers
the same ground and is tested in CI.

Both install the real package from the repository (`pip install -e ".[all]"` after `git clone`) and import from
`sintetico` / `trazabilidad`.

- `codigo_sintetico_colab.ipynb`: installation, test suite, the four pillars, a savings demo against a real API
  (optional), and how to start the observability dashboard from Colab.
- `codigo_sintetico_trazabilidad.ipynb`: structured logger + ReAct agent, with a real provider if you configure an API
  key, or a deterministic simulator if you do not.

**Before using them**, edit the cell with `REPO_URL = "https://github.com/<your-user>/<your-repo>.git"` at the top of
each notebook and set the URL of your fork.
