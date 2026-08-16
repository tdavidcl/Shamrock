# Doxygen warning history

CI records the number of Doxygen warnings produced when the documentation is
built. The chart below loads the live series from the
[`metrics-history`](https://github.com/Shamrock-code/Shamrock/tree/metrics-history)
branch.

```{raw} html
<iframe
  src="../_static/doxygen_warnings.html"
  title="Doxygen warning count over time"
  width="100%"
  height="600"
  style="border: none;"
></iframe>
```

Raw JSON:
[doxygen_warnings.json](https://raw.githubusercontent.com/Shamrock-code/Shamrock/refs/heads/metrics-history/output/doxygen_warnings.json).

The org website (`https://shamrock-code.github.io/`) is same-origin with the
published Sphinx docs, so it can embed this chart with:

```html
<iframe
  src="https://shamrock-code.github.io/Shamrock/sphinx/_static/doxygen_warnings.html"
  title="Doxygen warning count over time"
  width="100%"
  height="600"
  style="border: none;"
></iframe>
```
