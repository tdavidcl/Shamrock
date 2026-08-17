# History graphs

CI metrics are stored on the
[`metrics-history`](https://github.com/Shamrock-code/Shamrock/tree/metrics-history)
branch. The charts below load those live series.

## Doxygen warnings

CI records the number of Doxygen warnings produced when the documentation is
built.

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

## Compile peak RSS (top 10 files)

CI records per-translation-unit compiler peak RSS. The chart keeps only the
top 10 files of each commit: a file that drops out of that ranking is omitted
at that date (`null` y, `connectgaps: false`).

```{raw} html
<iframe
  src="../_static/compile_memory_top10.html"
  title="Top 10 compile peak RSS over time"
  width="100%"
  height="700"
  style="border: none;"
></iframe>
```

Raw Plotly JSON:
[compile_memory_top10.json](https://raw.githubusercontent.com/Shamrock-code/Shamrock/refs/heads/metrics-history/output/compile_memory_top10.json).

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
