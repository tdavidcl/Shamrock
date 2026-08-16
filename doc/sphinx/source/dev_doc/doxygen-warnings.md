# Doxygen warnings

CI records the number of Doxygen warnings on each documentation build.
The interactive chart below loads the live history from the `metrics-history` branch (hover and zoom are enabled).

```{raw} html
<iframe src="../_static/doxygen_warnings.html" title="Doxygen warning count over time" style="width: 100%; height: 600px; border: 0;" loading="lazy"></iframe>
```

Raw data: [doxygen_warnings.json](https://raw.githubusercontent.com/Shamrock-code/Shamrock/refs/heads/metrics-history/output/doxygen_warnings.json).

The org website (`https://shamrock-code.github.io/`) is same-origin with the published docs, so it can iframe this page:

```html
<iframe src="https://shamrock-code.github.io/Shamrock/sphinx/_static/doxygen_warnings.html" title="Doxygen warning count over time" style="width: 100%; height: 600px; border: 0;" loading="lazy"></iframe>
```
