package main

import "html/template"

// The page. One file, no build step, no CDN — the whole point is that a
// visitor can read what they are running.
var page = template.Must(template.New("page").Parse(`<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>simd — measured on your machine</title>
<script type="module" src="/assets/datastar.js"></script>
<style>
  :root { color-scheme: light dark; --fg:#111; --dim:#666; --line:#ddd; --acc:#0a6; --warn:#b40; }
  @media (prefers-color-scheme: dark) {
    :root { --fg:#e8e8e8; --dim:#999; --line:#333; --acc:#3d9; --warn:#f85; }
  }
  body { font: 16px/1.55 ui-sans-serif, system-ui, sans-serif; color: var(--fg);
         max-width: 60rem; margin: 0 auto; padding: 2rem 1.25rem 6rem; }
  h1 { font-size: 1.6rem; margin-bottom: .25rem; }
  .sub { color: var(--dim); margin-top: 0; }
  .machine { font: 13px/1.5 ui-monospace, monospace; color: var(--dim);
             border: 1px solid var(--line); border-radius: 6px; padding: .6rem .8rem; }
  .warn { color: var(--warn); font-weight: 600; }
  section { border-top: 1px solid var(--line); padding-top: 1.5rem; margin-top: 2rem; }
  h2 { font-size: 1.15rem; margin: 0 0 .2rem; }
  .claim { color: var(--dim); margin: 0 0 .1rem; }
  .where { color: var(--dim); font-size: .85rem; margin: 0 0 1rem; }
  .codes { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
  @media (max-width: 46rem) { .codes { grid-template-columns: 1fr; } }
  figure { margin: 0; }
  figcaption { font-size: .85rem; color: var(--dim); margin-bottom: .3rem; }
  pre { background: color-mix(in srgb, var(--fg) 6%, transparent); border-radius: 6px;
        padding: .7rem .8rem; overflow-x: auto; font: 13px/1.45 ui-monospace, monospace;
        margin: 0; }
  button { font: inherit; padding: .45rem 1rem; border-radius: 6px; cursor: pointer;
           border: 1px solid var(--acc); background: transparent; color: var(--acc); margin-top: 1rem; }
  button:hover { background: color-mix(in srgb, var(--acc) 12%, transparent); }
  .result { margin-top: 1rem; min-height: 1.5rem; }
  .result.running { color: var(--dim); font-style: italic; }
  table { border-collapse: collapse; }
  th { text-align: left; font-weight: 500; padding: .2rem 1.5rem .2rem 0; }
  td { padding: .2rem 1.5rem .2rem 0; font: 14px ui-monospace, monospace; }
  .ratio { color: var(--acc); font-weight: 700; }
  .note { color: var(--dim); font-size: .85rem; margin: .4rem 0 0; }
  footer { margin-top: 3rem; color: var(--dim); font-size: .9rem; }
</style>
</head>
<body>

<h1>simd, measured on your machine</h1>
<p class="sub">Every claim below comes from <code>docs/tutorial.md</code>. The
numbers come from running both implementations here, now.</p>

<p class="machine">{{.Machine}}{{if ge .Load 0.0}} · load {{printf "%.2f" .Load}}{{end}}</p>
{{if gt .Load 1.5}}
<p class="warn">This machine is busy (load {{printf "%.2f" .Load}}). Anything
measured now is worse than no measurement, because it will look like data.
Come back when it is idle.</p>
{{end}}

{{range .Scenarios}}
<section>
  <h2>{{.Title}}</h2>
  <p class="claim">{{.Claim}}</p>
  <p class="where">docs/tutorial.md — {{.Section}}</p>
  <div class="codes">
    <figure>
      <figcaption>{{.BaseName}}</figcaption>
      <pre>{{.BaseCode}}</pre>
    </figure>
    <figure>
      <figcaption>{{.FastName}}</figcaption>
      <pre>{{.FastCode}}</pre>
    </figure>
  </div>
  <button data-on-click="@get('/run/{{.ID}}')">Run it</button>
  <div id="result-{{.ID}}" class="result"></div>
</section>
{{end}}

<footer>
<p>Minimum of {{.Samples}} samples per implementation, never the mean —
benchmark noise is one-sided, so the fastest run is the one least interfered
with. See <code>testdata/bench/README.md</code> and
<code>docs/wrong.md</code> entry 48, which is about twenty-one phantom
regressions measured on a loaded machine.</p>
<p>Datastar is vendored in <code>cmd/site/assets/</code> and served from this
binary. Nothing here contacts a CDN.</p>
</footer>

</body>
</html>`))
