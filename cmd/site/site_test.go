package main

import (
	"html/template"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// Every scenario must actually run. A page listing a scenario whose handler
// panics, or reports a degenerate ratio, is worse than a page listing fewer.
func TestEveryScenarioRuns(t *testing.T) {
	for _, s := range scenarios() {
		t.Run(s.ID, func(t *testing.T) {
			// A small n and one sample: this checks the plumbing, not the
			// speed. The page uses the real sizes.
			small := s
			small.N = 4096
			w := httptest.NewRecorder()
			runScenario(w, small, 1)

			body := w.Body.String()
			if !strings.Contains(body, "datastar-merge-fragments") {
				t.Fatal("no Datastar event in the response")
			}
			if !strings.Contains(body, "result-"+s.ID) {
				t.Errorf("the fragment does not target result-%s", s.ID)
			}
			if !strings.Contains(body, "ratio") {
				t.Error("no ratio reported")
			}
			if strings.Contains(body, "Inf") || strings.Contains(body, "NaN") {
				t.Errorf("a degenerate ratio was reported:\n%s", body)
			}
		})
	}
}

// Both sides of a scenario must actually do work. A baseline the compiler
// deleted would give a spectacular ratio and measure nothing, which is the
// most likely way a page like this becomes a lie.
func TestScenariosDoWork(t *testing.T) {
	for _, s := range scenarios() {
		t.Run(s.ID, func(t *testing.T) {
			a, b := s.setup(2048), s.setup(2048)
			s.base(a)
			s.fast(b)
			if !wroteSomething(a) {
				t.Error("the baseline wrote nothing")
			}
			if !wroteSomething(b) {
				t.Error("the library implementation wrote nothing")
			}
		})
	}
}

func wroteSomething(state any) bool {
	switch v := state.(type) {
	case *soa:
		for _, x := range v.e {
			if x != 0 {
				return true
			}
		}
		for _, x := range v.x {
			if x != 0 {
				return true
			}
		}
	case *aos:
		for _, x := range v.e {
			if x != 0 {
				return true
			}
		}
	case [2]any:
		return wroteSomething(v[0]) || wroteSomething(v[1])
	}
	return false
}

// The page must render, name the machine, and give every scenario a button.
func TestPageRenders(t *testing.T) {
	all := scenarios()
	var sb strings.Builder
	if err := page.Execute(&sb, pageData{
		Scenarios: all, Machine: machine(), Load: loadAverage(), Samples: 5,
	}); err != nil {
		t.Fatalf("template: %v", err)
	}
	body := sb.String()
	for _, s := range all {
		// Escaped, because the template escapes: "Fuse, don't chain" reaches
		// the page as "Fuse, don&#39;t chain".
		if !strings.Contains(body, template.HTMLEscapeString(s.Title)) {
			t.Errorf("scenario %q is missing from the page", s.ID)
		}
		if !strings.Contains(body, "@get('/run/"+s.ID+"')") {
			t.Errorf("scenario %q has no run button", s.ID)
		}
	}
	// The honesty requirements are part of the page rather than decoration,
	// so their absence is a failure.
	for _, want := range []string{"Minimum of", "Datastar is vendored", "entry 48"} {
		if !strings.Contains(body, want) {
			t.Errorf("the page does not mention %q", want)
		}
	}
}

// The load warning is the whole point of entry 48: it has to appear when the
// machine is busy and stay away when it is not.
func TestLoadWarningThreshold(t *testing.T) {
	for _, c := range []struct {
		load float64
		warn bool
	}{{0.2, false}, {1.0, false}, {1.6, true}, {17.0, true}, {-1, false}} {
		var sb strings.Builder
		if err := page.Execute(&sb, pageData{
			Scenarios: scenarios(), Machine: "test", Load: c.load, Samples: 3,
		}); err != nil {
			t.Fatal(err)
		}
		got := strings.Contains(sb.String(), "This machine is busy")
		if got != c.warn {
			t.Errorf("load %.1f: warning shown = %v, want %v", c.load, got, c.warn)
		}
	}
}

// The bundle must come out of the binary. If this starts failing because the
// file is missing, the page silently loses its interactivity rather than
// erroring, which is why it is checked here.
func TestAssetsAreEmbedded(t *testing.T) {
	mux := http.NewServeMux()
	mux.Handle("/assets/", http.FileServerFS(assets))
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, httptest.NewRequest("GET", "/assets/datastar.js", nil))
	if w.Code != 200 {
		t.Fatalf("datastar.js: status %d", w.Code)
	}
	if w.Body.Len() < 10000 {
		t.Errorf("datastar.js is %d bytes; that is not the browser bundle", w.Body.Len())
	}
	// A browser bundle, not the unbundled source: bare relative imports would
	// mean the page loads a module that immediately 404s.
	if strings.Contains(w.Body.String(), "from '../") {
		t.Error("datastar.js has bare relative imports; it is the source, not the bundle")
	}
}
