// Command site serves a page that runs this library's benchmarks on the
// machine visiting it, beside the code each one measures.
//
//	go run ./cmd/site
//
// # Why it runs them rather than quoting them
//
// Every performance number in this repository is qualified by the machine it
// came from, because that is the only honest way to state one. A page that
// quotes numbers measured elsewhere is telling a visitor about someone else's
// laptop. This one measures theirs, and shows the code beside the result so
// the question it answers is the one people actually have: how much speed does
// this small amount of extra code buy, here?
//
// # Honesty
//
// The rules are the ones testdata/bench/README.md and docs/wrong.md entry 48
// set out, and they are enforced here rather than hoped for:
//
//   - The **minimum** of several samples, never the mean. A benchmark's noise
//     is one-sided — interference can only make a run slower — so the minimum
//     is the closest thing to the machine's real capability.
//   - The machine and the tier are printed beside every number.
//   - A high load average is a **warning on the page**, not a footnote. Entry
//     48 is about twenty-one phantom regressions measured on a busy machine;
//     a number from a loaded box is worse than no number, because it looks
//     like data.
//
// # No CDN
//
// Datastar is vendored in assets/ and served from this binary. The repository
// works offline, and the JavaScript a visitor runs is in the tree where it can
// be read. The bundle contains one external URL, https://data-star.dev/errors,
// which appears in its error messages as a documentation link and is not
// fetched.
package main

import (
	"embed"
	"flag"
	"fmt"
	"html/template"
	"log"
	"net/http"
	"os"
	"runtime"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/sebishogun/simd"
)

//go:embed assets
var assets embed.FS

func main() {
	addr := flag.String("addr", "localhost:8080", "address to serve on")
	samples := flag.Int("samples", 5, "benchmark samples per implementation; the minimum is reported")
	flag.Parse()

	all := scenarios()
	byID := map[string]Scenario{}
	for _, s := range all {
		byID[s.ID] = s
	}

	mux := http.NewServeMux()
	mux.Handle("/assets/", http.FileServerFS(assets))
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/" {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		if err := page.Execute(w, pageData{
			Scenarios: all,
			Machine:   machine(),
			Load:      loadAverage(),
			Samples:   *samples,
		}); err != nil {
			log.Printf("template: %v", err)
		}
	})
	mux.HandleFunc("/run/{id}", func(w http.ResponseWriter, r *http.Request) {
		s, ok := byID[r.PathValue("id")]
		if !ok {
			http.NotFound(w, r)
			return
		}
		runScenario(w, s, *samples)
	})

	log.Printf("simd benchmark site on http://%s", *addr)
	log.Printf("  machine: %s", machine())
	srv := &http.Server{Addr: *addr, Handler: mux, ReadHeaderTimeout: 5 * time.Second}
	if err := srv.ListenAndServe(); err != nil {
		log.Fatal(err)
	}
}

type pageData struct {
	Scenarios []Scenario
	Machine   string
	Load      float64
	Samples   int
}

// machine labels the result. Not the hostname — a number is about a CPU and an
// instruction set, and naming the box adds nothing a reader can use.
func machine() string {
	return fmt.Sprintf("%s/%s, %d cores, %s, %s",
		runtime.GOOS, runtime.GOARCH, runtime.NumCPU(), runtime.Version(), simd.Describe())
}

// loadAverage returns the one-minute load, or -1 where it cannot be read.
// A number measured on a busy machine looks exactly like a number measured on
// an idle one, which is why this is on the page rather than in a log.
func loadAverage() float64 {
	b, err := os.ReadFile("/proc/loadavg")
	if err != nil {
		return -1
	}
	f := strings.Fields(string(b))
	if len(f) == 0 {
		return -1
	}
	v, err := strconv.ParseFloat(f[0], 64)
	if err != nil {
		return -1
	}
	return v
}

// runScenario measures both implementations and streams the result back as
// Datastar fragments.
func runScenario(w http.ResponseWriter, s Scenario, samples int) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	flusher, _ := w.(http.Flusher)

	send := func(html string) {
		fmt.Fprint(w, "event: datastar-merge-fragments\n")
		for _, line := range strings.Split(html, "\n") {
			fmt.Fprintf(w, "data: fragments %s\n", line)
		}
		fmt.Fprint(w, "\n")
		if flusher != nil {
			flusher.Flush()
		}
	}

	send(fmt.Sprintf(`<div id="result-%s" class="result running">measuring, %d samples each…</div>`,
		s.ID, samples))

	base := measure(s, s.base, samples)
	fast := measure(s, s.fast, samples)

	ratio := float64(base) / float64(fast)
	warn := ""
	if l := loadAverage(); l > 1.5 {
		warn = fmt.Sprintf(`<p class="warn">Load average is %.2f. These numbers are `+
			`not trustworthy — something else is using this machine.</p>`, l)
	}

	send(fmt.Sprintf(`<div id="result-%s" class="result">
<table>
<tr><th>%s</th><td>%s</td><td></td></tr>
<tr><th>%s</th><td>%s</td><td class="ratio">%.2f×</td></tr>
</table>
<p class="note">n = %s, minimum of %d samples on this machine.</p>
%s
</div>`,
		s.ID,
		template.HTMLEscapeString(s.BaseName), duration(base),
		template.HTMLEscapeString(s.FastName), duration(fast), ratio,
		commas(s.N), samples, warn))
}

// measure returns the minimum time per operation over several samples.
//
// The minimum rather than the mean, because benchmark noise is one-sided: a
// scheduler, another process or a thermal event can only ever make a run
// slower. The fastest run is the one least interfered with, which is the
// closest available estimate of what the machine can do.
func measure(s Scenario, f func(any), samples int) time.Duration {
	state := s.setup(s.N)
	best := time.Duration(1<<62 - 1)
	for range samples {
		r := testing.Benchmark(func(b *testing.B) {
			for b.Loop() {
				f(state)
			}
		})
		if d := r.T / time.Duration(max(r.N, 1)); d < best {
			best = d
		}
	}
	return best
}

func duration(d time.Duration) string {
	switch {
	case d < time.Microsecond:
		return fmt.Sprintf("%d ns", d.Nanoseconds())
	case d < time.Millisecond:
		return fmt.Sprintf("%.1f µs", float64(d.Nanoseconds())/1e3)
	default:
		return fmt.Sprintf("%.2f ms", float64(d.Nanoseconds())/1e6)
	}
}

func commas(n int) string {
	s := strconv.Itoa(n)
	var b strings.Builder
	for i, c := range s {
		if i > 0 && (len(s)-i)%3 == 0 {
			b.WriteByte(',')
		}
		b.WriteRune(c)
	}
	return b.String()
}
