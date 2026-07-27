// Command benchcheck compares a benchmark run against a stored baseline and
// fails when something got slower.
//
// Every performance number in this repository was measured once by hand. That
// is how they were arrived at honestly and it is also how they rot: a kernel
// rewrite that costs 30% looks exactly like one that costs nothing until
// somebody re-runs the benchmark and remembers what it used to say. This makes
// the remembering the machine's job.
//
//	go test -run '^$' -bench . -count 6 ./... > new.txt
//	go run ./tools/benchcheck -baseline testdata/bench/amd64.txt new.txt
//
// The comparison is deliberately crude — the median of each benchmark's
// samples, against a percentage threshold — rather than benchstat's
// distributional test. benchstat answers "is this difference real", which
// needs enough samples to be confident and reports "~" when it is not. A gate
// has to answer "should this merge", and for that a slow-but-certain 40%
// regression and a noisy-but-likely 40% regression deserve the same answer.
//
// Noise is handled by the threshold rather than by statistics: 25% by default,
// which is far above the run-to-run spread seen on this project's benchmarks
// (typically 1-3%, occasionally 12% on the memory-bound ones) and far below
// the smallest change worth blocking a merge over.
package main

import (
	"bufio"
	"flag"
	"fmt"
	"os"
	"sort"
	"strconv"
	"strings"
)

func main() {
	baseline := flag.String("baseline", "", "the stored benchmark output to compare against")
	threshold := flag.Float64("threshold", 25, "percent slower before a benchmark fails")
	update := flag.Bool("update", false, "overwrite the baseline with the new run instead of comparing")
	flag.Parse()

	if *baseline == "" || flag.NArg() != 1 {
		fmt.Fprintln(os.Stderr, "usage: benchcheck -baseline FILE [-threshold PCT] [-update] NEW")
		os.Exit(2)
	}
	newPath := flag.Arg(0)

	if *update {
		data, err := os.ReadFile(newPath)
		if err != nil {
			fatal(err)
		}
		if err := os.WriteFile(*baseline, data, 0o644); err != nil {
			fatal(err)
		}
		fmt.Printf("baseline updated from %s\n", newPath)
		return
	}

	base, err := parse(*baseline)
	if err != nil {
		fatal(err)
	}
	cur, err := parse(newPath)
	if err != nil {
		fatal(err)
	}

	var regressed, improved, missing []string
	names := make([]string, 0, len(base))
	for n := range base {
		names = append(names, n)
	}
	sort.Strings(names)

	for _, n := range names {
		c, ok := cur[n]
		if !ok {
			// A benchmark that disappeared is a change worth noticing — it is
			// usually a rename, and a rename silently drops its own history.
			missing = append(missing, n)
			continue
		}
		b := base[n]
		if b == 0 {
			continue
		}
		delta := (c - b) / b * 100
		switch {
		case delta > *threshold:
			regressed = append(regressed,
				fmt.Sprintf("  %-52s %9.2f -> %9.2f ns/op  %+6.1f%%", n, b, c, delta))
		case delta < -*threshold:
			improved = append(improved,
				fmt.Sprintf("  %-52s %9.2f -> %9.2f ns/op  %+6.1f%%", n, b, c, delta))
		}
	}

	fmt.Printf("%d benchmarks compared against %s (threshold %.0f%%)\n",
		len(base), *baseline, *threshold)
	if len(improved) > 0 {
		fmt.Printf("\n%d faster:\n%s\n", len(improved), strings.Join(improved, "\n"))
	}
	if len(missing) > 0 {
		fmt.Printf("\n%d in the baseline and not in this run (renamed or removed):\n  %s\n",
			len(missing), strings.Join(missing, "\n  "))
	}
	if len(regressed) > 0 {
		fmt.Printf("\n%d SLOWER:\n%s\n", len(regressed), strings.Join(regressed, "\n"))
		fmt.Fprintln(os.Stderr, "\nbenchcheck: regressions above the threshold; "+
			"re-run to rule out noise, then either fix them or update the baseline "+
			"with -update and say why in the commit.")
		os.Exit(1)
	}
	fmt.Println("\nno regressions")
}

// parse reads `go test -bench` output and returns the median ns/op per
// benchmark. The median rather than the minimum: the minimum is the right
// statistic for "what can this machine do", which is what internal/perf
// measures, but a gate wants the typical case rather than the luckiest one.
func parse(path string) (map[string]float64, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	samples := map[string][]float64{}
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 1<<20), 1<<20)
	for sc.Scan() {
		fields := strings.Fields(sc.Text())
		// A result line is: Name-P  iterations  value  ns/op  [more]
		if len(fields) < 4 || !strings.HasPrefix(fields[0], "Benchmark") {
			continue
		}
		var ns float64
		for i := 2; i+1 < len(fields); i++ {
			if fields[i+1] != "ns/op" {
				continue
			}
			v, err := strconv.ParseFloat(fields[i], 64)
			if err != nil {
				continue
			}
			ns = v
			break
		}
		if ns == 0 {
			continue
		}
		// Strip the -P suffix so a run on a machine with a different core
		// count still matches the baseline.
		name := fields[0]
		if i := strings.LastIndex(name, "-"); i > 0 {
			if _, err := strconv.Atoi(name[i+1:]); err == nil {
				name = name[:i]
			}
		}
		samples[name] = append(samples[name], ns)
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	out := make(map[string]float64, len(samples))
	for n, s := range samples {
		sort.Float64s(s)
		out[n] = s[len(s)/2]
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("%s: no benchmark results found", path)
	}
	return out, nil
}

func fatal(err error) {
	fmt.Fprintln(os.Stderr, "benchcheck:", err)
	os.Exit(1)
}
