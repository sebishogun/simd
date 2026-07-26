// Package perf is a repetition tester in the style Casey Muratori uses in
// Performance-Aware Programming.
//
// # Why not just use Go's benchmarks
//
// `go test -bench` reports a mean over a fixed wall-clock budget. A mean is
// the wrong statistic for this question. Every source of noise on a machine —
// a scheduler tick, an interrupt, a migration to a cold core, a competing
// process, a frequency change — makes a run *slower*. Nothing makes it faster
// than the hardware allows. So the distribution has a hard floor and a long
// right tail, and the mean measures how busy the machine was as much as how
// fast the code is.
//
// The minimum is the useful number: it is the closest observation to what the
// hardware actually does when nothing interferes. To find it, run the thing
// repeatedly and keep going until some period passes with no new minimum —
// rather than for a fixed number of iterations, which may or may not have
// included a clean run.
//
// The spread between minimum and mean is itself informative: a wide one means
// the measurement is contended and should be distrusted.
//
// # Why bandwidth and not just time
//
// Nanoseconds tell you which of two implementations is faster. Bytes per
// second tell you *what is limiting them*, because it can be compared against
// the machine. A kernel achieving 900 GB/s is working out of L1; one stuck at
// 60 GB/s is waiting on DRAM no matter how wide its vectors are. Plotting the
// same kernel across sizes shows exactly where it falls off each cache, and
// once it has, a wider vector unit buys nothing.
package perf

import (
	"fmt"
	"math"
	"strings"
	"time"
)

// Result is the outcome of testing one function.
//
// Times are float64 nanoseconds rather than time.Duration on purpose. A batch
// of calls is timed together and divided back out, and time.Duration is an
// integer count of nanoseconds — so dividing in that type truncates every
// measurement to a whole nanosecond. For a kernel that takes two or three,
// that is a quantization coarser than the thing being measured, and it turns
// a real difference between layers into a column of identical numbers.
type Result struct {
	Name string
	// Min is the fastest observed run in nanoseconds: the best estimate of the
	// true cost.
	Min float64
	// Mean and Max describe the noise around it, also in nanoseconds.
	Mean, Max float64
	// Runs is how many batches were timed.
	Runs int
	// Batch is how many calls were timed together.
	Batch int
	// Bytes is the memory traffic of one call, used for the bandwidth figure.
	// Zero means bandwidth is not reported.
	Bytes int64
}

// GBPerSec is the bandwidth implied by the minimum time.
func (r Result) GBPerSec() float64 {
	if r.Bytes == 0 || r.Min == 0 {
		return 0
	}
	return float64(r.Bytes) / (r.Min * 1e-9) / 1e9
}

// Noise is the ratio of mean to minimum. Close to 1 means a quiet machine and
// a trustworthy number; much above it means the measurement was contended.
func (r Result) Noise() float64 {
	if r.Min == 0 {
		return 0
	}
	return r.Mean / r.Min
}

// Options tunes the search.
type Options struct {
	// Patience is how long to keep going without seeing a new minimum before
	// concluding the floor has been found.
	Patience time.Duration
	// MaxTime bounds the whole search, so a pathological case cannot hang.
	MaxTime time.Duration
	// Inner is how many calls to time together. Timing a single call that
	// takes a handful of nanoseconds measures the clock, not the code, so
	// short functions are batched and the total divided back out.
	Inner int
}

// DefaultOptions are reasonable for kernels in the nanosecond to microsecond
// range.
func DefaultOptions() Options {
	return Options{Patience: 300 * time.Millisecond, MaxTime: 3 * time.Second, Inner: 0}
}

// Measure runs f until its minimum time stops improving.
func Measure(name string, bytes int64, f func(), opt Options) Result {
	if opt.Patience == 0 {
		opt = DefaultOptions()
	}
	inner := opt.Inner
	if inner == 0 {
		inner = calibrate(f)
	}

	r := Result{Name: name, Min: math.Inf(1), Bytes: bytes, Batch: inner}
	var total float64
	start := time.Now()
	lastImprovement := start

	for {
		t0 := time.Now()
		for i := 0; i < inner; i++ {
			f()
		}
		// Divide in floating point: the per-call cost of these kernels is a
		// few nanoseconds, so integer division would quantize it away.
		d := float64(time.Since(t0).Nanoseconds()) / float64(inner)

		r.Runs++
		total += d
		if d < r.Min {
			r.Min = d
			lastImprovement = time.Now()
		}
		if d > r.Max {
			r.Max = d
		}
		now := time.Now()
		if now.Sub(lastImprovement) > opt.Patience || now.Sub(start) > opt.MaxTime {
			break
		}
	}
	r.Mean = total / float64(r.Runs)
	return r
}

// calibrate picks a batch size so that one timed group lasts long enough for
// the clock to resolve it well.
//
// A millisecond is a million nanoseconds, so a batch of that length carries
// about six significant figures even from a timer with single-nanosecond
// granularity. That is what makes a difference of a fraction of a nanosecond
// per call measurable at all.
func calibrate(f func()) int {
	n := 1
	for range 40 {
		t0 := time.Now()
		for i := 0; i < n; i++ {
			f()
		}
		if time.Since(t0) > time.Millisecond {
			return n
		}
		n *= 2
	}
	return n
}

// Table renders results as a fixed-width table.
func Table(rs []Result) string {
	var b strings.Builder
	withBytes := false
	for _, r := range rs {
		if r.Bytes > 0 {
			withBytes = true
		}
	}
	if withBytes {
		fmt.Fprintf(&b, "%-34s %12s %10s %8s\n", "", "min", "GB/s", "noise")
	} else {
		fmt.Fprintf(&b, "%-34s %12s %8s\n", "", "min", "noise")
	}
	for _, r := range rs {
		if withBytes {
			fmt.Fprintf(&b, "%-34s %10.3f ns %10.1f %7.2fx\n",
				r.Name, r.Min, r.GBPerSec(), r.Noise())
		} else {
			fmt.Fprintf(&b, "%-34s %10.3f ns %7.2fx\n", r.Name, r.Min, r.Noise())
		}
	}
	return b.String()
}

// Compare renders two sets side by side with the speedup between them.
func Compare(baseName string, base []Result, name string, other []Result) string {
	var b strings.Builder
	fmt.Fprintf(&b, "%-24s %12s %12s %9s\n", "", baseName, name, "speedup")
	for i := range base {
		if i >= len(other) {
			break
		}
		x, y := base[i].Min, other[i].Min
		ratio := 0.0
		if y > 0 {
			ratio = x / y
		}
		fmt.Fprintf(&b, "%-24s %10.3f ns %10.3f ns %8.2fx\n",
			base[i].Name, x, y, ratio)
	}
	return b.String()
}
