package simd

import (
	"runtime"
	"sync"
)

// Parallel variants.
//
// Everything else in this library runs on one core, deliberately. A function
// that spawns goroutines takes them from the caller, and the caller usually has
// a better idea than the callee of where the parallelism should live — a batch
// of a thousand small multiplies wants one goroutine per multiply, not a
// thousand fan-outs of eight. A library that fans out internally turns that
// into oversubscription and gets slower.
//
// So parallelism here is opt-in and lives in its own name. The default stays
// composable; you reach for these when the multiply in front of you is the
// whole job.
//
// Only matrix multiplication is offered, and the reason is the numerical
// contract rather than effort. Splitting a reduction across goroutines means
// combining partial sums, and the number of partials would be GOMAXPROCS — so
// Sum would return different bits on a machine with a different core count,
// which is the one thing this library promises never happens. Matrix
// multiplication has no such problem: the work divides by output row, and the
// summation over k inside each element is untouched, so the result is
// bit-identical to [MatMulInto] and TestMatMulParallelIsBitIdentical checks
// that rather than assuming it.

// matMulParallelMinWork is the multiply-accumulate count below which the
// goroutines cost more than they save.
//
// Measured on a 32-thread Zen 5, square float64, serial against parallel:
//
//	n=128     2.1M    0.88x   <- parallel is slower
//	n=256    16.8M    3.93x
//	n=512   134.2M   13.00x
//	n=1024    1.07G   15.68x
//
// The first version of this constant was 1<<20, which put n=128 just inside the
// parallel path and cost 12%. It is set above that now. Below the threshold the
// serial kernel runs, so calling this on a small multiply is pointless rather
// than harmful.
const matMulParallelMinWork = 4 << 20

// MatMulParallelInto is [MatMulInto] across several goroutines.
//
// dst is m×n, a is m×k and b is k×n, all row-major, and the result is
// bit-identical to [MatMulInto] on the same input — the work is divided by
// output row, so no element's accumulation order changes.
//
// It uses up to GOMAXPROCS goroutines and returns once they are done. Below
// roughly a million multiply-accumulates, or when GOMAXPROCS is 1, it runs the
// serial kernel instead.
//
// Use this when the multiply is the whole job. If you are already running work
// in parallel — one goroutine per matrix in a batch, say — use [MatMulInto] and
// keep the parallelism where you can see it.
func MatMulParallelInto[T Number](dst, a, b []T, m, k, n int) {
	workers := runtime.GOMAXPROCS(0)
	if workers > m {
		workers = m
	}
	// int64 because m*n*k overflows an int well before it stops being worth
	// parallelising, and a wrapped negative product would read as "too small".
	if workers < 2 || int64(m)*int64(n)*int64(k) < matMulParallelMinWork ||
		len(dst) < m*n || len(a) < m*k || len(b) < k*n {
		// The length checks are not validation. MatMulInto does not reject a
		// short destination — it writes nothing — and this must behave the same
		// way rather than inventing a panic the serial call never had. Handing
		// the whole call over is also what keeps any future panic in the
		// caller's goroutine, where it can be recovered, instead of inside a
		// worker where it would take the program down.
		MatMulInto(dst, a, b, m, k, n)
		return
	}

	rowsPer := (m + workers - 1) / workers
	var wg sync.WaitGroup
	for r0 := 0; r0 < m; r0 += rowsPer {
		r1 := min(r0+rowsPer, m)
		wg.Add(1)
		go func(r0, r1 int) {
			defer wg.Done()
			MatMulInto(dst[r0*n:r1*n], a[r0*k:r1*k], b, r1-r0, k, n)
		}(r0, r1)
	}
	wg.Wait()
}

// gemvParallelMinWork is the multiply-accumulate count below which fanning out
// a matrix-vector product costs more than it saves.
//
// Measured on a 32-thread Zen 5, square float64, serial against parallel:
//
//	n=512      262K   1.00x   <- below the threshold, runs serial
//	n=1024     1.05M  2.62x
//	n=2048     4.19M  5.79x
//	n=4096    16.8M   3.34x
//
// Gemv touches each matrix element once, so it is bound by memory rather than
// arithmetic, and the speedup falls away again at 4096 where the matrix no
// longer fits anywhere useful and the bus is the limit rather than the cores.
const gemvParallelMinWork = 1 << 20

// GemvParallelInto is [GemvInto] across several goroutines.
//
// dst is length m, a is m×k row-major and x is length k, and the result is
// bit-identical to [GemvInto] — the work divides by output row, so no element's
// summation over k changes order.
//
// The same advice as [MatMulParallelInto] applies: use it when this product is
// the whole job, and stay with [GemvInto] when you are already running work in
// parallel yourself.
func GemvParallelInto[T Number](dst, a, x []T, m, k int) {
	workers := runtime.GOMAXPROCS(0)
	if workers > m {
		workers = m
	}
	if workers < 2 || int64(m)*int64(k) < gemvParallelMinWork ||
		len(dst) < m || len(a) < m*k || len(x) < k {
		GemvInto(dst, a, x, m, k)
		return
	}

	rowsPer := (m + workers - 1) / workers
	var wg sync.WaitGroup
	for r0 := 0; r0 < m; r0 += rowsPer {
		r1 := min(r0+rowsPer, m)
		wg.Add(1)
		go func(r0, r1 int) {
			defer wg.Done()
			GemvInto(dst[r0:r1], a[r0*k:r1*k], x, r1-r0, k)
		}(r0, r1)
	}
	wg.Wait()
}
