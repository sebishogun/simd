// Package benchmarks holds every benchmark in this repository.
//
// They live apart from the library for two reasons. They are a gate rather
// than a test — `make bench-check` compares them against a recorded baseline
// and they need a quiet machine to mean anything — and they only ever call the
// public API, so nothing is lost by putting a package boundary between them
// and the implementation.
//
// It is under internal/ so that it does not appear on pkg.go.dev as something
// a caller might import.
//
//	make bench-run      # run them, write the raw output
//	make bench-check    # and compare against testdata/bench/<goarch>.txt
//
// benchcheck matches on benchmark name rather than package path, so moving
// these files did not invalidate the recorded baseline.
package benchmarks
