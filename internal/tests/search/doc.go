// Package search holds the search tests for github.com/sebishogun/simd.
//
// They exercise only the public API, which is why they can live behind a
// package boundary at all. Tests that need an unexported detail, or a hook
// from export_test.go, stay in the root package next to what they test.
package search
