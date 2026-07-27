// Command csvscan is a complete program: it parses a CSV column of numbers and
// reports statistics on it, doing both halves with this library.
//
// It is here rather than in the README because it shows the shape of a real
// use, which is two phases with different characters:
//
//   - Finding structure is a byte scan. IndexAll locates every delimiter in one
//     pass and writes their offsets, which is the step a vector unit is
//     genuinely good at — one compare per sixteen bytes rather than a branch
//     per byte.
//   - Summarising is arithmetic over a slice, where the reductions each make a
//     single pass and none of them allocate.
//
// Run it:
//
//	go run ./docs/examples/csvscan
package main

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/sebishogun/simd"
)

const data = `date,region,revenue
2026-01-03,north,1420.50
2026-01-04,south,980.25
2026-01-05,north,2310.00
2026-01-06,east,175.75
2026-01-07,south,3050.10
2026-01-08,west,640.00
2026-01-09,north,1875.40
2026-01-10,east,2200.85
`

func main() {
	lines := strings.Split(strings.TrimSpace(data), "\n")
	header, rows := lines[0], lines[1:]

	// Which column is "revenue"? Find the commas in the header the same way we
	// will find them in every row.
	col := fieldIndex(header, "revenue")
	if col < 0 {
		fmt.Println("no revenue column")
		return
	}

	// One allocation for the whole parse, reused for every row. A CSV line has
	// far fewer commas than bytes, so this is generously sized at 64.
	offsets := make([]int32, 64)

	values := make([]float64, 0, len(rows))
	for _, row := range rows {
		f := field(row, col, offsets)
		v, err := strconv.ParseFloat(f, 64)
		if err != nil {
			fmt.Printf("skipping %q: %v\n", f, err)
			continue
		}
		values = append(values, v)
	}

	fmt.Printf("rows      %d\n", len(values))
	fmt.Printf("total     %10.2f\n", simd.Sum(values))
	fmt.Printf("mean      %10.2f\n", simd.Mean(values))
	fmt.Printf("stddev    %10.2f\n", simd.StdDev(values))

	lo, hi := simd.MinMax(values)
	fmt.Printf("range     %10.2f .. %.2f\n", lo, hi)
	fmt.Printf("best row  %d\n", simd.ArgMax(values))

	// Which rows are more than one standard deviation above the mean? A
	// comparison writes the mask; CompressInto packs the values that passed.
	// The two together are the vectorized form of a filter, and the same pair
	// works for any predicate you can express as a comparison.
	cut := simd.Mean(values) + simd.StdDev(values)
	mask := make([]bool, len(values))
	simd.GreaterScalarInto(mask, values, cut)

	high := make([]float64, len(values))
	n := simd.CompressInto(high, values, mask)
	fmt.Printf("above %.2f: %v\n", cut, high[:n])
}

// fieldIndex reports which comma-separated field of line equals name.
func fieldIndex(line, name string) int {
	offsets := make([]int32, 64)
	n := simd.IndexAll(offsets, line, ',')
	prev, idx := 0, 0
	for _, off := range offsets[:n] {
		if line[prev:off] == name {
			return idx
		}
		prev, idx = int(off)+1, idx+1
	}
	if line[prev:] == name {
		return idx
	}
	return -1
}

// field returns the i-th comma-separated field of line.
//
// offsets is passed in rather than allocated here so that parsing a million
// rows allocates once rather than a million times — the whole reason the
// library's own functions take a destination.
func field(line string, i int, offsets []int32) string {
	n := simd.IndexAll(offsets, line, ',')
	prev := 0
	for k, off := range offsets[:n] {
		if k == i {
			return line[prev:off]
		}
		prev = int(off) + 1
	}
	if i == n {
		return line[prev:]
	}
	return ""
}
