package simd

import (
	"math"

	"github.com/sebishogun/simd/internal/kernel"
)

// Streaming reductions, for data that arrives in pieces.
//
// Everything else in this package takes a whole slice. That is the right shape
// when you have one, and it is the wrong shape for a file being read, a socket,
// or a column stored in chunks — there the alternative is buffering the entire
// input first, which defeats the point.
//
// # The promise, and why it constrains the design
//
// **A chunked result equals the whole-slice result, bit for bit, whatever the
// chunk boundaries.** Feeding 1000 elements as one chunk, as ten chunks of a
// hundred, or as a thousand chunks of one gives the same float64 — the same
// bits, not the same value to within rounding.
//
// That is not free, and it is why the state below is sixteen accumulators and
// a count rather than a running total. Floating-point addition is not
// associative, so [Sum] fixes the order: it keeps [kernel.SumLanes]
// accumulators and element i contributes to lane i%SumLanes, then folds them
// with a fixed binary tree. A streaming form has to reproduce that exactly,
// which means carrying all sixteen lanes across a chunk boundary *and*
// remembering how many elements have been seen, because the next chunk's first
// element belongs to lane count%16 rather than lane 0.
//
// A single running total would be simpler, smaller and wrong: its answer would
// depend on how the caller happened to split the input, which is exactly the
// class of bug this library exists to not have.

// Accumulator is a resumable sum over float32 or float64.
//
// The zero value is ready to use. Add it chunks in order; [Accumulator.Sum],
// [Accumulator.Mean] and [Accumulator.Count] can be read at any point and do
// not disturb the accumulation.
//
//	var acc simd.Accumulator[float64]
//	for {
//		n, err := read(buf)
//		acc.Add(buf[:n])
//		if err != nil {
//			break
//		}
//	}
//	fmt.Println(acc.Sum(), acc.Mean())
//
// The result equals [Sum] over the concatenation, bit for bit, however the
// chunks fell. It allocates nothing.
type Accumulator[T Float] struct {
	// acc holds the same sixteen partial sums the whole-slice reduction uses.
	acc [kernel.SumLanes]T
	// n is the number of elements seen, which is what makes the lane
	// assignment continue across a chunk boundary rather than restart.
	n int
}

// Add folds a chunk into the accumulator.
//
// It processes the chunk in three parts, and the split is what preserves the
// lane assignment: a head of elements up to the next multiple of SumLanes,
// then whole blocks through the vectorized kernel, then a tail. Only the
// middle part is accelerated, which is the right trade — the head and tail are
// at most fifteen elements each per call.
func (s *Accumulator[T]) Add(a []T) {
	i := 0
	// Head: finish the block the previous chunk left partly filled.
	for ; i < len(a) && s.n%kernel.SumLanes != 0; i++ {
		s.acc[s.n%kernel.SumLanes] += a[i]
		s.n++
	}
	// Body: whole blocks, lane-aligned, straight into the accumulators. The
	// kernel adds to them rather than overwriting, which is what keeps each
	// lane a single sequential sum across chunk boundaries — grouping the
	// per-chunk partials and adding those would give a different last bit.
	if blocks := (len(a) - i) / kernel.SumLanes * kernel.SumLanes; blocks > 0 {
		ops[T]().SumLanes(s.acc[:], a[i:i+blocks])
		i += blocks
		s.n += blocks
	}
	// Tail.
	for ; i < len(a); i++ {
		s.acc[s.n%kernel.SumLanes] += a[i]
		s.n++
	}
}

// Sum returns the sum of everything added so far.
//
// It folds a copy of the accumulators, so it can be called at any point
// without disturbing the accumulation.
func (s *Accumulator[T]) Sum() T {
	acc := s.acc
	return kernel.CombineTree(&acc)
}

// Count returns how many elements have been added.
func (s *Accumulator[T]) Count() int { return s.n }

// Mean returns the arithmetic mean of everything added so far, or zero if
// nothing has been.
func (s *Accumulator[T]) Mean() T {
	if s.n == 0 {
		return 0
	}
	return s.Sum() / T(s.n)
}

// Reset returns the accumulator to its zero state, keeping any allocation.
// There is none to keep, which is the point — it is here so a caller can reuse
// one in a loop without thinking about it.
func (s *Accumulator[T]) Reset() { *s = Accumulator[T]{} }

// MinMaxAccumulator is a resumable minimum and maximum.
//
// The zero value is ready to use. Unlike [Accumulator] this needs no lane
// discipline: minimum and maximum are associative and commutative, so no
// grouping is observable and the state is just the two values.
//
// NaN propagates the way [Min] and [Max] define it: a NaN anywhere in the
// input makes both results NaN, and it cannot be un-seen by a later chunk.
type MinMaxAccumulator[T Float] struct {
	min, max T
	seen     bool
}

// Add folds a chunk in.
func (s *MinMaxAccumulator[T]) Add(a []T) {
	if len(a) == 0 {
		return
	}
	lo, hi := MinMax(a)
	if !s.seen {
		s.min, s.max, s.seen = lo, hi, true
		return
	}
	// math.Min and math.Max propagate NaN, which is what Min and Max promise
	// and what a plain comparison would silently drop.
	s.min = T(math.Min(float64(s.min), float64(lo)))
	s.max = T(math.Max(float64(s.max), float64(hi)))
}

// MinMax returns the minimum and maximum seen so far, and whether anything has
// been added. Both are zero when nothing has.
func (s *MinMaxAccumulator[T]) MinMax() (min, max T, ok bool) {
	return s.min, s.max, s.seen
}

// Reset returns the accumulator to its zero state.
func (s *MinMaxAccumulator[T]) Reset() { *s = MinMaxAccumulator[T]{} }

// IntAccumulator is a resumable integer sum.
//
// Separate from [Accumulator] because integer addition is associative, so
// there is no lane discipline to preserve and the state is one value. It wraps
// on overflow, like [Sum] and like the hardware.
type IntAccumulator[T Integer] struct {
	sum T
	n   int
}

// Add folds a chunk in.
func (s *IntAccumulator[T]) Add(a []T) {
	s.sum += Sum(a)
	s.n += len(a)
}

// Sum returns the sum of everything added so far.
func (s *IntAccumulator[T]) Sum() T { return s.sum }

// Count returns how many elements have been added.
func (s *IntAccumulator[T]) Count() int { return s.n }

// Reset returns the accumulator to its zero state.
func (s *IntAccumulator[T]) Reset() { *s = IntAccumulator[T]{} }
