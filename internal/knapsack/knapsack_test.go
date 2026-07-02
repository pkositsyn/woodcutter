package knapsack

import (
	"math"
	"math/rand"
	"testing"
)

// brute enumerates every count combination within limits and returns the best
// value whose exact total weight is <= capacity (mirrors the reference DP's
// exact-weight semantics is NOT required here: brute is a looser oracle used
// only to sanity-check that Solve never exceeds capacity and never beats the
// true optimum). We therefore only assert Solve's value <= brute optimum and
// that the returned vector is self-consistent.
func bruteBestValue(values []float64, weights, counts []int, capacity int) float64 {
	n := len(values)
	best := 0.0
	var rec func(i, w int, v float64)
	rec = func(i, w int, v float64) {
		if i == n {
			if v > best {
				best = v
			}
			return
		}
		for c := 0; c <= counts[i]; c++ {
			nw := w + c*weights[i]
			if nw > capacity {
				break
			}
			rec(i+1, nw, v+float64(c)*values[i])
		}
	}
	rec(0, 0, 0)
	return best
}

func TestSolveExactSmall(t *testing.T) {
	// One item type, weight 3, value 10, up to 3 copies, capacity 10.
	// Exact-weight DP: best exact weight <= 10 is 9 (three copies), value 30.
	val, vec := Solve([]float64{10}, []int{3}, []int{3}, 10)
	if val != 30 {
		t.Fatalf("value = %v, want 30", val)
	}
	if vec[0] != 3 {
		t.Fatalf("vec = %v, want [3]", vec)
	}
}

func TestSolveVectorConsistent(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	for iter := 0; iter < 200; iter++ {
		n := 1 + rng.Intn(4)
		values := make([]float64, n)
		weights := make([]int, n)
		counts := make([]int, n)
		for i := 0; i < n; i++ {
			values[i] = float64(1 + rng.Intn(20))
			weights[i] = 1 + rng.Intn(6)
			counts[i] = 1 + rng.Intn(4)
		}
		capacity := 1 + rng.Intn(25)
		val, vec := Solve(values, weights, counts, capacity)
		// Reconstructed value matches vec dotted with values.
		var got, wsum float64
		for i := 0; i < n; i++ {
			got += float64(vec[i]) * values[i]
			wsum += float64(vec[i] * weights[i])
		}
		if math.Abs(got-val) > 1e-9 {
			t.Fatalf("value %v != reconstructed %v (vec %v)", val, got, vec)
		}
		if int(wsum) > capacity {
			t.Fatalf("weight %v exceeds capacity %d (vec %v)", wsum, capacity, vec)
		}
		if val > bruteBestValue(values, weights, counts, capacity)+1e-9 {
			t.Fatalf("value %v beats brute optimum", val)
		}
	}
}
