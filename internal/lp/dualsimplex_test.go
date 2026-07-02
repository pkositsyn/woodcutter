package lp

import (
	"math"
	"testing"
)

// The bounded-variable dual simplex must agree with the two-phase Solve on the
// same LPs (objective and feasibility status).
func bvSolve(p Problem) Solution {
	s := newBVSolver(p)
	st := s.solve()
	if st != Optimal {
		return Solution{Status: st}
	}
	return s.solution()
}

func TestBVMatchesSolveKnown(t *testing.T) {
	cases := []Problem{
		{Objective: []float64{2, 3}, Constraints: []Constraint{
			{Coeffs: []float64{1, 1}, Type: GreaterEqual, RHS: 10},
			{Coeffs: []float64{1, 0}, Type: GreaterEqual, RHS: 3}}},
		{Objective: []float64{5, 4}, Constraints: []Constraint{
			{Coeffs: []float64{2, 1}, Type: GreaterEqual, RHS: 3},
			{Coeffs: []float64{1, 3}, Type: GreaterEqual, RHS: 4}}},
		{Objective: []float64{1, 1, 1}, Constraints: []Constraint{
			{Coeffs: []float64{3, 2, 1}, Type: GreaterEqual, RHS: 10},
			{Coeffs: []float64{1, 1, 1}, Type: LessEqual, RHS: 8}}},
	}
	for i, p := range cases {
		want := Solve(p)
		got := bvSolve(p)
		if got.Status != want.Status {
			t.Fatalf("case %d: status %v != %v", i, got.Status, want.Status)
		}
		if want.Status == Optimal && math.Abs(got.Objective-want.Objective) > 1e-6 {
			t.Fatalf("case %d: obj %v != %v", i, got.Objective, want.Objective)
		}
	}
}

// Randomized differential test: many small covering LPs, bvSolve vs Solve.
func TestBVMatchesSolveRandom(t *testing.T) {
	rng := newTestRand(1) // deterministic; see helper below
	for iter := 0; iter < 500; iter++ {
		n := 2 + rng.intn(3)
		m := 1 + rng.intn(3)
		obj := make([]float64, n)
		for j := range obj {
			obj[j] = float64(1 + rng.intn(9)) // non-negative costs (problem class)
		}
		cons := make([]Constraint, m)
		for i := range cons {
			row := make([]float64, n)
			for j := range row {
				row[j] = float64(rng.intn(5))
			}
			cons[i] = Constraint{Coeffs: row, Type: GreaterEqual, RHS: float64(1 + rng.intn(15))}
		}
		p := Problem{Objective: obj, Constraints: cons}
		want := Solve(p)
		got := bvSolve(p)
		if got.Status != want.Status {
			t.Fatalf("iter %d status %v != %v (p=%+v)", iter, got.Status, want.Status, p)
		}
		if want.Status == Optimal && math.Abs(got.Objective-want.Objective) > 1e-6 {
			t.Fatalf("iter %d obj %v != %v (p=%+v)", iter, got.Objective, want.Objective, p)
		}
	}
}

type testRand struct{ s uint64 }

func newTestRand(seed uint64) *testRand { return &testRand{s: seed*2862933555777941757 + 1} }
func (r *testRand) next() uint64 {
	r.s = r.s*6364136223846793005 + 1442695040888963407
	return r.s
}
func (r *testRand) intn(n int) int { return int(r.next() >> 33 % uint64(n)) }
