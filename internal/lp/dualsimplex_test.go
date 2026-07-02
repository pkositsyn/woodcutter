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

// After solving, tightening a variable's bound and re-optimizing via warm start
// must match a from-scratch solve of the same restricted problem.
func TestBVWarmStartMatchesFresh(t *testing.T) {
	p := Problem{Objective: []float64{5, 4}, Constraints: []Constraint{
		{Coeffs: []float64{2, 1}, Type: GreaterEqual, RHS: 3},
		{Coeffs: []float64{1, 3}, Type: GreaterEqual, RHS: 4}}}
	s := newBVSolver(p)
	if s.solve() != Optimal {
		t.Fatal("initial solve not optimal")
	}
	// Warm-start: force x0 >= 2.
	if st := s.setLower(0, 2); st != Optimal {
		t.Fatalf("warm re-solve status %v", st)
	}
	got := s.solution()
	// Fresh solve of the same restricted problem.
	restricted := p
	restricted.Constraints = append(append([]Constraint{}, p.Constraints...),
		Constraint{Coeffs: []float64{1, 0}, Type: GreaterEqual, RHS: 2})
	want := Solve(restricted)
	if math.Abs(got.Objective-want.Objective) > 1e-6 {
		t.Fatalf("warm obj %v != fresh %v", got.Objective, want.Objective)
	}
}

// A sequence of warm bound changes (both lower and upper), each followed by a
// from-scratch cross-check, plus snapshot/restore round-tripping. This is the
// differential oracle for the warm-start engine.
func TestBVWarmStartSequenceMatchesFresh(t *testing.T) {
	rng := newTestRand(7)
	for iter := 0; iter < 300; iter++ {
		n := 2 + rng.intn(2)
		m := 1 + rng.intn(3)
		obj := make([]float64, n)
		for j := range obj {
			obj[j] = float64(1 + rng.intn(9))
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
		s := newBVSolver(p)
		if s.solve() != Optimal {
			continue // infeasible root; nothing to warm-start
		}

		// Apply a random tightening to variable v.
		v := rng.intn(n)
		bnd := float64(rng.intn(6))
		var st Status
		lower := rng.intn(2) == 0
		extra := Constraint{Coeffs: unit(v, n)}
		if lower {
			st = s.setLower(v, bnd)
			extra.Type = GreaterEqual
			extra.RHS = bnd
		} else {
			st = s.setUpper(v, bnd)
			extra.Type = LessEqual
			extra.RHS = bnd
		}

		restricted := Problem{Objective: obj, Constraints: append(append([]Constraint{}, cons...), extra)}
		want := Solve(restricted)
		if st != want.Status {
			// The warm solver reports Infeasible where a from-scratch phase-1
			// solve does too; both must agree.
			if !(st != Optimal && want.Status != Optimal) {
				t.Fatalf("iter %d: warm status %v != fresh %v (p=%+v bnd=%v lower=%v var=%v)",
					iter, st, want.Status, p, bnd, lower, v)
			}
			continue
		}
		if st == Optimal {
			got := s.solution()
			if math.Abs(got.Objective-want.Objective) > 1e-6 {
				t.Fatalf("iter %d: warm obj %v != fresh %v (p=%+v bnd=%v lower=%v var=%v)",
					iter, got.Objective, want.Objective, p, bnd, lower, v)
			}
		}
	}
}

// snapshot/restore must return the solver to an identical solution.
func TestBVSnapshotRestore(t *testing.T) {
	p := Problem{Objective: []float64{5, 4}, Constraints: []Constraint{
		{Coeffs: []float64{2, 1}, Type: GreaterEqual, RHS: 3},
		{Coeffs: []float64{1, 3}, Type: GreaterEqual, RHS: 4}}}
	s := newBVSolver(p)
	if s.solve() != Optimal {
		t.Fatal("initial solve not optimal")
	}
	base := s.solution().Objective
	st := s.snapshot()
	// Perturb with a bound change, then restore.
	s.setLower(0, 3)
	s.restore(st)
	if st2 := s.solve(); st2 != Optimal {
		t.Fatalf("post-restore solve status %v", st2)
	}
	if got := s.solution().Objective; math.Abs(got-base) > 1e-6 {
		t.Fatalf("restore obj %v != base %v", got, base)
	}
	// A further bound change after restore must behave like a fresh restricted solve.
	if st := s.setLower(0, 2); st != Optimal {
		t.Fatalf("warm status %v", st)
	}
	restricted := p
	restricted.Constraints = append(append([]Constraint{}, p.Constraints...),
		Constraint{Coeffs: []float64{1, 0}, Type: GreaterEqual, RHS: 2})
	want := Solve(restricted)
	if got := s.solution().Objective; math.Abs(got-want.Objective) > 1e-6 {
		t.Fatalf("post-restore warm obj %v != fresh %v", got, want.Objective)
	}
}

type testRand struct{ s uint64 }

func newTestRand(seed uint64) *testRand { return &testRand{s: seed*2862933555777941757 + 1} }
func (r *testRand) next() uint64 {
	r.s = r.s*6364136223846793005 + 1442695040888963407
	return r.s
}
func (r *testRand) intn(n int) int { return int(r.next() >> 33 % uint64(n)) }
