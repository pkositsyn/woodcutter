package lp

import (
	"math"
	"testing"
	"time"
)

// minimize x+y s.t. x+y >= 3.5 ; integers -> optimum 4.
func TestBranchCutRounds(t *testing.T) {
	p := Problem{
		Objective:   []float64{1, 1},
		Constraints: []Constraint{{Coeffs: []float64{1, 1}, Type: GreaterEqual, RHS: 3.5}},
	}
	s := solveMILPWarm(p, []bool{true, true}, time.Time{})
	if s.Status != Optimal {
		t.Fatalf("status = %v", s.Status)
	}
	if !approx(s.Objective, 4) {
		t.Fatalf("obj = %v, want 4", s.Objective)
	}
	for j, v := range s.X {
		if math.Abs(v-math.Round(v)) > 1e-6 {
			t.Fatalf("x[%d]=%v not integer", j, v)
		}
	}
}

// A tiny covering ILP against brute force, proven optimal.
func TestBranchCutBruteAgree(t *testing.T) {
	p := Problem{
		Objective: []float64{5, 4},
		Constraints: []Constraint{
			{Coeffs: []float64{2, 1}, Type: GreaterEqual, RHS: 3},
			{Coeffs: []float64{1, 3}, Type: GreaterEqual, RHS: 4},
		},
	}
	s := solveMILPWarm(p, []bool{true, true}, time.Time{})
	best := math.Inf(1)
	for a := 0; a <= 6; a++ {
		for b := 0; b <= 6; b++ {
			if 2*a+b >= 3 && a+3*b >= 4 {
				if c := float64(5*a + 4*b); c < best {
					best = c
				}
			}
		}
	}
	if s.Status != Optimal || !approx(s.Objective, best) {
		t.Fatalf("warm obj %v (status %v), brute %v", s.Objective, s.Status, best)
	}
	if !s.Proven {
		t.Fatalf("expected Proven=true with no deadline")
	}
	if !approx(s.LowerBound, s.Objective) {
		t.Fatalf("proven LowerBound=%v must equal Objective=%v", s.LowerBound, s.Objective)
	}
}

// Larger covering ILP vs brute force, proven.
func TestBranchCutBruteAgreeLarger(t *testing.T) {
	p := Problem{
		Objective: []float64{7, 5, 9},
		Constraints: []Constraint{
			{Coeffs: []float64{3, 2, 4}, Type: GreaterEqual, RHS: 9},
			{Coeffs: []float64{1, 4, 2}, Type: GreaterEqual, RHS: 8},
		},
	}
	s := solveMILPWarm(p, []bool{true, true, true}, time.Time{})
	best := math.Inf(1)
	for a := 0; a <= 5; a++ {
		for b := 0; b <= 5; b++ {
			for c := 0; c <= 5; c++ {
				if 3*a+2*b+4*c >= 9 && a+4*b+2*c >= 8 {
					if v := float64(7*a + 5*b + 9*c); v < best {
						best = v
					}
				}
			}
		}
	}
	if s.Status != Optimal || !approx(s.Objective, best) {
		t.Fatalf("warm obj %v (status %v), brute %v", s.Objective, s.Status, best)
	}
	if !s.Proven {
		t.Fatalf("expected Proven=true with no deadline")
	}
}

// Randomized differential test: solveMILPWarm must agree with a brute-force
// enumeration on many small covering ILPs, and prove optimality.
func TestBranchCutBruteAgreeRandom(t *testing.T) {
	rng := newTestRand(42)
	for iter := 0; iter < 200; iter++ {
		n := 2 + rng.intn(2) // 2 or 3 vars
		m := 1 + rng.intn(2) // 1 or 2 constraints
		obj := make([]float64, n)
		for j := range obj {
			obj[j] = float64(1 + rng.intn(9))
		}
		cons := make([]Constraint, m)
		for i := range cons {
			row := make([]float64, n)
			for j := range row {
				row[j] = float64(1 + rng.intn(4)) // >=1 so a bounded cover exists
			}
			cons[i] = Constraint{Coeffs: row, Type: GreaterEqual, RHS: float64(1 + rng.intn(12))}
		}
		p := Problem{Objective: obj, Constraints: cons}
		integer := make([]bool, n)
		for j := range integer {
			integer[j] = true
		}

		// Brute force over each var in 0..limit (covering: rounding all up is
		// feasible, so limit = max RHS suffices as an upper enumeration bound).
		limit := 0
		for _, c := range cons {
			if int(c.RHS) > limit {
				limit = int(c.RHS)
			}
		}
		best := bruteForceCover(cons, obj, n, limit)

		s := solveMILPWarm(p, integer, time.Time{})
		if math.IsInf(best, 1) {
			if s.Status == Optimal {
				t.Fatalf("iter %d: warm found %v but brute infeasible (p=%+v)", iter, s.Objective, p)
			}
			continue
		}
		if s.Status != Optimal || !approx(s.Objective, best) {
			t.Fatalf("iter %d: warm obj %v (status %v) != brute %v (p=%+v)",
				iter, s.Objective, s.Status, best, p)
		}
		if !s.Proven || !approx(s.LowerBound, s.Objective) {
			t.Fatalf("iter %d: expected proven optimum, got proven=%v lb=%v obj=%v",
				iter, s.Proven, s.LowerBound, s.Objective)
		}
	}
}

// bruteForceCover enumerates integer points in [0,limit]^n and returns the min
// objective of feasible points (Inf if none).
func bruteForceCover(cons []Constraint, obj []float64, n, limit int) float64 {
	x := make([]float64, n)
	best := math.Inf(1)
	var rec func(idx int)
	rec = func(idx int) {
		if idx == n {
			if feasiblePoint(cons, x) {
				o := 0.0
				for j := range obj {
					o += obj[j] * x[j]
				}
				if o < best {
					best = o
				}
			}
			return
		}
		for v := 0; v <= limit; v++ {
			x[idx] = float64(v)
			rec(idx + 1)
		}
	}
	rec(0)
	return best
}

// A past deadline must terminate promptly (no hang).
func TestBranchCutBackstopTerminates(t *testing.T) {
	p := Problem{
		Objective:   []float64{1, 1},
		Constraints: []Constraint{{Coeffs: []float64{1, 1}, Type: GreaterEqual, RHS: 3.5}},
	}
	done := make(chan Solution, 1)
	go func() { done <- solveMILPWarm(p, []bool{true, true}, time.Now().Add(-time.Second)) }()
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("solveMILPWarm did not terminate under a past deadline")
	}
}

// Under a past deadline but with a heuristic incumbent available, solveMILPWarm
// returns a feasible incumbent flagged Proven=false with LowerBound <= Objective.
func TestBranchCutBackstopGap(t *testing.T) {
	p := Problem{
		Objective: []float64{1, 1},
		Constraints: []Constraint{
			{Coeffs: []float64{2, 3}, Type: GreaterEqual, RHS: 7},
			{Coeffs: []float64{3, 2}, Type: GreaterEqual, RHS: 7},
		},
	}
	s := solveMILPWarm(p, []bool{true, true}, time.Now().Add(-time.Second))
	if s.Status != Optimal {
		t.Fatalf("status=%v, want Optimal best-effort incumbent", s.Status)
	}
	if s.Proven {
		t.Fatalf("Proven must be false under a past deadline")
	}
	if s.LowerBound > s.Objective+1e-9 {
		t.Fatalf("LowerBound %v must be <= Objective %v", s.LowerBound, s.Objective)
	}
	// Incumbent must be genuinely integer-feasible.
	if !feasiblePoint(p.Constraints, s.X) {
		t.Fatalf("backstop incumbent infeasible: %v", s.X)
	}
	for j, v := range s.X {
		if math.Abs(v-math.Round(v)) > 1e-6 {
			t.Fatalf("backstop x[%d]=%v not integer", j, v)
		}
	}
}
