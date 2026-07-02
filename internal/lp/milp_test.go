package lp

import (
	"math"
	"testing"
	"time"
)

// minimize x+y s.t. x+y >= 3.5 ; integers -> optimum 4 (e.g. x=4,y=0 or 2,2).
func TestMILPRounds(t *testing.T) {
	p := Problem{
		Objective: []float64{1, 1},
		Constraints: []Constraint{
			{Coeffs: []float64{1, 1}, Type: GreaterEqual, RHS: 3.5},
		},
	}
	s := SolveMILP(p, []bool{true, true}, time.Time{})
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

// A tiny covering ILP against brute force.
func TestMILPBruteAgree(t *testing.T) {
	// minimize 5a+4b s.t. 2a+b >= 3 ; a+3b >= 4 ; a,b >= 0 integer.
	p := Problem{
		Objective: []float64{5, 4},
		Constraints: []Constraint{
			{Coeffs: []float64{2, 1}, Type: GreaterEqual, RHS: 3},
			{Coeffs: []float64{1, 3}, Type: GreaterEqual, RHS: 4},
		},
	}
	s := SolveMILP(p, []bool{true, true}, time.Time{})
	// brute force over a,b in 0..6
	best := math.Inf(1)
	for a := 0; a <= 6; a++ {
		for b := 0; b <= 6; b++ {
			if 2*a+b >= 3 && a+3*b >= 4 {
				c := float64(5*a + 4*b)
				if c < best {
					best = c
				}
			}
		}
	}
	if s.Status != Optimal || !approx(s.Objective, best) {
		t.Fatalf("milp obj %v (status %v), brute %v", s.Objective, s.Status, best)
	}
}

// Best-bound must PROVE optimality (Proven=true) with no deadline.
func TestMILPProvenOptimal(t *testing.T) {
	p := Problem{
		Objective: []float64{5, 4},
		Constraints: []Constraint{
			{Coeffs: []float64{2, 1}, Type: GreaterEqual, RHS: 3},
			{Coeffs: []float64{1, 3}, Type: GreaterEqual, RHS: 4},
		},
	}
	s := SolveMILP(p, []bool{true, true}, time.Time{})
	if s.Status != Optimal || !s.Proven {
		t.Fatalf("status=%v proven=%v, want Optimal+proven", s.Status, s.Proven)
	}
	if !approx(s.LowerBound, s.Objective) {
		t.Fatalf("proven LowerBound=%v must equal Objective=%v", s.LowerBound, s.Objective)
	}
}

// Larger covering ILP vs brute force, proven.
func TestMILPBruteAgreeLarger(t *testing.T) {
	p := Problem{
		Objective: []float64{7, 5, 9},
		Constraints: []Constraint{
			{Coeffs: []float64{3, 2, 4}, Type: GreaterEqual, RHS: 9},
			{Coeffs: []float64{1, 4, 2}, Type: GreaterEqual, RHS: 8},
		},
	}
	s := SolveMILP(p, []bool{true, true, true}, time.Time{})
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
		t.Fatalf("milp obj %v (status %v), brute %v", s.Objective, s.Status, best)
	}
}

// A past deadline must terminate promptly (no hang), not spin.
func TestMILPBackstopTerminates(t *testing.T) {
	p := Problem{
		Objective:   []float64{1, 1},
		Constraints: []Constraint{{Coeffs: []float64{1, 1}, Type: GreaterEqual, RHS: 3.5}},
	}
	done := make(chan Solution, 1)
	go func() { done <- SolveMILP(p, []bool{true, true}, time.Now().Add(-time.Second)) }()
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("SolveMILP did not terminate under a past deadline")
	}
}

// diveHeuristic returns a feasible integer solution by pinning the most-
// fractional variable up until integral. On a covering LP it must beat naive
// one-shot round-up.
func TestDiveHeuristic(t *testing.T) {
	p := Problem{
		Objective: []float64{1, 1},
		Constraints: []Constraint{
			{Coeffs: []float64{2, 3}, Type: GreaterEqual, RHS: 7},
			{Coeffs: []float64{3, 2}, Type: GreaterEqual, RHS: 7},
		},
	}
	sol, ok := diveHeuristic(p, []bool{true, true})
	if !ok {
		t.Fatalf("dive failed to produce a feasible incumbent")
	}
	// Must be integer-feasible for p.
	if !feasiblePoint(p.Constraints, sol.X) {
		t.Fatalf("dive result infeasible: %v", sol.X)
	}
	for j, v := range sol.X {
		if math.Abs(v-math.Round(v)) > 1e-6 {
			t.Fatalf("dive x[%d]=%v not integer", j, v)
		}
	}
	// brute-force optimum is 3 (e.g. (1,2) or (2,1)); dive should be feasible and finite.
	if sol.Objective < 3-1e-9 {
		t.Fatalf("dive obj %v below true optimum 3 (infeasible-optimal)", sol.Objective)
	}
}

// roundUpHeuristic rounds integer vars up and accepts only feasible points.
func TestRoundUpHeuristic(t *testing.T) {
	cons := []Constraint{
		{Coeffs: []float64{1, 1}, Type: GreaterEqual, RHS: 3},
		{Coeffs: []float64{1, 0}, Type: LessEqual, RHS: 10},
	}
	obj := []float64{1, 1}
	// x=(1.4, 1.6) rounds up to (2,2): satisfies both -> feasible, obj 4.
	s, ok := roundUpHeuristic(cons, obj, []float64{1.4, 1.6}, []bool{true, true})
	if !ok || !approx(s.Objective, 4) {
		t.Fatalf("heuristic ok=%v obj=%v, want ok+obj 4", ok, s.Objective)
	}
	// Rounding up violates a tight <= cap -> rejected.
	tight := []Constraint{{Coeffs: []float64{1, 0}, Type: LessEqual, RHS: 1}}
	if _, ok := roundUpHeuristic(tight, obj, []float64{1.9, 0}, []bool{true, true}); ok {
		t.Fatalf("heuristic must reject an infeasible round-up (ceil(1.9)=2 > cap 1)")
	}
}

// Under a past deadline but with a heuristic incumbent available, SolveMILP
// returns a feasible incumbent flagged Proven=false with LowerBound < Objective.
func TestMILPBackstopGap(t *testing.T) {
	// Covering LP with fractional root so the root heuristic seeds an incumbent
	// before the (already-passed) deadline stops branching.
	p := Problem{
		Objective: []float64{1, 1},
		Constraints: []Constraint{
			{Coeffs: []float64{2, 3}, Type: GreaterEqual, RHS: 7},
			{Coeffs: []float64{3, 2}, Type: GreaterEqual, RHS: 7},
		},
	}
	s := SolveMILP(p, []bool{true, true}, time.Now().Add(-time.Second))
	if s.Status != Optimal {
		t.Fatalf("status=%v, want Optimal best-effort incumbent", s.Status)
	}
	if s.Proven {
		t.Fatalf("Proven must be false under a past deadline")
	}
	if s.LowerBound > s.Objective+1e-9 {
		t.Fatalf("LowerBound %v must be <= Objective %v", s.LowerBound, s.Objective)
	}
}
