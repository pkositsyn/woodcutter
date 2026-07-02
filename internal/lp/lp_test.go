package lp

import (
	"math"
	"testing"
)

func approx(a, b float64) bool { return math.Abs(a-b) < 1e-6 }

// Classic LP: maximize 3x+5y  =>  minimize -3x-5y
// s.t. x <= 4 ; 2y <= 12 ; 3x+2y <= 18 ; x,y >= 0. Optimum (2,6), obj -36.
func TestSolveMaxProfit(t *testing.T) {
	p := Problem{
		Objective: []float64{-3, -5},
		Constraints: []Constraint{
			{Coeffs: []float64{1, 0}, Type: LessEqual, RHS: 4},
			{Coeffs: []float64{0, 2}, Type: LessEqual, RHS: 12},
			{Coeffs: []float64{3, 2}, Type: LessEqual, RHS: 18},
		},
	}
	s := Solve(p)
	if s.Status != Optimal {
		t.Fatalf("status = %v, want Optimal", s.Status)
	}
	if !approx(s.Objective, -36) {
		t.Fatalf("obj = %v, want -36", s.Objective)
	}
	if !approx(s.X[0], 2) || !approx(s.X[1], 6) {
		t.Fatalf("x = %v, want [2 6]", s.X)
	}
}

// Min with >= constraints, checks duals sign/values.
// minimize 2a+3b  s.t. a+b >= 10 ; a >= 3 ; a,b >= 0.
// Optimum a=3? cost pushes b up: a+b>=10, min 2a+3b. Cheapest is maximize a.
// a can be large but a only lower-bounded; b fills rest. Min at a=10,b=0 -> 20.
// Constraint a+b>=10 binding (dual 2), a>=3 slack (dual 0).
func TestSolveMinDuals(t *testing.T) {
	p := Problem{
		Objective: []float64{2, 3},
		Constraints: []Constraint{
			{Coeffs: []float64{1, 1}, Type: GreaterEqual, RHS: 10},
			{Coeffs: []float64{1, 0}, Type: GreaterEqual, RHS: 3},
		},
	}
	s := Solve(p)
	if s.Status != Optimal {
		t.Fatalf("status = %v", s.Status)
	}
	if !approx(s.Objective, 20) {
		t.Fatalf("obj = %v, want 20", s.Objective)
	}
	// >= duals must be non-negative; binding row dual = 2, slack row dual = 0.
	if s.Duals[0] < -1e-9 || s.Duals[1] < -1e-9 {
		t.Fatalf("duals must be >= 0 for >=: %v", s.Duals)
	}
	if !approx(s.Duals[0], 2) || !approx(s.Duals[1], 0) {
		t.Fatalf("duals = %v, want [2 0]", s.Duals)
	}
}

func TestSolveInfeasible(t *testing.T) {
	p := Problem{
		Objective: []float64{1},
		Constraints: []Constraint{
			{Coeffs: []float64{1}, Type: GreaterEqual, RHS: 5},
			{Coeffs: []float64{1}, Type: LessEqual, RHS: 2},
		},
	}
	s := Solve(p)
	if s.Status != Infeasible {
		t.Fatalf("status = %v, want Infeasible", s.Status)
	}
}

// assertFeasible checks that the returned X is actually feasible: non-negative
// and satisfying every constraint within tolerance.
func assertFeasible(t *testing.T, p Problem, s Solution) {
	t.Helper()
	for j, v := range s.X {
		if v < -1e-6 {
			t.Fatalf("x[%d]=%v is negative (infeasible)", j, v)
		}
	}
	for i, c := range p.Constraints {
		lhs := 0.0
		for j := range c.Coeffs {
			if j < len(s.X) {
				lhs += c.Coeffs[j] * s.X[j]
			}
		}
		switch c.Type {
		case Equal:
			if math.Abs(lhs-c.RHS) > 1e-6 {
				t.Fatalf("Equal constraint %d violated: %v != %v", i, lhs, c.RHS)
			}
		case LessEqual:
			if lhs > c.RHS+1e-6 {
				t.Fatalf("LessEqual constraint %d violated: %v > %v", i, lhs, c.RHS)
			}
		case GreaterEqual:
			if lhs < c.RHS-1e-6 {
				t.Fatalf("GreaterEqual constraint %d violated: %v < %v", i, lhs, c.RHS)
			}
		}
	}
}

// Regression: degenerate Equal rows (row 1 is 3x row 0) leave an artificial
// basic at zero after phase 1. The phase-1 drive-out must pivot it out on a
// well-conditioned column; an exact ==0 pivot check would instead pivot on
// float noise, blow up the tableau, and return an infeasible X (with a
// negative component) as Optimal.
func TestSolveDegenerateArtificialsFeasible(t *testing.T) {
	p := Problem{
		Objective: []float64{4, 1, 4, 5},
		Constraints: []Constraint{
			{Coeffs: []float64{1, 1, 2, 3}, Type: Equal, RHS: 4},
			{Coeffs: []float64{3, 3, 6, 9}, Type: Equal, RHS: 12},
			{Coeffs: []float64{2, 3, 1, 1}, Type: Equal, RHS: 5},
			{Coeffs: []float64{3, 1, 0, 2}, Type: LessEqual, RHS: 7},
		},
	}
	s := Solve(p)
	if s.Status != Optimal {
		t.Fatalf("status = %v, want Optimal", s.Status)
	}
	assertFeasible(t, p, s)
}

// Second degenerate case: a scaled duplicate Equal row (row 1 = 3x row 0) plus
// a GreaterEqual, again forcing a redundant artificial basic at zero. The old
// ==0 pivot check returns an infeasible X=[-6 8 6] (Optimal) on this input.
func TestSolveDegenerateScaledRowsFeasible(t *testing.T) {
	p := Problem{
		Objective: []float64{2, 1, 4},
		Constraints: []Constraint{
			{Coeffs: []float64{1, 0, 2}, Type: Equal, RHS: 6},
			{Coeffs: []float64{3, 0, 6}, Type: Equal, RHS: 18},
			{Coeffs: []float64{3, 2, 1}, Type: GreaterEqual, RHS: 4},
		},
	}
	s := Solve(p)
	if s.Status != Optimal {
		t.Fatalf("status = %v, want Optimal", s.Status)
	}
	assertFeasible(t, p, s)
}
