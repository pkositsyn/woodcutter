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
