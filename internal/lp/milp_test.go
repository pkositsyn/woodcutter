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
