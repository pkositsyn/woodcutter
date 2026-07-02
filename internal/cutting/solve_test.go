package cutting

import "testing"

// One 6000mm board type, unlimited-ish supply, need three 2000mm pieces.
// With padding 5, two 2000mm+kerf pieces fit one board (2*2005=4010 <= 6005),
// three pieces need two boards -> total_material 12000.
func TestSolveSimple(t *testing.T) {
	stock := []Pair{{6000, 100}}
	reqs := []Pair{{2000, 3}}
	plan := Solve(stock, reqs, Options{Padding: 5})
	if !plan.Feasible {
		t.Fatal("expected feasible")
	}
	if plan.TotalMaterial != 12000 {
		t.Fatalf("total = %d, want 12000", plan.TotalMaterial)
	}
	assertPlanValid(t, plan, stock, reqs, 5)
}

func TestSolveInfeasibleSupply(t *testing.T) {
	// Need three 2000mm pieces but only one 6000mm board -> cannot satisfy.
	stock := []Pair{{6000, 1}}
	reqs := []Pair{{2000, 3}}
	plan := Solve(stock, reqs, Options{Padding: 5})
	if plan.Feasible || plan.TotalMaterial != -1 {
		t.Fatalf("expected infeasible (-1), got feasible=%v total=%d", plan.Feasible, plan.TotalMaterial)
	}
}

func TestSolveDropsTooLong(t *testing.T) {
	stock := []Pair{{3000, 10}}
	reqs := []Pair{{5000, 1}, {1000, 2}}
	plan := Solve(stock, reqs, Options{Padding: 5})
	if !plan.Feasible {
		t.Fatal("expected feasible")
	}
	if len(plan.Dropped) != 1 || plan.Dropped[0].Length != 5000 {
		t.Fatalf("dropped = %v, want one 5000", plan.Dropped)
	}
}

func TestSolveEmptyStock(t *testing.T) {
	plan := Solve(nil, []Pair{{1000, 1}}, Options{Padding: 5})
	if !plan.Feasible || plan.TotalMaterial != 0 {
		t.Fatalf("empty stock: feasible=%v total=%d", plan.Feasible, plan.TotalMaterial)
	}
}
