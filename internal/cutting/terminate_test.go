package cutting

import (
	"testing"
	"time"
)

// TestSolveHardInputTerminates reproduces the input that previously hung the
// MILP branch & bound (8 coprime requirement types create a huge search
// tree). With a wall-clock deadline wired through Options.MILPTimeout, the
// solver must return promptly with a feasible best-effort incumbent instead
// of running for minutes.
//
// NOTE: we deliberately do NOT assert TotalMaterial against a specific value.
// Under a short time budget on this pathological input, optimality is not
// guaranteed (mirrors Python reference's CBC timeLimit best-effort
// behavior) — only termination, feasibility, and structural validity are
// the contract being tested here.
func TestSolveHardInputTerminates(t *testing.T) {
	stock := []Pair{{6100, 30}, {5900, 30}, {5300, 30}, {4700, 30}, {4100, 30}}
	reqs := []Pair{
		{2333, 13}, {1777, 17}, {1301, 19}, {997, 23},
		{733, 29}, {511, 31}, {389, 37}, {271, 41},
	}
	opts := Options{Padding: 4, MILPTimeout: 2 * time.Second}

	start := time.Now()
	plan := Solve(stock, reqs, opts)
	elapsed := time.Since(start)

	t.Logf("elapsed=%v totalMaterial=%d feasible=%v", elapsed, plan.TotalMaterial, plan.Feasible)

	if elapsed >= 10*time.Second {
		t.Fatalf("solve took %v, want < 10s (termination guarantee failed)", elapsed)
	}
	if !plan.Feasible {
		t.Fatalf("expected a feasible best-effort incumbent within the time budget, got Feasible=false")
	}
	if plan.TotalMaterial <= 0 {
		t.Fatalf("TotalMaterial = %d, want > 0", plan.TotalMaterial)
	}
	assertPlanValid(t, plan, stock, reqs, 4)
}
