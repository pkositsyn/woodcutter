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
// Honest-gap contract: we do NOT assert a specific optimum or that the result
// is proven optimal. Under a short time budget on this pathological input,
// proof of optimality is not guaranteed. Instead we require termination,
// feasibility, structural validity, and — when not proven — that the
// reported lower bound puts the incumbent within a sane (<=5%) gap of
// optimal, backed by the rounding-dive heuristic.
func TestSolveHardInputTerminates(t *testing.T) {
	stock := []Pair{{6100, 30}, {5900, 30}, {5300, 30}, {4700, 30}, {4100, 30}}
	reqs := []Pair{
		{2333, 13}, {1777, 17}, {1301, 19}, {997, 23},
		{733, 29}, {511, 31}, {389, 37}, {271, 41},
	}
	opts := Options{Padding: 4, MILPTimeout: 3 * time.Second}

	start := time.Now()
	plan := Solve(stock, reqs, opts)
	elapsed := time.Since(start)
	t.Logf("elapsed=%v total=%d proven=%v lb=%d", elapsed, plan.TotalMaterial, plan.Proven, plan.LowerBound)

	if elapsed >= 8*time.Second {
		t.Fatalf("solve took %v, want prompt termination under a 3s backstop", elapsed)
	}
	if !plan.Feasible {
		t.Fatalf("expected a feasible plan, got Feasible=false")
	}
	assertPlanValid(t, plan, stock, reqs, 4)
	// Honest-gap contract: if not proven, the incumbent must be within a sane
	// gap of the reported lower bound (dive heuristic guarantees a decent plan).
	if !plan.Proven {
		if plan.LowerBound <= 0 || plan.LowerBound > plan.TotalMaterial {
			t.Fatalf("invalid lower bound %d for total %d", plan.LowerBound, plan.TotalMaterial)
		}
		gap := float64(plan.TotalMaterial-plan.LowerBound) / float64(plan.TotalMaterial)
		if gap > 0.05 {
			t.Fatalf("gap %.3f exceeds 5%%; dive heuristic too weak", gap)
		}
	}
}
