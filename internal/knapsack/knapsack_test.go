package knapsack

import (
	"encoding/json"
	"math"
	"os"
	"testing"
)

// kCase mirrors one entry in testdata/cases.json, whose expected value/vec were
// produced by running the actual Python reference solve_bounded_knapsack_optimized.
type kCase struct {
	Values   []float64 `json:"values"`
	Weights  []int     `json:"weights"`
	Counts   []int     `json:"counts"`
	Capacity int       `json:"capacity"`
	Value    float64   `json:"value"`
	Vec      []int     `json:"vec"`
}

func TestSolveMatchesPythonReference(t *testing.T) {
	data, err := os.ReadFile("testdata/cases.json")
	if err != nil {
		t.Fatal(err)
	}
	var cases []kCase
	if err := json.Unmarshal(data, &cases); err != nil {
		t.Fatal(err)
	}
	if len(cases) == 0 {
		t.Fatal("no cases in testdata/cases.json")
	}
	for idx, c := range cases {
		val, vec := Solve(c.Values, c.Weights, c.Counts, c.Capacity)
		if math.Abs(val-c.Value) > 1e-9 {
			t.Fatalf("case %d: value %v != python %v (in %+v)", idx, val, c.Value, c)
		}
		if len(vec) != len(c.Vec) {
			t.Fatalf("case %d: vec len %d != %d (%v vs %v)", idx, len(vec), len(c.Vec), vec, c.Vec)
		}
		for i := range vec {
			if vec[i] != c.Vec[i] {
				t.Fatalf("case %d: vec %v != python %v", idx, vec, c.Vec)
			}
		}
	}
}

func TestSolveExactSmall(t *testing.T) {
	val, vec := Solve([]float64{10}, []int{3}, []int{3}, 10)
	if val != 30 || vec[0] != 3 {
		t.Fatalf("got (%v,%v), want (30,[3])", val, vec)
	}
}
