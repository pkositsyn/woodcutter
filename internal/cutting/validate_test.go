package cutting

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// assertPlanValid checks supply limits, demand fulfilment, and board geometry.
func assertPlanValid(t *testing.T, plan Plan, stock, reqs []Pair, padding int) {
	t.Helper()

	supply := map[int]int{}
	for _, s := range stock {
		supply[s.Length] += s.Count
	}
	maxStock := 0
	for L := range supply {
		if L > maxStock {
			maxStock = L
		}
	}

	used := map[int]int{}
	produced := map[int]int{}
	total := 0
	for _, b := range plan.Boards {
		used[b.StockLength]++
		total += b.StockLength

		pos := 0
		for _, p := range b.Pieces {
			if p.Start != pos {
				t.Fatalf("piece start %d != running pos %d", p.Start, pos)
			}
			if p.End != p.Start+p.Length {
				t.Fatalf("piece end %d != start+length", p.End)
			}
			pos = p.End
			if p.Kind == PieceCut {
				produced[p.Length]++
			}
		}
		if pos != b.StockLength {
			t.Fatalf("board fills %d, want %d", pos, b.StockLength)
		}
	}
	if total != plan.TotalMaterial {
		t.Fatalf("sum of boards %d != TotalMaterial %d", total, plan.TotalMaterial)
	}
	for L, u := range used {
		if u > supply[L] {
			t.Fatalf("stock %d used %d > supply %d", L, u, supply[L])
		}
	}
	for _, r := range reqs {
		if r.Length > maxStock {
			continue // legitimately dropped
		}
		if produced[r.Length] < r.Count {
			t.Fatalf("demand %d mm: produced %d < required %d", r.Length, produced[r.Length], r.Count)
		}
	}
}

func TestGoldenCrossCheck(t *testing.T) {
	files, err := filepath.Glob("testdata/*.txt")
	if err != nil || len(files) == 0 {
		t.Fatalf("no testdata: %v", err)
	}
	for _, f := range files {
		name := strings.TrimSuffix(filepath.Base(f), ".txt")
		t.Run(name, func(t *testing.T) {
			data, err := os.ReadFile(f)
			if err != nil {
				t.Fatal(err)
			}
			stock, reqs, err := ParseInput(strings.NewReader(string(data)))
			if err != nil {
				t.Fatal(err)
			}
			plan := Solve(stock, reqs, Options{Padding: 5})

			gb, err := os.ReadFile("testdata/" + name + ".golden.json")
			if err != nil {
				t.Fatalf("missing golden (run reference/gen_golden.py): %v", err)
			}
			var g struct {
				TotalMaterial int  `json:"total_material"`
				Feasible      bool `json:"feasible"`
			}
			if err := json.Unmarshal(gb, &g); err != nil {
				t.Fatal(err)
			}
			if plan.Feasible != g.Feasible {
				t.Fatalf("feasible %v != golden %v", plan.Feasible, g.Feasible)
			}
			if plan.TotalMaterial != g.TotalMaterial {
				t.Fatalf("total_material %d != golden %d", plan.TotalMaterial, g.TotalMaterial)
			}
			if plan.Feasible {
				assertPlanValid(t, plan, stock, reqs, 5)
			}
		})
	}
}
