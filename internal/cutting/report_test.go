package cutting

import (
	"strings"
	"testing"
)

func TestWriteReportSmoke(t *testing.T) {
	plan := Plan{
		Feasible:      true,
		TotalMaterial: 6000,
		Boards: []Board{{
			StockLength: 6000,
			Pieces: []Piece{
				{Length: 2000, Start: 0, End: 2000, Kind: PieceCut},
				{Length: 5, Start: 2000, End: 2005, Kind: PiecePadding},
				{Length: 3995, Start: 2005, End: 6000, Kind: PieceWaste},
			},
		}},
	}
	var sb strings.Builder
	WriteReport(&sb, plan, []Pair{{2000, 1}}, Options{Padding: 5})
	out := sb.String()
	if !strings.Contains(out, "6000") || !strings.Contains(out, "2000") {
		t.Fatalf("report missing expected content:\n%s", out)
	}
}

func TestWriteReportUnproven(t *testing.T) {
	plan := Plan{
		Feasible: true, TotalMaterial: 10000, LowerBound: 9800, Proven: false,
		Boards: []Board{{StockLength: 10000, Pieces: []Piece{{Length: 10000, Start: 0, End: 10000, Kind: PieceCut}}}},
	}
	var sb strings.Builder
	WriteReport(&sb, plan, []Pair{{10000, 1}}, Options{Padding: 5})
	if !strings.Contains(sb.String(), "не доказана") {
		t.Fatalf("expected unproven warning, got:\n%s", sb.String())
	}
}

func TestWriteReportProvenNoWarning(t *testing.T) {
	plan := Plan{
		Feasible: true, TotalMaterial: 10000, LowerBound: 10000, Proven: true,
		Boards: []Board{{StockLength: 10000, Pieces: []Piece{{Length: 10000, Start: 0, End: 10000, Kind: PieceCut}}}},
	}
	var sb strings.Builder
	WriteReport(&sb, plan, []Pair{{10000, 1}}, Options{Padding: 5})
	if strings.Contains(sb.String(), "не доказана") {
		t.Fatalf("proven plan must not print warning:\n%s", sb.String())
	}
}

func TestWriteReportInfeasible(t *testing.T) {
	var sb strings.Builder
	WriteReport(&sb, Plan{Feasible: false, TotalMaterial: -1}, nil, Options{Padding: 5})
	if !strings.Contains(sb.String(), "недопуст") {
		t.Fatalf("infeasible report unexpected:\n%s", sb.String())
	}
}
