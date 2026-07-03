package cutting

import (
	"fmt"
	"math"
	"sort"
	"time"

	"github.com/pkositsyn/woodcutter/internal/knapsack"
	"github.com/pkositsyn/woodcutter/internal/lp"
)

const tolerance = 1e-6

// defaultMILPTimeout bounds the wall-clock time of the integer solve when
// Options.MILPTimeout is not set, guaranteeing prompt termination even on
// pathological inputs.
const defaultMILPTimeout = 30 * time.Second

// Solve is a Go port of the reference min_material_cutting: Gilmore-Gomory
// column generation over multiple stock lengths with limited supply, then an
// integer solve over the generated pattern pool.
func Solve(stock, requirements []Pair, opts Options) Plan {
	padding := opts.Padding

	// Aggregate stock by length, sort descending.
	stockAgg := map[int]int{}
	for _, s := range stock {
		stockAgg[s.Length] += s.Count
	}
	if len(stockAgg) == 0 {
		return Plan{Feasible: true, TotalMaterial: 0, Proven: true}
	}
	stockLen := make([]int, 0, len(stockAgg))
	for L := range stockAgg {
		stockLen = append(stockLen, L)
	}
	sort.Sort(sort.Reverse(sort.IntSlice(stockLen)))
	nStock := len(stockLen)
	stockSup := make([]int, nStock)
	capacity := make([]int, nStock)
	for k, L := range stockLen {
		stockSup[k] = stockAgg[L]
		capacity[k] = L + padding
	}
	maxStock := stockLen[0]

	// Filter requirements longer than the longest board; sort desc by length.
	var valid, dropped []Pair
	for _, r := range requirements {
		if r.Length <= maxStock {
			valid = append(valid, r)
		} else {
			dropped = append(dropped, r)
		}
	}
	sort.Slice(valid, func(i, j int) bool { return valid[i].Length > valid[j].Length })
	if len(valid) == 0 {
		return Plan{Feasible: true, TotalMaterial: 0, Dropped: dropped, Proven: true}
	}

	nReq := len(valid)
	reqLen := make([]int, nReq)
	reqCnt := make([]int, nReq)
	reqW := make([]int, nReq)
	for i, r := range valid {
		reqLen[i] = r.Length
		reqCnt[i] = r.Count
		reqW[i] = r.Length + padding
	}

	// Pattern pool.
	var patterns [][]int
	var patStock []int
	existing := map[string]bool{}
	addPattern := func(k int, vec []int) bool {
		key := fmt.Sprintf("%d|%v", k, vec)
		if existing[key] {
			return false
		}
		existing[key] = true
		cp := append([]int(nil), vec...)
		patterns = append(patterns, cp)
		patStock = append(patStock, k)
		return true
	}

	// Initial greedy single-type patterns.
	for k := 0; k < nStock; k++ {
		for i := 0; i < nReq; i++ {
			if reqW[i] <= capacity[k] {
				vec := make([]int, nReq)
				vec[i] = capacity[k] / reqW[i]
				addPattern(k, vec)
			}
		}
	}

	buildConstraints := func() []lp.Constraint {
		np := len(patterns)
		cons := make([]lp.Constraint, 0, nReq+nStock)
		for i := 0; i < nReq; i++ {
			row := make([]float64, np)
			for j := 0; j < np; j++ {
				row[j] = float64(patterns[j][i])
			}
			cons = append(cons, lp.Constraint{Coeffs: row, Type: lp.GreaterEqual, RHS: float64(reqCnt[i])})
		}
		for k := 0; k < nStock; k++ {
			row := make([]float64, np)
			for j := 0; j < np; j++ {
				if patStock[j] == k {
					row[j] = 1
				}
			}
			cons = append(cons, lp.Constraint{Coeffs: row, Type: lp.LessEqual, RHS: float64(stockSup[k])})
		}
		return cons
	}
	objective := func() []float64 {
		np := len(patterns)
		obj := make([]float64, np)
		for j := 0; j < np; j++ {
			obj[j] = float64(stockLen[patStock[j]])
		}
		return obj
	}

	// Column generation.
	const maxIter = 1000
	for iter := 0; iter < maxIter; iter++ {
		sol := lp.Solve(lp.Problem{Objective: objective(), Constraints: buildConstraints()})
		if sol.Status != lp.Optimal {
			break
		}
		y := sol.Duals[:nReq]
		z := sol.Duals[nReq : nReq+nStock]
		added := false
		for k := 0; k < nStock; k++ {
			bestVal, bestVec := knapsack.Solve(y, reqW, reqCnt, capacity[k])
			reduced := float64(stockLen[k]) - bestVal - z[k]
			if reduced < -tolerance {
				if addPattern(k, bestVec) {
					added = true
				}
			}
		}
		if !added {
			break
		}
	}

	// Integer solve over the pool.
	np := len(patterns)
	integer := make([]bool, np)
	for j := range integer {
		integer[j] = true
	}
	timeout := opts.MILPTimeout
	if timeout <= 0 {
		timeout = defaultMILPTimeout
	}
	sol := lp.SolveMILP(lp.Problem{Objective: objective(), Constraints: buildConstraints()}, integer, time.Now().Add(timeout))
	if sol.Status != lp.Optimal {
		return Plan{Feasible: false, TotalMaterial: -1, Dropped: dropped}
	}
	usage := make([]int, np)
	for j := 0; j < np; j++ {
		usage[j] = int(math.Round(sol.X[j]))
	}

	// Build the detailed plan.
	var boards []Board
	total := 0
	for j := 0; j < np; j++ {
		u := usage[j]
		if u <= 0 {
			continue
		}
		k := patStock[j]
		L := stockLen[k]
		total += u * L
		for b := 0; b < u; b++ {
			var pieces []Piece
			pos := 0
			for i := 0; i < nReq; i++ {
				for c := 0; c < patterns[j][i]; c++ {
					pieces = append(pieces, Piece{Length: reqLen[i], Start: pos, End: pos + reqLen[i], Kind: PieceCut})
					pos += reqLen[i]
					if padding > 0 {
						pieces = append(pieces, Piece{Length: padding, Start: pos, End: pos + padding, Kind: PiecePadding})
						pos += padding
					}
				}
			}
			if padding > 0 && len(pieces) > 0 && pieces[len(pieces)-1].Kind == PiecePadding {
				pieces = pieces[:len(pieces)-1]
				pos -= padding
			}
			if waste := L - pos; waste > 0 {
				pieces = append(pieces, Piece{Length: waste, Start: pos, End: L, Kind: PieceWaste})
			}
			boards = append(boards, Board{StockLength: L, Pieces: pieces})
		}
	}
	return Plan{
		Feasible:      true,
		TotalMaterial: total,
		Boards:        boards,
		Dropped:       dropped,
		Proven:        sol.Proven,
		LowerBound:    int(math.Round(sol.LowerBound)),
	}
}
