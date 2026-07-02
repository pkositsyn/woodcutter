package lp

import (
	"container/heap"
	"math"
	"time"
)

// milpNodeLimit caps branch-and-bound nodes so pathological inputs cannot hang.
// Hitting it yields a best-effort incumbent (Proven=false).
const milpNodeLimit = 5_000_000

// bbNode is a branch & bound node: branch-bound constraints layered on the root
// problem, plus a lower bound (its parent's LP objective) used as the heap key.
type bbNode struct {
	extra []Constraint
	bound float64
}

func (n bbNode) child(c Constraint, bound float64) bbNode {
	ex := make([]Constraint, len(n.extra)+1)
	copy(ex, n.extra)
	ex[len(n.extra)] = c
	return bbNode{extra: ex, bound: bound}
}

// nodeHeap is a min-heap of nodes keyed by lower bound (best-bound search).
type nodeHeap []bbNode

func (h nodeHeap) Len() int           { return len(h) }
func (h nodeHeap) Less(i, j int) bool { return h[i].bound < h[j].bound }
func (h nodeHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *nodeHeap) Push(x any)        { *h = append(*h, x.(bbNode)) }
func (h *nodeHeap) Pop() any {
	old := *h
	n := len(old)
	it := old[n-1]
	*h = old[:n-1]
	return it
}

// SolveMILP solves the MILP by best-bound LP-relaxation branch & bound.
// integer[j]==true forces x[j] integer. Minimization only; the objective is
// assumed integer-valued (used for the ceil bound). A non-zero deadline (or the
// node cap) bounds wall-clock time: on backstop the best incumbent so far is
// returned with Proven=false. Optimality is proven when the smallest remaining
// node bound cannot beat the incumbent.
func SolveMILP(p Problem, integer []bool, deadline time.Time) Solution {
	rp := Problem{Objective: p.Objective, Constraints: append([]Constraint(nil), p.Constraints...)}

	incumbent := Solution{Status: Infeasible, Objective: math.Inf(1)}
	nodes := 0
	proven := true
	lowerBound := math.Inf(-1)

	h := &nodeHeap{{extra: nil, bound: math.Inf(-1)}}
	for h.Len() > 0 {
		if nodes >= milpNodeLimit || (!deadline.IsZero() && time.Now().After(deadline)) {
			proven = false
			lowerBound = (*h)[0].bound // smallest remaining bound (min-heap root)
			break
		}
		node := heap.Pop(h).(bbNode)
		if node.bound >= incumbent.Objective-eps {
			// Best-bound: every remaining node is >= this -> incumbent optimal.
			break
		}
		nodes++

		cons := make([]Constraint, 0, len(rp.Constraints)+len(node.extra))
		cons = append(cons, rp.Constraints...)
		cons = append(cons, node.extra...)
		sol := Solve(Problem{Objective: rp.Objective, Constraints: cons})
		if sol.Status != Optimal {
			continue
		}
		lb := math.Ceil(sol.Objective - 1e-9) // integer-objective lower bound
		if lb >= incumbent.Objective-eps {
			continue // cannot beat incumbent
		}
		frac := mostFractional(sol.X, integer)
		if frac == -1 {
			sol.Status = Optimal
			incumbent = sol
			continue
		}
		v := sol.X[frac]
		down := node.child(Constraint{Coeffs: unit(frac, len(rp.Objective)), Type: LessEqual, RHS: math.Floor(v)}, lb)
		up := node.child(Constraint{Coeffs: unit(frac, len(rp.Objective)), Type: GreaterEqual, RHS: math.Ceil(v)}, lb)
		heap.Push(h, up)
		heap.Push(h, down)
	}

	if math.IsInf(incumbent.Objective, 1) {
		return Solution{Status: Infeasible}
	}
	incumbent.Proven = proven
	if proven {
		incumbent.LowerBound = incumbent.Objective
	} else if math.IsInf(lowerBound, -1) {
		incumbent.LowerBound = incumbent.Objective // no better info
	} else {
		incumbent.LowerBound = lowerBound
	}
	return incumbent
}

// mostFractional returns the integer-constrained variable closest to 0.5, or -1
// if the point is integer-feasible.
func mostFractional(x []float64, integer []bool) int {
	frac := -1
	bestDist := math.Inf(1)
	for j := range x {
		if j < len(integer) && integer[j] {
			d := x[j] - math.Floor(x[j])
			if d > 1e-6 && d < 1-1e-6 {
				if dist := math.Abs(d - 0.5); dist < bestDist {
					bestDist, frac = dist, j
				}
			}
		}
	}
	return frac
}

func unit(idx, n int) []float64 {
	v := make([]float64, n)
	v[idx] = 1
	return v
}
