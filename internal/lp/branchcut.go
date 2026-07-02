package lp

import (
	"math"
	"time"
)

// solveMILPWarm solves the MILP by depth-first branch-and-cut over the bounded-
// variable dual-simplex engine. Branching tightens variable bounds (down:
// setUpper(j, floor); up: setLower(j, ceil)) and re-optimizes by warm-starting
// the dual simplex from the parent basis, which is far cheaper than a fresh
// two-phase solve per node.
//
// It honours the same result contract as SolveMILP: Proven=true with
// LowerBound==Objective on a fully-exhausted search; Proven=false with a valid
// global LowerBound (min LP bound over the open frontier) when the deadline or
// node cap cuts the search short. The objective is assumed integer-valued (used
// for the ceil bound), matching the cutting-stock pattern MILP.
func solveMILPWarm(p Problem, integer []bool, deadline time.Time) Solution {
	n := len(p.Objective)

	// Root problem, optionally strengthened with Gomory cuts (pure-integer only).
	rp := Problem{Objective: p.Objective, Constraints: append([]Constraint(nil), p.Constraints...)}
	if allIntegerVars(integer, n) {
		if tab, st := solveTableau(rp); st == Optimal {
			if cuts := gomoryCuts(tab, integer); len(cuts) > 0 {
				rp.Constraints = append(rp.Constraints, cuts...)
			}
		}
	}

	// Seed an incumbent with the rounding dive (typically a strong bound).
	incumbent := Solution{Status: Infeasible, Objective: math.Inf(1)}
	if allIntegerVars(integer, n) {
		if d, ok := diveHeuristic(rp, integer); ok {
			incumbent = d
		}
	}

	s := newBVSolver(rp)
	rootStatus := s.solve()
	if rootStatus != Optimal && math.IsInf(incumbent.Objective, 1) {
		// Root LP infeasible and no heuristic incumbent: whole program infeasible.
		return Solution{Status: Infeasible}
	}

	// DFS frame. On first visit a frame applies its branch bound-change (root:
	// none), solves the LP by warm start, and either prunes/accepts or pushes two
	// child frames. On second visit (children resolved) it restores the solver to
	// its parent's basis and pops. `bound` is the frame's own LP lower bound while
	// it sits unresolved on the stack — the basis of the honest global lower bound.
	type frame struct {
		expanded bool    // has this frame been solved/branched yet?
		hasBound bool    // is bound meaningful (frame solved, still on stack)?
		bound    float64 // integer LP lower bound of this frame's subproblem
		parent   bvState // solver state to restore when this frame unwinds
		root     bool    // root frame: no bound change to apply
		setLo    bool    // true: setLower(col,val); false: setUpper(col,val)
		col      int     // branch variable
		val      float64 // branch bound value
	}

	// The stack drives the DFS. We restore the solver's basis precisely when
	// unwinding a subtree so siblings warm-start from the correct parent.
	stack := []*frame{{root: true}}

	nodes := 0
	proven := true
	backstop := func() bool {
		return nodes >= milpNodeLimit || (!deadline.IsZero() && time.Now().After(deadline))
	}

	for len(stack) > 0 {
		if backstop() {
			proven = false
			break
		}
		top := stack[len(stack)-1]

		if !top.expanded {
			top.expanded = true
			// Snapshot the parent basis and apply this frame's branch bound.
			top.parent = s.snapshot()
			st := rootStatus
			if !top.root {
				if top.setLo {
					st = s.setLower(top.col, top.val)
				} else {
					st = s.setUpper(top.col, top.val)
				}
			}
			if st != Optimal {
				s.restore(top.parent) // infeasible subproblem: prune
				stack = stack[:len(stack)-1]
				continue
			}
			nodes++

			sol := s.solution()
			lb := math.Ceil(sol.Objective - 1e-9) // integer-objective lower bound
			if lb >= incumbent.Objective-eps {
				s.restore(top.parent) // cannot beat incumbent: prune
				stack = stack[:len(stack)-1]
				continue
			}
			frac := mostFractional(sol.X, integer)
			if frac == -1 {
				sol.Status = Optimal
				incumbent = sol
				s.restore(top.parent) // integer feasible: accept, unwind
				stack = stack[:len(stack)-1]
				continue
			}
			// Fractional: keep this frame on the stack as an internal node (it now
			// carries a valid LP bound) and push its two children. The solver is
			// currently at this node's basis; children snapshot it as their parent.
			// LIFO: push up then down so the down child is explored first.
			top.hasBound, top.bound = true, lb
			v := sol.X[frac]
			up := &frame{setLo: true, col: frac, val: math.Ceil(v)}
			down := &frame{setLo: false, col: frac, val: math.Floor(v)}
			stack = append(stack, up, down)
			continue
		}

		// Children resolved: unwind. Restore the solver to this frame's parent.
		s.restore(top.parent)
		stack = stack[:len(stack)-1]
	}

	if !proven {
		// Honest global lower bound: min LP bound over frames still on the stack
		// (the open, unresolved frontier). Root frame's -Inf is ignored.
		lb := math.Inf(1)
		for _, f := range stack {
			if !f.hasBound {
				continue // unsolved (or root) frame carries no valid bound
			}
			if f.bound < lb {
				lb = f.bound
			}
		}
		if math.IsInf(lb, 1) {
			lb = math.Inf(-1) // no informative frontier bound
		}
		incumbent.Proven = false
		if math.IsInf(incumbent.Objective, 1) {
			return Solution{Status: Infeasible}
		}
		if math.IsInf(lb, -1) {
			incumbent.LowerBound = incumbent.Objective
		} else {
			incumbent.LowerBound = lb
		}
		incumbent.Status = Optimal
		return incumbent
	}

	if math.IsInf(incumbent.Objective, 1) {
		return Solution{Status: Infeasible}
	}
	incumbent.Status = Optimal
	incumbent.Proven = true
	incumbent.LowerBound = incumbent.Objective
	return incumbent
}
