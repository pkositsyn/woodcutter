// Package knapsack solves the bounded knapsack problem via binary
// decomposition, a direct port of the reference Python implementation.
package knapsack

import "math"

// Solve returns the maximum achievable value and the chosen count of each item
// type. Items are binary-decomposed into 0/1 chunks; dp[w] is the best value
// whose chosen chunks sum to EXACTLY w (unreached weights are tracked via a
// -Inf sentinel, distinct from a genuinely reachable value of 0). When the
// full capacity is not exactly composable, the best reachable weight <=
// capacity is used instead.
func Solve(values []float64, weights []int, counts []int, capacity int) (float64, []int) {
	n := len(values)

	var binVal []float64
	var binWt []int
	type mapEntry struct{ orig, cnt int }
	var mapping []mapEntry

	for i := 0; i < n; i++ {
		j := 1
		remaining := counts[i]
		for j <= remaining {
			binVal = append(binVal, values[i]*float64(j))
			binWt = append(binWt, weights[i]*j)
			mapping = append(mapping, mapEntry{i, j})
			remaining -= j
			j *= 2
		}
		if remaining > 0 {
			binVal = append(binVal, values[i]*float64(remaining))
			binWt = append(binWt, weights[i]*remaining)
			mapping = append(mapping, mapEntry{i, remaining})
		}
	}

	binN := len(binVal)

	// dp[i][w] / reached[i][w]: best exact-weight-w value (and whether w is
	// reachable at all) using only the first i binary items. Keeping the full
	// per-item history (rather than mutating a single 1D row in place) is
	// required for correct 0/1 backtracking: a 1D row overwritten by later
	// items no longer reflects the state that produced an earlier item's
	// transition, which lets reconstruction "reuse" the same binary chunk
	// (and thus over-count its source item's bounded count).
	dp := make([][]float64, binN+1)
	reached := make([][]bool, binN+1)
	dp[0] = make([]float64, capacity+1)
	reached[0] = make([]bool, capacity+1)
	for w := 1; w <= capacity; w++ {
		dp[0][w] = math.Inf(-1)
	}
	reached[0][0] = true

	for i := 0; i < binN; i++ {
		dp[i+1] = append([]float64(nil), dp[i]...)
		reached[i+1] = append([]bool(nil), reached[i]...)
		for w := capacity; w >= binWt[i]; w-- {
			prevW := w - binWt[i]
			if !reached[i][prevW] {
				continue
			}
			cand := dp[i][prevW] + binVal[i]
			if cand > dp[i+1][w] {
				dp[i+1][w] = cand
				reached[i+1][w] = true
			}
		}
	}

	target := 0
	for w := 0; w <= capacity; w++ {
		if reached[binN][w] && dp[binN][w] > dp[binN][target] {
			target = w
		}
	}

	result := make([]int, n)
	w := target
	for i := binN; i > 0; i-- {
		// Item i-1 was taken iff its weight fits and taking it reproduces
		// dp[i][w] from a reachable dp[i-1][w-binWt[i-1]].
		if w >= binWt[i-1] && reached[i-1][w-binWt[i-1]] &&
			dp[i-1][w-binWt[i-1]]+binVal[i-1] == dp[i][w] {
			m := mapping[i-1]
			result[m.orig] += m.cnt
			w -= binWt[i-1]
		}
	}
	return dp[binN][target], result
}
