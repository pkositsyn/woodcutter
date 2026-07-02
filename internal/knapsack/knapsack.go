// Package knapsack solves the bounded knapsack problem via binary
// decomposition, a direct 1:1 port of the reference Python implementation
// (solve_bounded_knapsack_optimized). NOTE: the reference DP is unbounded over
// its binary chunks and may select more copies of an item than its count when
// capacity allows — this behavior is intentionally preserved for equivalence.
package knapsack

// Solve returns the maximum achievable value and the chosen count of each item
// type, reproducing the reference DP exactly: items are binary-decomposed,
// dp[w] is the best value whose chosen weights sum to EXACTLY w, and when the
// full capacity is not exactly composable the best reachable weight is used.
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
	dp := make([]float64, capacity+1)
	pick := make([]int, capacity+1) // binary chunk chosen to reach w, or -1
	prev := make([]int, capacity+1) // predecessor weight, or -1
	for w := range pick {
		pick[w] = -1
		prev[w] = -1
	}
	maxWeight := 0

	for w := 1; w <= capacity; w++ {
		for i := 0; i < binN; i++ {
			if binWt[i] <= w && dp[w-binWt[i]]+binVal[i] > dp[w] {
				dp[w] = dp[w-binWt[i]] + binVal[i]
				pick[w] = i
				prev[w] = w - binWt[i]
				if dp[w] > 0 && w > maxWeight {
					maxWeight = w
				}
			}
		}
	}

	target := capacity
	if dp[capacity] == 0 {
		target = maxWeight
	}

	result := make([]int, n)
	for w := target; w > 0 && pick[w] != -1; w = prev[w] {
		m := mapping[pick[w]]
		result[m.orig] += m.cnt
	}
	return dp[target], result
}
