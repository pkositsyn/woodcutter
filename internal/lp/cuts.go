package lp

import "math"

const (
	cutFracTol  = 0.01 // ignore basic rows whose fractionality is ~0 or ~1
	cutCoeffEps = 1e-9 // drop cut coefficients below this magnitude
	cutMaxCoeff = 1e7  // drop ill-conditioned cuts (huge coefficients)
	fracSnap    = 1e-9 // snap near-integer tableau entries to integer
)

// frac returns the fractional part in [0,1), snapping near-integer values
// (from either side) to 0 so float noise cannot masquerade as a fraction.
func frac(v float64) float64 {
	f := v - math.Floor(v)
	if f < fracSnap || f > 1-fracSnap {
		return 0
	}
	return f
}

// allIntegerVars reports whether every structural variable is integer.
// Pure Gomory fractional cuts are only valid for pure integer programs.
func allIntegerVars(integer []bool, n int) bool {
	if len(integer) < n {
		return false
	}
	for j := 0; j < n; j++ {
		if !integer[j] {
			return false
		}
	}
	return true
}

// gomoryCuts derives Gomory fractional cuts from fractional integer basic
// variables and translates them into structural-variable space (substituting
// slack/surplus columns via their constraint definitions). Callers must ensure
// the program is pure-integer (allIntegerVars); otherwise the cuts are invalid.
func gomoryCuts(t *tableau, integer []bool) []Constraint {
	isBasic := make([]bool, t.total)
	for _, b := range t.basis {
		isBasic[b] = true
	}

	var cuts []Constraint
	for r := 0; r < len(t.rows); r++ {
		bv := t.basis[r]
		if bv >= t.n || bv >= len(integer) || !integer[bv] {
			continue
		}
		fb := frac(t.rows[r][t.total])
		if fb < cutFracTol || fb > 1-cutFracTol {
			continue
		}

		g := make([]float64, t.n)
		R := fb
		for j := 0; j < t.total; j++ {
			if isBasic[j] {
				continue // basic columns contribute 0 in exact arithmetic
			}
			fa := frac(t.rows[r][j])
			if fa == 0 {
				continue
			}
			switch t.kind[j] {
			case colStructural:
				g[j] += fa
			case colSlack: // s_k = RHS_k - coeffs_k·x
				c := t.cons[t.conOf[j]]
				for i := 0; i < t.n && i < len(c.Coeffs); i++ {
					g[i] -= fa * c.Coeffs[i]
				}
				R -= fa * c.RHS
			case colSurplus: // e_k = coeffs_k·x - RHS_k
				c := t.cons[t.conOf[j]]
				for i := 0; i < t.n && i < len(c.Coeffs); i++ {
					g[i] += fa * c.Coeffs[i]
				}
				R += fa * c.RHS
			case colArtificial:
				// nonbasic at zero, forbidden -> contributes nothing
			}
		}

		// Numerical hygiene: drop tiny coefficients and ill-conditioned cuts.
		maxC, nonzero := 0.0, false
		for i := range g {
			if math.Abs(g[i]) < cutCoeffEps {
				g[i] = 0
				continue
			}
			nonzero = true
			if a := math.Abs(g[i]); a > maxC {
				maxC = a
			}
		}
		if !nonzero || maxC > cutMaxCoeff {
			continue
		}
		cuts = append(cuts, Constraint{Coeffs: g, Type: GreaterEqual, RHS: R})
	}
	return cuts
}
