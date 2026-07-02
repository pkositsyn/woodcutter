package cutting

import (
	"fmt"
	"io"
	"sort"
	"strings"
)

// WriteReport prints a clean cutting report: per-board layout, boards by stock
// type, requirement fulfilment, and waste summary.
func WriteReport(w io.Writer, plan Plan, requirements []Pair, opts Options) {
	if !plan.Feasible {
		fmt.Fprintln(w, "Задача недопустима: запасов склада не хватает под спрос.")
		return
	}

	for _, d := range plan.Dropped {
		fmt.Fprintf(w, "Пропущено (длиннее максимальной доски): %d мм × %d\n", d.Length, d.Count)
	}

	boardsByStock := map[int]int{}
	produced := map[int]int{}
	for _, b := range plan.Boards {
		boardsByStock[b.StockLength]++
		for _, p := range b.Pieces {
			if p.Kind == PieceCut {
				produced[p.Length]++
			}
		}
	}

	fmt.Fprintf(w, "Детальный план раскроя (%d досок, %d мм материала):\n", len(plan.Boards), plan.TotalMaterial)
	for idx, b := range plan.Boards {
		fmt.Fprintf(w, "\nДоска #%d [%d мм]:\n", idx+1, b.StockLength)
		var cuts []string
		waste := 0
		for _, p := range b.Pieces {
			switch p.Kind {
			case PieceCut:
				cuts = append(cuts, fmt.Sprintf("%d мм (%d-%d)", p.Length, p.Start, p.End))
			case PieceWaste:
				waste += p.Length
			}
		}
		fmt.Fprintf(w, "  Куски: %s\n", strings.Join(cuts, ", "))
		fmt.Fprintf(w, "  Отходы: %d мм\n", waste)
	}

	fmt.Fprintln(w, "\nИспользовано досок по типам:")
	types := make([]int, 0, len(boardsByStock))
	for L := range boardsByStock {
		types = append(types, L)
	}
	sort.Sort(sort.Reverse(sort.IntSlice(types)))
	for _, L := range types {
		fmt.Fprintf(w, "  %d мм: %d шт\n", L, boardsByStock[L])
	}

	fmt.Fprintln(w, "\nПроверка выполнения требований:")
	for _, r := range requirements {
		status := "OK"
		if produced[r.Length] < r.Count {
			status = "НЕ ВЫПОЛНЕНО"
		}
		fmt.Fprintf(w, "  [%s] требуется %d × %d мм, произведено %d\n", status, r.Count, r.Length, produced[r.Length])
	}
}
