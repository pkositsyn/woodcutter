// Command woodcutter reads a "stock --- requirements" spec on stdin and prints
// an optimal 1D cutting plan. Go port of cutting.py.
package main

import (
	"fmt"
	"os"

	"github.com/pkositsyn/woodcutter/internal/cutting"
)

func main() {
	stock, requirements, err := cutting.ParseInput(os.Stdin)
	if err != nil {
		fmt.Fprintln(os.Stderr, "Ошибка:", err)
		os.Exit(1)
	}
	if len(stock) == 0 {
		fmt.Fprintln(os.Stderr, "Ошибка: пустая секция склада")
		os.Exit(1)
	}

	totalStock := 0
	for _, s := range stock {
		totalStock += s.Count
	}
	totalPieces := 0
	for _, r := range requirements {
		totalPieces += r.Count
	}
	fmt.Fprintf(os.Stderr, "Типов досок на складе: %d; всего досок: %d\n", len(stock), totalStock)
	fmt.Fprintf(os.Stderr, "Различных требуемых длин: %d; всего кусков: %d\n", len(requirements), totalPieces)

	opts := cutting.Options{Padding: 5}
	plan := cutting.Solve(stock, requirements, opts)
	cutting.WriteReport(os.Stdout, plan, requirements, opts)
}
