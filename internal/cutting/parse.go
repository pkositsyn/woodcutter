package cutting

import (
	"bufio"
	"errors"
	"io"
	"regexp"
	"strconv"
	"strings"
)

var sepRE = regexp.MustCompile(`[,\t]`)

// ParseInput reads the "stock --- requirements" format. Requirements are
// aggregated by length preserving first-seen order; stock is returned raw.
func ParseInput(r io.Reader) (stock, requirements []Pair, err error) {
	sc := bufio.NewScanner(r)
	var stockLines, reqLines []string
	cur := &stockLines
	seenSep := false
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		if line == "---" {
			cur = &reqLines
			seenSep = true
			continue
		}
		*cur = append(*cur, line)
	}
	if e := sc.Err(); e != nil {
		return nil, nil, e
	}
	if !seenSep {
		return nil, nil, errors.New("не найден разделитель '---' между секцией склада и секцией требований")
	}
	for _, l := range stockLines {
		p, e := parseLine(l)
		if e != nil {
			return nil, nil, e
		}
		stock = append(stock, p)
	}
	agg := map[int]int{}
	var order []int
	for _, l := range reqLines {
		p, e := parseLine(l)
		if e != nil {
			return nil, nil, e
		}
		if _, ok := agg[p.Length]; !ok {
			order = append(order, p.Length)
		}
		agg[p.Length] += p.Count
	}
	for _, L := range order {
		requirements = append(requirements, Pair{Length: L, Count: agg[L]})
	}
	return stock, requirements, nil
}

func parseLine(s string) (Pair, error) {
	parts := sepRE.Split(s, -1)
	length, err := strconv.Atoi(strings.ReplaceAll(strings.TrimSpace(parts[0]), " ", ""))
	if err != nil {
		return Pair{}, err
	}
	count := 1
	if len(parts) > 1 {
		count, err = strconv.Atoi(strings.TrimSpace(parts[1]))
		if err != nil {
			return Pair{}, err
		}
	}
	return Pair{Length: length, Count: count}, nil
}
