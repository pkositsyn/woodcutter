package cutting

import (
	"reflect"
	"strings"
	"testing"
)

func TestParseInputBasic(t *testing.T) {
	in := "6000, 10\n3000\n---\n2000, 3\n2000, 2\n1500\n"
	stock, reqs, err := ParseInput(strings.NewReader(in))
	if err != nil {
		t.Fatal(err)
	}
	wantStock := []Pair{{6000, 10}, {3000, 1}}
	if !reflect.DeepEqual(stock, wantStock) {
		t.Fatalf("stock = %v, want %v", stock, wantStock)
	}
	// 2000 aggregated to 5, first-seen order: 2000 then 1500.
	wantReq := []Pair{{2000, 5}, {1500, 1}}
	if !reflect.DeepEqual(reqs, wantReq) {
		t.Fatalf("reqs = %v, want %v", reqs, wantReq)
	}
}

func TestParseInputNoSeparator(t *testing.T) {
	_, _, err := ParseInput(strings.NewReader("6000, 10\n3000\n"))
	if err == nil {
		t.Fatal("expected error for missing '---'")
	}
}

func TestParseLineTab(t *testing.T) {
	p, err := parseLine("1 500\t4")
	if err != nil {
		t.Fatal(err)
	}
	if p != (Pair{1500, 4}) {
		t.Fatalf("got %v, want {1500 4}", p)
	}
}
