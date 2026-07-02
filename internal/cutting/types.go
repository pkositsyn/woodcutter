package cutting

import "time"

// Pair is a (length, count) tuple used for both stock and requirements.
type Pair struct {
	Length int
	Count  int
}

// PieceKind classifies a segment of a cut board.
type PieceKind int

const (
	PieceCut PieceKind = iota
	PiecePadding
	PieceWaste
)

// Piece is one segment on a board: a produced cut, a saw-kerf padding, or waste.
type Piece struct {
	Length int
	Start  int
	End    int
	Kind   PieceKind
}

// Board is a single physical board with its cutting layout.
type Board struct {
	StockLength int
	Pieces      []Piece
}

// Plan is the full cutting result.
type Plan struct {
	TotalMaterial int // sum of used board lengths; -1 when infeasible
	Feasible      bool
	Boards        []Board
	Dropped       []Pair // requirements longer than the longest stock board
}

// Options configures the solver.
type Options struct {
	Padding int

	// MILPTimeout is the max wall-clock time allowed for the integer solve;
	// 0 means use the default.
	MILPTimeout time.Duration
}
