// Package solfile writes CBC-compatible .sol files and reads PuLP .mst
// warm-start files, matching PuLP's readsol_MPS/get_status parser.
package solfile

import (
	"bufio"
	"fmt"
	"io"
	"strconv"
	"strings"
)

// Write emits a status line then one "<idx> <name> <value> <dual>" line per
// variable, then per row. PuLP only reads positions 1-3 of each data line.
func Write(w io.Writer, statusToken string, obj float64,
	colNames []string, x, reducedCost []float64,
	rowNames []string, rowActivity, rowPrice []float64) error {
	bw := bufio.NewWriter(w)
	fmt.Fprintf(bw, "%s - objective value %.8f\n", statusToken, obj)
	for i, name := range colNames {
		if name == "" { // internal (e.g. cut) entry: no user name, skip
			continue
		}
		v := valueAt(x, i)
		rc := valueAt(reducedCost, i)
		fmt.Fprintf(bw, "%7d %-20s %15.8g %15.8g\n", i, name, v, rc)
	}
	for i, name := range rowNames {
		if name == "" { // cut row added during B&B: PuLP ignores it, skip
			continue
		}
		a := valueAt(rowActivity, i)
		pr := valueAt(rowPrice, i)
		fmt.Fprintf(bw, "%7d %-20s %15.8g %15.8g\n", i, name, a, pr)
	}
	return bw.Flush()
}

func valueAt(s []float64, i int) float64 {
	if i < len(s) {
		return s[i]
	}
	return 0
}

// ReadMIPStart parses PuLP's warm-start line shape into a name->value map.
// Unparseable lines (including the header) are skipped, not errors.
func ReadMIPStart(r io.Reader) (map[string]float64, error) {
	values := map[string]float64{}
	sc := bufio.NewScanner(r)
	for sc.Scan() {
		line := sc.Text()
		if len(line) <= 2 {
			break
		}
		fields := strings.Fields(line)
		if len(fields) > 0 && fields[0] == "**" {
			fields = fields[1:]
		}
		if len(fields) < 3 {
			continue
		}
		if val, err := strconv.ParseFloat(fields[2], 64); err == nil {
			values[fields[1]] = val
		}
	}
	return values, sc.Err()
}
