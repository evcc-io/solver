package solfile

import (
	"bytes"
	"strings"
	"testing"
)

// pulpGetStatus mirrors PuLP's get_status exactly (coin_api.py) to verify
// our Write output classifies the way PuLP's own parser would.
func pulpGetStatus(t *testing.T, sol string) string {
	t.Helper()
	first := strings.SplitN(sol, "\n", 2)[0]
	tokens := strings.Fields(first)
	switch tokens[0] {
	case "Optimal", "Infeasible", "Unbounded":
		return tokens[0]
	case "Stopped":
		if len(tokens) >= 5 && tokens[4] == "objective" {
			return "Optimal"
		}
		return "Stopped"
	}
	return "Undefined"
}

func TestWriteOptimalParsesAsOptimal(t *testing.T) {
	var buf bytes.Buffer
	err := Write(&buf, "Optimal", 41, []string{"x0", "x1"}, []float64{1, 0}, []float64{0, -2},
		[]string{"cap"}, []float64{11}, []float64{0.5})
	if err != nil {
		t.Fatalf("Write: %v", err)
	}
	if got := pulpGetStatus(t, buf.String()); got != "Optimal" {
		t.Fatalf("status = %q, want Optimal", got)
	}
}

func TestWriteStoppedWithIncumbentParsesAsOptimal(t *testing.T) {
	var buf bytes.Buffer
	Write(&buf, "Stopped on time", 41, []string{"x0"}, []float64{1}, []float64{0}, nil, nil, nil)
	if got := pulpGetStatus(t, buf.String()); got != "Optimal" {
		t.Fatalf("status = %q, want Optimal (matches real CBC's own quirk)", got)
	}
}

func TestReadWriteRoundTrip(t *testing.T) {
	var buf bytes.Buffer
	Write(&buf, "Optimal", 41, []string{"x0", "x1"}, []float64{1, 3.5}, []float64{0, 0}, nil, nil, nil)

	values, err := ReadMIPStart(&buf)
	if err != nil {
		t.Fatalf("ReadMIPStart: %v", err)
	}
	if values["x0"] != 1 || values["x1"] != 3.5 {
		t.Fatalf("values = %+v, want x0=1 x1=3.5", values)
	}
}

func TestReadPuLPOwnMSTFormat(t *testing.T) {
	// Exact shape of PuLP's own writesol(): a placeholder header line, then
	// "{:>7} {} {:>15} {:>23}\n".format(i, name, value, 0) per variable.
	mst := "Stopped on time - objective value 0\n" +
		"      0 x0             1.0                       0\n" +
		"      1 x1             0.0                       0\n"
	values, err := ReadMIPStart(strings.NewReader(mst))
	if err != nil {
		t.Fatalf("ReadMIPStart: %v", err)
	}
	if values["x0"] != 1.0 || values["x1"] != 0.0 {
		t.Fatalf("values = %+v, want x0=1 x1=0", values)
	}
}
