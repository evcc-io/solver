// Package integration runs PuLP's own COIN_CMDTest suite against the built
// cbc binary — the real acceptance bar for this project (see README.md).
package integration

import (
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

// TestPuLPCompatibility runs the full PuLP COIN_CMDTest suite and fails
// only on a regression — a failure not already listed in
// testdata/pulp_known_failures.txt. Opt in with RUN_PULP_TESTS=1, since it
// needs python3/network on first run to set up a venv.
func TestPuLPCompatibility(t *testing.T) {
	if os.Getenv("RUN_PULP_TESTS") == "" {
		t.Skip("set RUN_PULP_TESTS=1 to run PuLP's COIN_CMDTest suite (needs python3/network)")
	}
	if _, err := exec.LookPath("python3"); err != nil {
		t.Skip("python3 not found on PATH")
	}

	root, err := filepath.Abs("..")
	if err != nil {
		t.Fatal(err)
	}
	script := filepath.Join(root, "scripts", "run-pulp-tests.sh")

	cmd := exec.Command(script)
	cmd.Dir = root
	out, err := cmd.CombinedOutput()
	t.Log(string(out))
	if err != nil {
		t.Fatalf("run-pulp-tests.sh failed: %v", err)
	}
}
