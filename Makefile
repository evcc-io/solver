.PHONY: build test test-pulp bench-pulp vet clean

build:
	go build -o bin/cbc ./cmd/cbc

test:
	go test ./...

test-pulp: build
	./scripts/run-pulp-tests.sh

bench-pulp: build
	PULP_DURATIONS=0 ./scripts/run-pulp-tests.sh || true

vet:
	go vet ./...

clean:
	rm -rf bin .pulpenv .pulpenv-bin
