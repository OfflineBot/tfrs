# `make`                       — build release binary, copy to bin/tfrs
# `make train`                  — fresh training run (writes bin/mail_model.bin)
# `make eval`                   — evaluate the checkpoint
# `make release VERSION=v0.1.0` — tag, push, create GitHub release with
#                                  Linux binary + trained model. The CI
#                                  workflow will then attach macOS and
#                                  Windows binaries to the same release.
# `make lines`                  — line counts

BIN := bin/tfrs
MODEL := bin/mail_model.bin

.PHONY: make train eval release lines

make: $(BIN)

$(BIN): $(shell find src -name '*.rs') Cargo.toml
	@cargo build --release
	@mkdir -p bin
	@cp -f target/release/tfrs $(BIN)
	@echo "built $(BIN)"

train: $(BIN)
	@./$(BIN) train --max-steps 20000 --target-label-acc 0.95 --eval-every 500 --model $(MODEL)

eval: $(BIN)
	@./$(BIN) eval --model $(MODEL)

# ----- release -----------------------------------------------------------
# Usage: make release VERSION=v0.1.0 [NOTES="release notes here"]
release:
	@if [ -z "$(VERSION)" ]; then \
		echo "error: VERSION not set. Usage: make release VERSION=v0.1.0"; exit 2; \
	fi
	@case "$(VERSION)" in v*) ;; *) echo "error: VERSION must start with 'v' (got '$(VERSION)')"; exit 2 ;; esac
	@command -v gh >/dev/null || { echo "error: 'gh' (GitHub CLI) not installed"; exit 2; }
	@gh auth status >/dev/null 2>&1 || { echo "error: 'gh' not authenticated — run 'gh auth login'"; exit 2; }
	@if ! git diff --quiet || ! git diff --cached --quiet; then \
		echo "error: working tree is dirty — commit or stash first"; exit 2; \
	fi
	@if [ ! -f "$(MODEL)" ]; then \
		echo "error: $(MODEL) not found — run 'make train' first"; exit 2; \
	fi
	@$(MAKE) --no-print-directory $(BIN)
	@echo "→ tagging $(VERSION)"
	@git tag -a "$(VERSION)" -m "$(VERSION)"
	@git push origin "$(VERSION)"
	@mkdir -p dist
	@cp -f $(BIN)   dist/tfrs-linux-x86_64
	@cp -f $(MODEL) dist/mail_model.bin
	@echo "→ creating GitHub release $(VERSION)"
	@gh release create "$(VERSION)" \
		--title "$(VERSION)" \
		--notes "$(if $(NOTES),$(NOTES),Automated release from \`make release\`. Binary + trained model attached.)" \
		dist/tfrs-linux-x86_64 dist/mail_model.bin
	@echo "→ done. CI will attach macOS / Windows binaries when its build finishes."

lines:
	@cloc . --exclude-dir=target,bin,dist
