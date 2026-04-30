
make:
	@cargo run --release

lines:
	@cloc . --exclude-dir=target
