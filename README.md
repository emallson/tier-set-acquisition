# Tier Set Acquisition Simulator

Calculates stats about how long it will take to acquire 4pc tier sets for the
whole raid with different strategies and comps.

## Usage

1. Install [rust](//rustup.rs)
2. Copy `oew.toml` and change the comp + any settings you care about.
3. `cargo run --release -- <your-config.toml> samples.json`

## TODOs

- [x] Basic simulation
- [ ] Adjustable vault tier chance
- [ ] Per-class tier chance
- [ ] Per-player completion time

## License

BSD 3-clause, copyright 2022 emallson
