#!/bin/bash

set -e

for s in settings/*.toml; do
        sname=`echo "$s" | sed -e 's/settings\/\(.*\)\.toml/\1/'`
        cargo run --release -- $s analysis/data/$sname.json
done

for s in settings/*.toml; do
        sname=`echo "$s" | sed -e 's/settings\/\(.*\)\.toml/\1/'`
        cargo run --release --features old_tokens -- $s analysis/data/old-tokens/$sname.json
done
