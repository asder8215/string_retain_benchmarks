use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;

fn current_string_retain() {
    // 1 KiB of ASCII text with a control character every 16 bytes: 64 removals per call.
    let mut dirty = String::new();
    while dirty.len() < 1024 {
        dirty.push_str("fifteen bytes..");
        dirty.push('\u{1}');
    }
    // Same length, nothing to remove.
    let clean = "sixteen bytes...".repeat(64);

    for (_, input) in [("removes 1 of 16", &dirty), ("removes nothing", &clean)] {
        for _ in 0..15 {
            // for _ in 0..100_000 {
            let mut s = black_box(input).clone();
            s.retain(|c| !c.is_control());
            black_box(s);
            // }
        }
    }
}

fn bench_string_retain(c: &mut Criterion) {
    c.bench_function("Current String Retain Benchmarks", |b| {
        b.iter(|| black_box(current_string_retain()))
    });
}

criterion_group!(benches, bench_string_retain);
criterion_main!(benches);
