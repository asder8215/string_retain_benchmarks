#![allow(internal_features)]
#![feature(char_internals)]
#![feature(const_trait_impl)]
use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};

pub struct StringTest {
    vec: Vec<u8>,
}

impl Clone for StringTest {
    fn clone(&self) -> Self {
        StringTest {
            vec: self.vec.clone(),
        }
    }

    /// Clones the contents of `source` into `self`.
    ///
    /// This method is preferred over simply assigning `source.clone()` to `self`,
    /// as it avoids reallocation if possible.
    fn clone_from(&mut self, source: &Self) {
        self.vec.clone_from(&source.vec);
    }
}

impl StringTest {
    #[inline]
    pub const fn len(&self) -> usize {
        self.vec.len()
    }

    #[inline]
    pub const fn new() -> StringTest {
        StringTest { vec: Vec::new() }
    }

    #[inline]
    pub fn push(&mut self, ch: char) {
        let len = self.len();
        let ch_len = ch.len_utf8();
        self.vec.reserve(ch_len);

        // SAFETY: Just reserved capacity for at least the length needed to encode `ch`.
        unsafe {
            core::char::encode_utf8_raw_unchecked(ch as u32, self.vec.as_mut_ptr().add(len));
            self.vec.set_len(len + ch_len);
        }
    }

    #[inline]
    pub fn push_str(&mut self, string: &str) {
        self.vec.extend_from_slice(string.as_bytes())
    }

    #[inline]
    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(char) -> bool,
    {
        struct SetLenOnDrop<'a> {
            s: &'a mut StringTest,
            idx: usize,
            del_bytes: usize,
        }

        impl<'a> Drop for SetLenOnDrop<'a> {
            fn drop(&mut self) {
                let new_len = self.idx - self.del_bytes;
                debug_assert!(new_len <= self.s.vec.len());
                // SAFETY: Restore the string length to the number of bytes written so far.
                unsafe { self.s.vec.set_len(new_len) };
            }
        }

        let len = self.vec.len();
        let mut guard = SetLenOnDrop {
            s: self,
            idx: 0,
            del_bytes: 0,
        };

        while guard.idx < len {
            // SAFETY: Use `vec`: `String`'s deref includes `write..read`,
            // which may be invalid UTF-8. The vector's length is still `len`;
            // `read < len`, so `read..len` is in bounds and non-empty.
            // At least one character was removed, so `write < read`.
            // Copies end before the next `read`, preserving unread bytes.
            // `read` starts after a `char_indices` character and advances by whole
            // characters, so it stays on a char boundary. The original string was valid UTF-8,
            // so the unchanged, non-empty tail is valid UTF-8 and `chars().next()` returns `Some`.
            let ch = unsafe {
                let tail = str::from_utf8_unchecked(guard.s.vec.get_unchecked(guard.idx..len));
                tail.chars().next().unwrap_unchecked()
            };
            let ch_len = ch.len_utf8();

            if !f(ch) {
                guard.del_bytes += ch_len;
            } else if guard.del_bytes > 0 {
                // SAFETY: `guard.idx` is in bound and `guard.del_bytes` represent the number of
                // bytes that are erased from the string so the resulting `guard.idx -
                // guard.del_bytes` always represent a valid unicode code point.
                //
                // `guard.del_bytes` >= `ch.len_utf8()`, so taking a slice with `ch.len_utf8()` len
                // is safe.
                ch.encode_utf8(unsafe {
                    core::slice::from_raw_parts_mut(
                        guard.s.vec.as_mut_ptr().add(guard.idx - guard.del_bytes),
                        ch.len_utf8(),
                    )
                });
            }

            // Point idx to the next char
            guard.idx += ch_len;
        }

        drop(guard);
    }
}

fn previous_string_retain() {
    // 1 KiB of ASCII text with a control character every 16 bytes: 64 removals per call.
    let mut dirty = StringTest::new();
    while dirty.len() < 1024 {
        dirty.push_str("fifteen bytes..");
        dirty.push('\u{1}');
    }
    // Same length, nothing to remove.
    let clean = StringTest {
        vec: "sixteen bytes...".as_bytes().repeat(64),
    };

    for (_, input) in [("removes 1 of 16", &dirty), ("removes nothing", &clean)] {
        for _ in 0..15 {
            let mut s = black_box(input).clone();
            s.retain(|c| !c.is_control());
            black_box(s);
        }
    }
}

fn bench_string_retain(c: &mut Criterion) {
    c.bench_function("Previous String Retain Benchmarks", |b| {
        b.iter(|| black_box(previous_string_retain()))
    });
}

criterion_group!(benches, bench_string_retain);
criterion_main!(benches);
