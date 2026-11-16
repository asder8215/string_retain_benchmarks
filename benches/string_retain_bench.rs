#![allow(internal_features)]
#![allow(unexpected_cfgs)]
#![feature(char_internals)]
#![feature(likely_unlikely)]
#![feature(iter_advance_by)]
#![feature(core_intrinsics)]
#![feature(str_internals)]
#![feature(const_trait_impl)]
use core::{fmt, slice, str::utf8_char_width};
use std::{hint::{self, black_box}, intrinsics::unlikely, iter::FusedIterator, mem, num::NonZero};

use criterion::{Criterion, criterion_group, criterion_main};

const USIZE_SIZE: usize = size_of::<usize>();
const UNROLL_INNER: usize = 4;

#[inline]
pub fn count_chars(s: &str) -> usize {
    if cfg!(feature = "optimize_for_size") || s.len() < USIZE_SIZE * UNROLL_INNER {
        // Avoid entering the optimized implementation for strings where the
        // difference is not likely to matter, or where it might even be slower.
        // That said, a ton of thought was not spent on the particular threshold
        // here, beyond "this value seems to make sense".
        char_count_general_case(s.as_bytes())
    } else {
        do_count_chars(s)
    }
}

pub const trait Repeat16 {
    #[inline]
    fn repeat_u8(x: u8) -> usize {
        usize::from_ne_bytes([x; size_of::<usize>()])
    }

    #[inline]
    fn repeat_u16(x: u16) -> usize {
        let mut r = 0usize;
        let mut i = 0;
        while i < size_of::<usize>() {
            // Use `wrapping_shl` to make it work on targets with 16-bit `usize`
            r = r.wrapping_shl(16) | (x as usize);
            i += 2;
        }
        r
    }
}

const impl Repeat16 for usize {}

#[inline]
fn sum_bytes_in_usize(values: usize) -> usize {
    const LSB_SHORTS: usize = usize::repeat_u16(0x0001);
    const SKIP_BYTES: usize = usize::repeat_u16(0x00ff);

    let pair_sum: usize = (values & SKIP_BYTES) + ((values >> 8) & SKIP_BYTES);
    pair_sum.wrapping_mul(LSB_SHORTS) >> ((USIZE_SIZE - 2) * 8)
}

fn char_count_general_case(s: &[u8]) -> usize {
    s.iter().filter(|&&byte| !utf8_is_cont_byte(byte)).count()
}

#[inline]
fn contains_non_continuation_byte(w: usize) -> usize {
    const LSB: usize = usize::repeat_u8(0x01);
    ((!w >> 7) | (w >> 6)) & LSB
}

fn do_count_chars(s: &str) -> usize {
    // For correctness, `CHUNK_SIZE` must be:
    //
    // - Less than or equal to 255, otherwise we'll overflow bytes in `counts`.
    // - A multiple of `UNROLL_INNER`, otherwise our `break` inside the
    //   `body.chunks(CHUNK_SIZE)` loop is incorrect.
    //
    // For performance, `CHUNK_SIZE` should be:
    // - Relatively cheap to `/` against (so some simple sum of powers of two).
    // - Large enough to avoid paying for the cost of the `sum_bytes_in_usize`
    //   too often.
    const CHUNK_SIZE: usize = 192;

    // Check the properties of `CHUNK_SIZE` and `UNROLL_INNER` that are required
    // for correctness.
    const _: () = assert!(CHUNK_SIZE < 256);
    const _: () = assert!(CHUNK_SIZE.is_multiple_of(UNROLL_INNER));

    // SAFETY: transmuting `[u8]` to `[usize]` is safe except for size
    // differences which are handled by `align_to`.
    let (head, body, tail) = unsafe { s.as_bytes().align_to::<usize>() };

    // This should be quite rare, and basically exists to handle the degenerate
    // cases where align_to fails (as well as miri under symbolic alignment
    // mode).
    //
    // The `unlikely` helps discourage LLVM from inlining the body, which is
    // nice, as we would rather not mark the `char_count_general_case` function
    // as cold.
    if unlikely(body.is_empty() || head.len() > USIZE_SIZE || tail.len() > USIZE_SIZE) {
        return char_count_general_case(s.as_bytes());
    }

    let mut total = char_count_general_case(head) + char_count_general_case(tail);
    // Split `body` into `CHUNK_SIZE` chunks to reduce the frequency with which
    // we call `sum_bytes_in_usize`.
    for chunk in body.chunks(CHUNK_SIZE) {
        // We accumulate intermediate sums in `counts`, where each byte contains
        // a subset of the sum of this chunk, like a `[u8; size_of::<usize>()]`.
        let mut counts = 0;

        let (unrolled_chunks, remainder) = chunk.as_chunks::<UNROLL_INNER>();
        for unrolled in unrolled_chunks {
            for &word in unrolled {
                // Because `CHUNK_SIZE` is < 256, this addition can't cause the
                // count in any of the bytes to overflow into a subsequent byte.
                counts += contains_non_continuation_byte(word);
            }
        }

        // Sum the values in `counts` (which, again, is conceptually a `[u8;
        // size_of::<usize>()]`), and accumulate the result into `total`.
        total += sum_bytes_in_usize(counts);

        // If there's any data in `remainder`, then handle it. This will only
        // happen for the last `chunk` in `body.chunks()` (because `CHUNK_SIZE`
        // is divisible by `UNROLL_INNER`), so we explicitly break at the end
        // (which seems to help LLVM out).
        if !remainder.is_empty() {
            // Accumulate all the data in the remainder.
            let mut counts = 0;
            for &word in remainder {
                counts += contains_non_continuation_byte(word);
            }
            total += sum_bytes_in_usize(counts);
            break;
        }
    }
    total
}

/// Mask of the value bits of a continuation byte.
const CONT_MASK: u8 = 0b0011_1111;

#[inline]
const fn utf8_first_byte(byte: u8, width: u32) -> u32 {
    (byte & (0x7F >> width)) as u32
}

/// Returns the value of `ch` updated with continuation byte `byte`.
#[inline]
const fn utf8_acc_cont_byte(ch: u32, byte: u8) -> u32 {
    (ch << 6) | (byte & CONT_MASK) as u32
}

/// Checks whether the byte is a UTF-8 continuation byte (i.e., starts with the
/// bits `10`).
#[inline]
pub const fn utf8_is_cont_byte(byte: u8) -> bool {
    (byte as i8) < -64
}

#[inline]
pub unsafe fn next_code_point<'a, I: Iterator<Item = &'a u8>>(bytes: &mut I) -> Option<u32> {
    // Decode UTF-8
    let x = *bytes.next()?;
    if x < 128 {
        return Some(x as u32);
    }

    // Multibyte case follows
    // Decode from a byte combination out of: [[[x y] z] w]
    // NOTE: Performance is sensitive to the exact formulation here
    let init = utf8_first_byte(x, 2);
    // SAFETY: `bytes` produces an UTF-8-like string,
    // so the iterator must produce a value here.
    let y = unsafe { *bytes.next().unwrap_unchecked() };
    let mut ch = utf8_acc_cont_byte(init, y);
    if x >= 0xE0 {
        // [[x y z] w] case
        // 5th bit in 0xE0 .. 0xEF is always clear, so `init` is still valid
        // SAFETY: `bytes` produces an UTF-8-like string,
        // so the iterator must produce a value here.
        let z = unsafe { *bytes.next().unwrap_unchecked() };
        let y_z = utf8_acc_cont_byte((y & CONT_MASK) as u32, z);
        ch = init << 12 | y_z;
        if x >= 0xF0 {
            // [x y z w] case
            // use only the lower 3 bits of `init`
            // SAFETY: `bytes` produces an UTF-8-like string,
            // so the iterator must produce a value here.
            let w = unsafe { *bytes.next().unwrap_unchecked() };
            ch = (init & 7) << 18 | utf8_acc_cont_byte(y_z, w);
        }
    }

    Some(ch)
}

#[inline]
pub unsafe fn next_code_point_reverse<'a, I>(bytes: &mut I) -> Option<u32>
where
    I: DoubleEndedIterator<Item = &'a u8>,
{
    // Decode UTF-8
    let w = match *bytes.next_back()? {
        next_byte if next_byte < 128 => return Some(next_byte as u32),
        back_byte => back_byte,
    };

    // Multibyte case follows
    // Decode from a byte combination out of: [x [y [z w]]]
    let mut ch;
    // SAFETY: `bytes` produces an UTF-8-like string,
    // so the iterator must produce a value here.
    let z = unsafe { *bytes.next_back().unwrap_unchecked() };
    ch = utf8_first_byte(z, 2);
    if utf8_is_cont_byte(z) {
        // SAFETY: `bytes` produces an UTF-8-like string,
        // so the iterator must produce a value here.
        let y = unsafe { *bytes.next_back().unwrap_unchecked() };
        ch = utf8_first_byte(y, 3);
        if utf8_is_cont_byte(y) {
            // SAFETY: `bytes` produces an UTF-8-like string,
            // so the iterator must produce a value here.
            let x = unsafe { *bytes.next_back().unwrap_unchecked() };
            ch = utf8_first_byte(x, 4);
            ch = utf8_acc_cont_byte(ch, y);
        }
        ch = utf8_acc_cont_byte(ch, z);
    }
    ch = utf8_acc_cont_byte(ch, w);

    Some(ch)
}

#[derive(Clone)]
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct Chars<'a> {
    pub iter: slice::Iter<'a, u8>,
}

impl<'a> Iterator for Chars<'a> {
    type Item = char;

    #[inline]
    fn next(&mut self) -> Option<char> {
        // SAFETY: `str` invariant says `self.iter` is a valid UTF-8 string and
        // the resulting `ch` is a valid Unicode Scalar Value.
        unsafe { next_code_point(&mut self.iter).map(|ch| char::from_u32_unchecked(ch)) }
    }

    #[inline]
    fn count(self) -> usize {
        count_chars(self.as_str())
    }

    #[inline]
    fn advance_by(&mut self, mut remainder: usize) -> Result<(), NonZero<usize>> {
        const CHUNK_SIZE: usize = 32;

        if remainder >= CHUNK_SIZE {
            let mut chunks = self.iter.as_slice().as_chunks::<CHUNK_SIZE>().0.iter();
            let mut bytes_skipped: usize = 0;

            while remainder > CHUNK_SIZE
                && let Some(chunk) = chunks.next()
            {
                bytes_skipped += CHUNK_SIZE;

                let mut start_bytes = [false; CHUNK_SIZE];

                for i in 0..CHUNK_SIZE {
                    start_bytes[i] = !utf8_is_cont_byte(chunk[i]);
                }

                remainder -= start_bytes.into_iter().map(|i| i as u8).sum::<u8>() as usize;
            }

            // SAFETY: The amount of bytes exists since we just iterated over them,
            // so advance_by will succeed.
            unsafe { self.iter.advance_by(bytes_skipped).unwrap_unchecked() };

            // skip trailing continuation bytes
            while self.iter.len() > 0 {
                let b = self.iter.as_slice()[0];
                if !utf8_is_cont_byte(b) {
                    break;
                }
                // SAFETY: We just peeked at the byte, therefore it exists
                unsafe { self.iter.advance_by(1).unwrap_unchecked() };
            }
        }

        while (remainder > 0) && (self.iter.len() > 0) {
            remainder -= 1;
            let b = self.iter.as_slice()[0];
            let slurp = utf8_char_width(b);
            // SAFETY: utf8 validity requires that the string must contain
            // the continuation bytes (if any)
            unsafe { self.iter.advance_by(slurp).unwrap_unchecked() };
        }

        NonZero::new(remainder).map_or(Ok(()), Err)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.iter.len();
        (len.div_ceil(4), Some(len))
    }

    #[inline]
    fn last(mut self) -> Option<char> {
        // No need to go through the entire string.
        self.next_back()
    }
}

impl fmt::Debug for Chars<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Chars(")?;
        f.debug_list().entries(self.clone()).finish()?;
        write!(f, ")")?;
        Ok(())
    }
}

impl<'a> DoubleEndedIterator for Chars<'a> {
    #[inline]
    fn next_back(&mut self) -> Option<char> {
        // SAFETY: `str` invariant says `self.iter` is a valid UTF-8 string and
        // the resulting `ch` is a valid Unicode Scalar Value.
        unsafe { next_code_point_reverse(&mut self.iter).map(|ch| char::from_u32_unchecked(ch)) }
    }
}

impl FusedIterator for Chars<'_> {}

impl<'a> Chars<'a> {
    /// Views the underlying data as a subslice of the original data.
    ///
    /// This has the same lifetime as the original slice, and so the
    /// iterator can continue to be used while this exists.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut chars = "abc".chars();
    ///
    /// assert_eq!(chars.as_str(), "abc");
    /// chars.next();
    /// assert_eq!(chars.as_str(), "bc");
    /// chars.next();
    /// chars.next();
    /// assert_eq!(chars.as_str(), "");
    /// ```
    #[must_use]
    #[inline]
    pub fn as_str(&self) -> &'a str {
        // SAFETY: `Chars` is only made from a str, which guarantees the iter is valid UTF-8.
        unsafe { from_utf8_unchecked(self.iter.as_slice()) }
    }
}

#[inline]
#[must_use]
pub const unsafe fn from_utf8_unchecked(v: &[u8]) -> &str {
    // SAFETY: the caller must guarantee that the bytes `v` are valid UTF-8.
    // Also relies on `&str` and `&[u8]` having the same layout.
    unsafe { mem::transmute(v) }
}

#[derive(Clone, Debug)]
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct CharIndices<'a> {
    pub front_offset: usize,
    pub iter: Chars<'a>,
}

impl<'a> Iterator for CharIndices<'a> {
    type Item = (usize, char);

    #[inline]
    fn next(&mut self) -> Option<(usize, char)> {
        let pre_len = self.iter.iter.len();
        match self.iter.next() {
            None => None,
            Some(ch) => {
                let index = self.front_offset;
                let len = self.iter.iter.len();
                self.front_offset += pre_len - len;
                Some((index, ch))
            }
        }
    }

    #[inline]
    fn count(self) -> usize {
        self.iter.count()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }

    #[inline]
    fn last(mut self) -> Option<(usize, char)> {
        // No need to go through the entire string.
        self.next_back()
    }
}

impl<'a> DoubleEndedIterator for CharIndices<'a> {
    #[inline]
    fn next_back(&mut self) -> Option<(usize, char)> {
        self.iter.next_back().map(|ch| {
            let index = self.front_offset + self.iter.iter.len();
            (index, ch)
        })
    }
}

impl FusedIterator for CharIndices<'_> {}

impl<'a> CharIndices<'a> {
    /// Views the underlying data as a subslice of the original data.
    ///
    /// This has the same lifetime as the original slice, and so the
    /// iterator can continue to be used while this exists.
    #[must_use]
    #[inline]
    pub fn as_str(&self) -> &'a str {
        self.iter.as_str()
    }

    /// Returns the byte position of the next character, or the length
    /// of the underlying string if there are no more characters.
    ///
    /// This means that, when the iterator has not been fully consumed,
    /// the returned value will match the index that will be returned
    /// by the next call to [`next()`](Self::next).
    ///
    /// # Examples
    ///
    /// ```
    /// let mut chars = "a楽".char_indices();
    ///
    /// // `next()` has not been called yet, so `offset()` returns the byte
    /// // index of the first character of the string, which is always 0.
    /// assert_eq!(chars.offset(), 0);
    /// // As expected, the first call to `next()` also returns 0 as index.
    /// assert_eq!(chars.next(), Some((0, 'a')));
    ///
    /// // `next()` has been called once, so `offset()` returns the byte index
    /// // of the second character ...
    /// assert_eq!(chars.offset(), 1);
    /// // ... which matches the index returned by the next call to `next()`.
    /// assert_eq!(chars.next(), Some((1, '楽')));
    ///
    /// // Once the iterator has been consumed, `offset()` returns the length
    /// // in bytes of the string.
    /// assert_eq!(chars.offset(), 4);
    /// assert_eq!(chars.next(), None);
    /// ```
    #[inline]
    #[must_use]
    pub fn offset(&self) -> usize {
        self.front_offset
    }
}

pub struct StringTest {
    vec: Vec<u8>
}

impl Clone for StringTest {
    fn clone(&self) -> Self {
        StringTest { vec: self.vec.clone() }
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
    pub fn char_indices(&self) -> CharIndices<'_> {
        CharIndices { front_offset: 0, iter: Chars { iter: self.vec.iter() } }
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
        let len = self.len();
        if len == 0 {
            // Explicit check results in better optimization
            return;
        }

        struct PanicGuard<'a> {
            s: &'a mut StringTest,
            write: usize,
        }

        impl Drop for PanicGuard<'_> {
            fn drop(&mut self) {
                debug_assert!(self.write <= self.s.len());
                debug_assert!(str::from_utf8(&self.s.vec[..self.write]).is_ok());
                // SAFETY: Restore the string length to the number of bytes written so far.
                unsafe { self.s.vec.set_len(self.write) }
            }
        }

        // Fast path: find the first character that should be removed or return early.
        let mut chars = self.char_indices();
        let (mut read, write) = loop {
            let Some((idx, ch)) = chars.next() else { return };
            if hint::unlikely(!f(ch)) {
                break (idx + ch.len_utf8(), idx);
            }
        };
        drop(chars);

        // Slow path: at least one character is going to be removed.
        let mut g = PanicGuard { s: self, write };
        while read < len {
            // SAFETY: Use `vec`: `String`'s deref includes `write..read`,
            // which may be invalid UTF-8. The vector's length is still `len`;
            // `read < len`, so `read..len` is in bounds and non-empty.
            // At least one character was removed, so `write < read`.
            // Copies end before the next `read`, preserving unread bytes.
            // `read` starts after a `char_indices` character and advances by whole
            // characters, so it stays on a char boundary. The original string was valid UTF-8,
            // so the unchanged, non-empty tail is valid UTF-8 and `chars().next()` returns `Some`.
            let ch = unsafe {
                let tail = str::from_utf8_unchecked(g.s.vec.get_unchecked(read..len));
                tail.chars().next().unwrap_unchecked()
            };
            let ch_len = ch.len_utf8();
            if f(ch) {
                // SAFETY: `g.write` is always behind `g.read`, so we should have space for at least 
                // the current `ch_len` bytes, so copying `ch` from write..read should be safe. 
                // Moreover, we do not have to worry about overlap safety with `char::encode_utf8`
                // (unlike ptr::copy_nonoverlapping`) because we have a copy of the char that
                // we populate within write..write + ch_len one byte at a time.
                unsafe {
                    core::char::encode_utf8_raw_unchecked(ch as u32, g.s.vec.as_mut_ptr().add(g.write));
                };
                g.write += ch_len;
            }
            read += ch_len;
        }

        // All bytes processed; commit the final length by dropping the guard.
        drop(g);
    }
}

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

fn previous_string_retain() {
    // 1 KiB of ASCII text with a control character every 16 bytes: 64 removals per call.
    let mut dirty = StringTest::new();
    while dirty.len() < 1024 {
        dirty.push_str("fifteen bytes..");
        dirty.push('\u{1}');
    }
    // Same length, nothing to remove.
    let clean = StringTest { vec: "sixteen bytes...".as_bytes().repeat(64) };

    for (_, input) in [("removes 1 of 16", &dirty), ("removes nothing", &clean)] {
        for _ in 0..15 {
            let mut s = black_box(input).clone();
            s.retain(|c| !c.is_control());
            black_box(s);
        }
    }
}

fn bench_string_retain(c: &mut Criterion) {
    c.bench_function("Current String Retain Benchmarks", |b| {
        b.iter(|| {
            black_box(current_string_retain())
        })
    });

    c.bench_function("Previous String Retain Benchmarks", |b| {
        b.iter(|| {
            black_box(previous_string_retain())
        })
    });

}

criterion_group!(benches, bench_string_retain);
criterion_main!(benches);
