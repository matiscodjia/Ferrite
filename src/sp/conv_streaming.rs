use crate::Scalar;

/// Row-at-a-time 2D convolution: RAM is `O(KH * W)` (one ring buffer of `KH`
/// rows), not `O(H * W)` — the frame itself is never held in memory. Feed it
/// one sensor row at a time via [`push_row`](Self::push_row); once
/// [`ready_to_compute`](Self::ready_to_compute) is true, [`conv2d`](Self::conv2d)
/// produces the output row for the oldest window currently buffered.
pub struct ConvStreaming<const W: usize, const KH: usize, const KW: usize> {
    rows: [[Scalar; W]; KH],
    next_row_to_write: usize,
    rows_filled: usize,
}

impl<const W: usize, const KH: usize, const KW: usize> ConvStreaming<W, KH, KW> {
    pub fn new() -> Self {
        ConvStreaming {
            rows: [[0.0; W]; KH],
            next_row_to_write: 0,
            rows_filled: 0,
        }
    }
    pub fn push_row(self: &mut Self, row: [Scalar; W]) {
        self.rows[self.next_row_to_write] = row;
        self.next_row_to_write = (self.next_row_to_write + 1) % KH;
        self.rows_filled = (self.rows_filled + 1).min(KH);
    }
    pub fn ready_to_compute(self: &Self) -> bool {
        self.rows_filled == KH
    }
    pub fn conv2d<const W_OUT: usize>(self: &Self, kernel: &[[Scalar; KW]; KH]) -> [Scalar; W_OUT] {
        let mut res = [0.0; W_OUT];
        for l in 0..W_OUT {
            for j in 0..KH {
                for k in 0..KW {
                    let physical = (self.next_row_to_write + j) % KH;
                    res[l] += self.rows[physical][l + k] * kernel[j][k];
                }
            }
        }
        return res;
    }
}

impl<const W: usize, const KH: usize, const KW: usize> Default for ConvStreaming<W, KH, KW> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_row_fills_buffer_and_caps_at_kh() {
        let mut cs = ConvStreaming::<4, 2, 2>::new();
        assert_eq!(cs.rows_filled, 0);
        cs.push_row([1.0, 2.0, 3.0, 4.0]);
        assert_eq!(cs.rows_filled, 1);
        cs.push_row([5.0, 6.0, 7.0, 8.0]);
        assert_eq!(cs.rows_filled, 2);
        // KH = 2: a third push wraps the ring buffer, rows_filled stays capped.
        cs.push_row([9.0, 10.0, 11.0, 12.0]);
        assert_eq!(cs.rows_filled, 2);
    }

    #[test]
    fn ready_to_compute_true_once_kh_rows_pushed() {
        let mut cs = ConvStreaming::<4, 2, 2>::new();
        assert!(!cs.ready_to_compute(), "empty buffer should not be ready");
        cs.push_row([1.0, 2.0, 3.0, 4.0]);
        assert!(
            !cs.ready_to_compute(),
            "only 1 of 2 rows pushed, should not be ready yet"
        );
        cs.push_row([5.0, 6.0, 7.0, 8.0]);
        assert!(
            cs.ready_to_compute(),
            "both rows pushed (rows_filled == KH), should be ready"
        );
    }

    #[test]
    fn conv2d_identity_kernel_picks_diagonal_elements() {
        let mut cs = ConvStreaming::<4, 2, 2>::new();
        cs.push_row([1.0, 2.0, 3.0, 4.0]);
        cs.push_row([5.0, 6.0, 7.0, 8.0]);
        let kernel = [[1.0, 0.0], [0.0, 1.0]];
        // res[l] = row0[l] * 1 + row0[l+1] * 0 + row1[l] * 0 + row1[l+1] * 1
        //        = row0[l] + row1[l+1]
        let res: [Scalar; 3] = cs.conv2d(&kernel);
        assert_eq!(res, [1.0 + 6.0, 2.0 + 7.0, 3.0 + 8.0]);
    }

    #[test]
    fn conv2d_box_kernel_sums_2x2_window() {
        let mut cs = ConvStreaming::<4, 2, 2>::new();
        cs.push_row([1.0, 2.0, 3.0, 4.0]);
        cs.push_row([5.0, 6.0, 7.0, 8.0]);
        let kernel = [[1.0, 1.0], [1.0, 1.0]];
        let res: [Scalar; 3] = cs.conv2d(&kernel);
        assert_eq!(
            res,
            [
                1.0 + 2.0 + 5.0 + 6.0,
                2.0 + 3.0 + 6.0 + 7.0,
                3.0 + 4.0 + 7.0 + 8.0
            ]
        );
    }

    #[test]
    fn conv2d_after_wraparound_uses_two_most_recent_rows() {
        // KH = 2: push 3 rows, the ring buffer should hold rows 1 and 2 only
        // (row 0 evicted), in the right temporal order (oldest-remaining first).
        let mut cs = ConvStreaming::<4, 2, 2>::new();
        cs.push_row([1.0, 2.0, 3.0, 4.0]); // evicted
        cs.push_row([5.0, 6.0, 7.0, 8.0]);
        cs.push_row([9.0, 10.0, 11.0, 12.0]);
        let kernel = [[1.0, 0.0], [0.0, 1.0]];
        // row0 = [5,6,7,8] (oldest remaining), row1 = [9,10,11,12] (newest)
        // res[l] = row0[l] + row1[l+1]
        let res: [Scalar; 3] = cs.conv2d(&kernel);
        assert_eq!(res, [5.0 + 10.0, 6.0 + 11.0, 7.0 + 12.0]);
    }

    /// Load-ramp test: streams real 720p rows into `ConvStreaming` at
    /// increasing simulated sensor line rates (`fps` ascending, so the
    /// inter-row budget `1 / (fps * H)` shrinks), pacing each row with a real
    /// `sleep()` rather than just measuring compute time — so this also
    /// picks up OS scheduling jitter, not just raw throughput. Every
    /// resolution below 720p already fits comfortably in a full-frame
    /// buffer, so there's nothing to observe there; 720p (1280 wide) is the
    /// case the line-buffer design (RAM in O(KH*W), not O(H*W)) actually
    /// exists for.
    ///
    /// Ignored by default: needs `tests/fixtures/sequence.npy` locally
    /// (280x3x720x1280, 3.1 GB, gitignored — not present in CI).
    /// `cargo test -- --ignored --nocapture`
    #[cfg(feature = "std")]
    #[test]
    #[ignore = "needs tests/fixtures/sequence.npy locally (3.1 GB, gitignored)"]
    fn conv_streaming_720p_load_ramp() {
        use crate::io::load_inputs::grayscale_row;
        use crate::io::npy::read_npy;
        use std::path::Path;
        use std::time::{Duration, Instant};

        const W: usize = 1280;
        const H: usize = 720;
        const KH: usize = 3;
        const KW: usize = 3;
        const W_OUT: usize = W - KW + 1;

        let arr = read_npy(Path::new("tests/fixtures/sequence.npy"))
            .expect("tests/fixtures/sequence.npy not found — this test needs the local fixture");
        assert_eq!(arr.shape, std::vec![280, 3, 720, 1280]);

        // Uniform 3x3 box blur — not the point of this test, just something
        // real for conv2d to spend time on.
        let kernel = [[1.0 / 9.0; KW]; KH];

        for &fps in &[1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 240.0] {
            let row_interval = Duration::from_secs_f64(1.0 / (fps * H as f64));
            let mut cs = ConvStreaming::<W, KH, KW>::new();
            let start = Instant::now();
            let mut max_lag = Duration::ZERO;

            for row in 0..H {
                let due = row_interval * (row as u32 + 1);
                let elapsed = start.elapsed();
                match due.checked_sub(elapsed) {
                    Some(remaining) => std::thread::sleep(remaining),
                    None => max_lag = max_lag.max(elapsed - due),
                }

                let px_row = grayscale_row::<W>(&arr, 0, row);
                cs.push_row(px_row);
                if cs.ready_to_compute() {
                    let _out: [Scalar; W_OUT] = cs.conv2d(&kernel);
                }
            }

            println!(
                "fps={fps:>5.0} | row budget={row_interval:>9.2?} | max lag={max_lag:>9.2?} | {}",
                if max_lag.is_zero() { "OK" } else { "EN RETARD" }
            );
        }
    }
}
