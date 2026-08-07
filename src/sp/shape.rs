//! `conv_shape!` — derives every size `im2col_view`/`tensordot_3`/
//! `cross_correlate2d` need (`H_OUT`, `W_OUT`, `NUMEL_X`, `NUMEL_F`, `NUMEL_Y`)
//! from the handful of numbers that actually change when you swap
//! resolutions. Without it, changing one resolution means recomputing and
//! retyping those constants by hand at every `Tensor4D<...>` call site that
//! touches the frame, the im2col view and the output — this macro shrinks
//! that to one call site.

/// Generates a module of `usize` constants describing a convolution/
/// cross-correlation shape: `H_OUT`, `W_OUT` (output spatial size, stride
/// `stride`, no padding — same formula `im2col_view` checks internally) and
/// `NUMEL_X`/`NUMEL_F`/`NUMEL_Y`, the flat element counts `Tensor4D::<.., NUMEL>`
/// and `tensordot_3`/`cross_correlate2d` expect for the input sequence, the
/// filter bank and the output.
///
/// `N` (batch), `C` (channels) and `K` (filter count) default to `1` when
/// omitted — the common case of exploring a single-frame, single-channel
/// resolution with one filter.
///
/// ```
/// use ferrite::conv_shape;
///
/// // Shorthand: N = C = K = 1.
/// conv_shape!(pub frame96, H = 96, W = 96, KH = 3, KW = 3, stride = 1);
///
/// assert_eq!(frame96::H_OUT, 94);
/// assert_eq!(frame96::W_OUT, 94);
/// assert_eq!(frame96::NUMEL_X, 9216);
/// assert_eq!(frame96::NUMEL_F, 9);
/// assert_eq!(frame96::NUMEL_Y, 8836);
///
/// // Full form: RGB frame, a bank of 4 filters.
/// conv_shape!(pub rgb64, N = 1, C = 3, H = 64, W = 64, K = 4, KH = 3, KW = 3, stride = 1);
///
/// assert_eq!(rgb64::H_OUT, 62);
/// assert_eq!(rgb64::NUMEL_X, 12288);
/// assert_eq!(rgb64::NUMEL_F, 108);
/// assert_eq!(rgb64::NUMEL_Y, 15376);
/// ```
///
/// The generated constants plug straight into the existing types — no new
/// abstraction to learn:
///
/// ```
/// use ferrite::conv_shape;
/// use ferrite::linalg::{tensordot_3, Tensor4D};
///
/// conv_shape!(frame96, H = 96, W = 96, KH = 3, KW = 3, stride = 1);
///
/// let frame = Tensor4D::<{ frame96::N }, { frame96::C }, { frame96::H }, { frame96::W }, { frame96::NUMEL_X }>::new([0.0; frame96::NUMEL_X]);
/// let filters = Tensor4D::<{ frame96::K }, { frame96::C }, { frame96::KH }, { frame96::KW }, { frame96::NUMEL_F }>::new([0.0; frame96::NUMEL_F]);
///
/// let _out: Tensor4D<{ frame96::N }, { frame96::H_OUT }, { frame96::W_OUT }, { frame96::K }, { frame96::NUMEL_Y }> =
///     tensordot_3(&frame.im2col_view::<{ frame96::H_OUT }, { frame96::W_OUT }, { frame96::KH }, { frame96::KW }>(frame96::STRIDE), &filters);
/// ```
#[macro_export]
macro_rules! conv_shape {
    (
        $vis:vis $name:ident,
        N = $n:expr, C = $c:expr, H = $h:expr, W = $w:expr, K = $k:expr,
        KH = $kh:expr, KW = $kw:expr, stride = $s:expr
    ) => {
        $vis mod $name {
            pub const N: usize = $n;
            pub const C: usize = $c;
            pub const H: usize = $h;
            pub const W: usize = $w;
            pub const K: usize = $k;
            pub const KH: usize = $kh;
            pub const KW: usize = $kw;
            pub const STRIDE: usize = $s;
            pub const H_OUT: usize = (H - KH) / STRIDE + 1;
            pub const W_OUT: usize = (W - KW) / STRIDE + 1;
            pub const NUMEL_X: usize = N * C * H * W;
            pub const NUMEL_F: usize = K * C * KH * KW;
            pub const NUMEL_Y: usize = N * H_OUT * W_OUT * K;
        }
    };
    (
        $vis:vis $name:ident,
        H = $h:expr, W = $w:expr, KH = $kh:expr, KW = $kw:expr, stride = $s:expr
    ) => {
        $crate::conv_shape!(
            $vis $name,
            N = 1, C = 1, H = $h, W = $w, K = 1, KH = $kh, KW = $kw, stride = $s
        );
    };
}
