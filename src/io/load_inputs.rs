use crate::io::npy::{read_npy, write_npy, NpyArray};
use crate::linalg::{tensordot_3, Tensor4DBoxed};
use crate::scalar::Scalar;
use std::fs::{self, DirEntry};
use std::path::Path;
use std::string::{String, ToString};
use std::vec::Vec;

/// Extracts one row of `frame` from an (N, C, H, W) npy array as grayscale,
/// via the standard luminance formula (0.299 R + 0.587 G + 0.114 B) when
/// `C == 3`, or a straight copy when `C == 1`. Reads only the `W` pixels of
/// that row from each channel plane — never materialises a whole grayscale
/// frame, so a caller streaming rows into `ConvStreaming` pays no extra
/// memory for the conversion.
///
/// # Panics
/// Panics if `arr.shape` isn't 4-dimensional, `frame`/`row` are out of
/// range, the array's width doesn't match `W`, or `C` isn't 1 or 3.
pub fn grayscale_row<const W: usize>(arr: &NpyArray, frame: usize, row: usize) -> [Scalar; W] {
    assert_eq!(arr.shape.len(), 4, "expected an (N, C, H, W) array");
    let (n, c, h, w) = (arr.shape[0], arr.shape[1], arr.shape[2], arr.shape[3]);
    assert!(frame < n && row < h && w == W);
    assert!(
        c == 1 || c == 3,
        "expected 1 (grayscale) or 3 (RGB) channels, got {c}"
    );

    let plane = h * w;
    let frame_base = frame * c * plane;
    let row_base = row * w;

    let mut out = [0.0; W];
    if c == 1 {
        for x in 0..W {
            out[x] = arr.data[frame_base + row_base + x] as Scalar;
        }
    } else {
        let (r_base, g_base, b_base) = (frame_base, frame_base + plane, frame_base + 2 * plane);
        for x in 0..W {
            let r = arr.data[r_base + row_base + x];
            let g = arr.data[g_base + row_base + x];
            let b = arr.data[b_base + row_base + x];
            out[x] = (0.299 * r + 0.587 * g + 0.114 * b) as Scalar;
        }
    }
    out
}

pub fn read_files(path: &str) -> std::io::Result<Vec<DirEntry>> {
    let mut files_path = Vec::new();
    for entree in fs::read_dir(path)? {
        let entree = entree?;
        files_path.push(entree);
    }
    Ok(files_path)
}
pub fn npy_to_arrays(entries: Vec<DirEntry>) -> std::io::Result<Vec<(String, NpyArray, NpyArray)>> {
    let mut couples = Vec::new();
    for vid_entry in &entries {
        let vid_file = vid_entry.path();
        let vid_name = match vid_file.to_str() {
            Some(name) => name.rsplit("/").next().unwrap_or(name),
            None => continue,
        };
        if !vid_name.starts_with("vid") {
            continue;
        }
        let video_key = match vid_name.strip_suffix(".npy") {
            Some(k) => &k[4..],
            None => continue,
        };
        for fil_entry in &entries {
            let fil_file = fil_entry.path();
            let fil_name = match fil_file.to_str() {
                Some(name) => name.rsplit("/").next().unwrap_or(name),
                None => continue,
            };
            if !fil_name.starts_with("fil") {
                continue;
            }
            let fil_key = match fil_name.strip_suffix(".npy") {
                Some(k) => &k[4..],
                None => continue,
            };
            if fil_key == video_key {
                let vid = read_npy(&vid_file)?;
                let fil = read_npy(&fil_file)?;
                println!(" {:?}    {:?}", &vid.shape, &fil.shape);
                couples.push((video_key.to_string(), vid, fil));
            }
        }
    }
    Ok(couples)
}

/// One measurement point in the sweep: loads the (video, filter bank) pair,
/// contracts, writes the result.
///
/// The tensors are `Tensor4DBoxed`: at 1x3x720x720 the input is 6 MB and
/// the output 4 MB, which the stack cannot carry. Shapes stay const
/// generics — only the storage location changes, not the static
/// verification.
///
/// `NUMEL` and `H_OUT`/`W_OUT` aren't derivable from the other parameters
/// without `generic_const_exprs`, hence passing them explicitly; they are
/// checked at compile time by `Tensor4D::new` and at runtime by
/// `im2col_view`.
macro_rules! bench_case {
    (
        $vid:expr, $fil:expr, $key:expr,
        video: [$n:literal, $c:literal, $h:literal, $w:literal] = $numel_x:literal,
        filters: [$k:literal, $kh:literal, $kw:literal] = $numel_f:literal,
        output: [$h_out:literal, $w_out:literal] = $numel_y:literal,
        stride: $stride:literal $(,)?
    ) => {{
        let vid_tensor = Tensor4DBoxed::<$n, $c, $h, $w, $numel_x>::from_vec($vid.data)
            .unwrap_or_else(|_| panic!("video dimensions unexpected"));
        let fil_tensor = Tensor4DBoxed::<$k, $c, $kh, $kw, $numel_f>::from_vec($fil.data)
            .unwrap_or_else(|_| panic!("filter dimensions unexpected"));

        let result: Tensor4DBoxed<$n, $h_out, $w_out, $k, $numel_y> = tensordot_3(
            &vid_tensor.im2col_view::<$h_out, $w_out, $kh, $kw>($stride),
            &fil_tensor,
        );

        let _ = write_npy(
            Path::new(&format!("output_{}.npy", $key)),
            result.get_shape(),
            result.get_data(),
        );
    }};
}

pub fn compute_cross_corr_output_npy(couples: Vec<(String, NpyArray, NpyArray)>) -> () {
    for (key, vid, fil) in couples {
        match (vid.shape.as_slice(), fil.shape.as_slice()) {
            // --- H/W variation ---
            ([1, 3, 32, 32], [2, 3, 3, 3]) => bench_case!(
                vid, fil, key,
                video: [1, 3, 32, 32] = 3072,
                filters: [2, 3, 3] = 54,
                output: [30, 30] = 1800,
                stride: 1,
            ),
            ([1, 3, 64, 64], [2, 3, 3, 3]) => bench_case!(
                vid, fil, key,
                video: [1, 3, 64, 64] = 12288,
                filters: [2, 3, 3] = 54,
                output: [62, 62] = 7688,
                stride: 1,
            ),
            ([1, 3, 128, 128], [2, 3, 3, 3]) => bench_case!(
                vid, fil, key,
                video: [1, 3, 128, 128] = 49152,
                filters: [2, 3, 3] = 54,
                output: [126, 126] = 31752,
                stride: 1,
            ),
            ([1, 3, 256, 256], [2, 3, 3, 3]) => bench_case!(
                vid, fil, key,
                video: [1, 3, 256, 256] = 196608,
                filters: [2, 3, 3] = 54,
                output: [254, 254] = 129032,
                stride: 1,
            ),
            ([1, 3, 720, 720], [2, 3, 3, 3]) => bench_case!(
                vid, fil, key,
                video: [1, 3, 720, 720] = 1555200,
                filters: [2, 3, 3] = 54,
                output: [718, 718] = 1031048,
                stride: 1,
            ),

            // --- N (batch) variation ---
            ([2, 3, 128, 128], [2, 3, 3, 3]) => bench_case!(
                vid, fil, key,
                video: [2, 3, 128, 128] = 98304,
                filters: [2, 3, 3] = 54,
                output: [126, 126] = 63504,
                stride: 1,
            ),
            ([4, 3, 128, 128], [2, 3, 3, 3]) => bench_case!(
                vid, fil, key,
                video: [4, 3, 128, 128] = 196608,
                filters: [2, 3, 3] = 54,
                output: [126, 126] = 127008,
                stride: 1,
            ),
            ([8, 3, 128, 128], [2, 3, 3, 3]) => bench_case!(
                vid, fil, key,
                video: [8, 3, 128, 128] = 393216,
                filters: [2, 3, 3] = 54,
                output: [126, 126] = 254016,
                stride: 1,
            ),
            ([16, 3, 128, 128], [2, 3, 3, 3]) => bench_case!(
                vid, fil, key,
                video: [16, 3, 128, 128] = 786432,
                filters: [2, 3, 3] = 54,
                output: [126, 126] = 508032,
                stride: 1,
            ),
            ([32, 3, 128, 128], [2, 3, 3, 3]) => bench_case!(
                vid, fil, key,
                video: [32, 3, 128, 128] = 1572864,
                filters: [2, 3, 3] = 54,
                output: [126, 126] = 1016064,
                stride: 1,
            ),

            // --- C (input channels) variation ---
            ([1, 1, 128, 128], [2, 1, 3, 3]) => bench_case!(
                vid, fil, key,
                video: [1, 1, 128, 128] = 16384,
                filters: [2, 3, 3] = 18,
                output: [126, 126] = 31752,
                stride: 1,
            ),
            ([1, 8, 128, 128], [2, 8, 3, 3]) => bench_case!(
                vid, fil, key,
                video: [1, 8, 128, 128] = 131072,
                filters: [2, 3, 3] = 144,
                output: [126, 126] = 31752,
                stride: 1,
            ),
            ([1, 16, 128, 128], [2, 16, 3, 3]) => bench_case!(
                vid, fil, key,
                video: [1, 16, 128, 128] = 262144,
                filters: [2, 3, 3] = 288,
                output: [126, 126] = 31752,
                stride: 1,
            ),

            // --- K (filter count) variation ---
            ([1, 3, 128, 128], [1, 3, 3, 3]) => bench_case!(
                vid, fil, key,
                video: [1, 3, 128, 128] = 49152,
                filters: [1, 3, 3] = 27,
                output: [126, 126] = 15876,
                stride: 1,
            ),
            ([1, 3, 128, 128], [4, 3, 3, 3]) => bench_case!(
                vid, fil, key,
                video: [1, 3, 128, 128] = 49152,
                filters: [4, 3, 3] = 108,
                output: [126, 126] = 63504,
                stride: 1,
            ),
            ([1, 3, 128, 128], [8, 3, 3, 3]) => bench_case!(
                vid, fil, key,
                video: [1, 3, 128, 128] = 49152,
                filters: [8, 3, 3] = 216,
                output: [126, 126] = 127008,
                stride: 1,
            ),
            ([1, 3, 128, 128], [16, 3, 3, 3]) => bench_case!(
                vid, fil, key,
                video: [1, 3, 128, 128] = 49152,
                filters: [16, 3, 3] = 432,
                output: [126, 126] = 254016,
                stride: 1,
            ),

            _ => {}
        }
    }
    println!("Done !");
}
