use crate::io::npy::{read_npy, write_npy, NpyArray};
use crate::linalg::{tensordot_3, Tensor4DBoxed};
use std::fs::{self, DirEntry};
use std::path::Path;
use std::string::{String, ToString};
use std::vec::Vec;
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

/// Un point de mesure du sweep : charge le couple (vidéo, banc de filtres),
/// contracte, écrit le résultat.
///
/// Les tenseurs sont `Tensor4DBoxed` : à 1x3x720x720 l'entrée fait 6 Mo et la
/// sortie 4 Mo, ce que la pile ne porte pas. Les shapes restent des const
/// generics — seul le lieu de stockage change, pas la vérification statique.
///
/// Les `NUMEL` et les `H_OUT`/`W_OUT` ne sont pas dérivables des autres
/// paramètres sans `generic_const_exprs`, d'où leur passage explicite ; ils sont
/// vérifiés à la compilation par `Tensor4D::new` et à l'exécution par
/// `im2col_view`.
macro_rules! bench_case {
    (
        $vid:expr, $fil:expr, $key:expr,
        video: [$n:literal, $c:literal, $h:literal, $w:literal] = $numel_x:literal,
        filters: [$k:literal, $kh:literal, $kw:literal] = $numel_f:literal,
        output: [$h_out:literal, $w_out:literal] = $numel_y:literal,
        stride: $stride:literal $(,)?
    ) => {{
        let mut vid_tensor = Tensor4DBoxed::<$n, $c, $h, $w, $numel_x>::new();
        let mut fil_tensor = Tensor4DBoxed::<$k, $c, $kh, $kw, $numel_f>::new();
        vid_tensor
            .load_vec($vid.data)
            .unwrap_or_else(|_| panic!("video dimensions unexpected"));
        fil_tensor
            .load_vec($fil.data)
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
            // --- variation de H/W ---
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

            // --- variation de N (batch) ---
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

            // --- variation de C (canaux d'entrée) ---
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

            // --- variation de K (nombre de filtres) ---
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
