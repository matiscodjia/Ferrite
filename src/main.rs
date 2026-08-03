use ferrite::io::load_inputs::{compute_cross_corr_output_npy, npy_to_arrays, read_files};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let vec = read_files("./tests/fixtures")?;
    let couples = npy_to_arrays(vec)?;
    compute_cross_corr_output_npy(couples);
    Ok(())
}
