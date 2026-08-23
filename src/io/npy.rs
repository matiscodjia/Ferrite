use regex::Regex;
#[cfg(feature = "std")]
use std::fs::File;
use std::io::Read;
use std::string::{String, ToString};
use std::vec::Vec;

#[derive(Debug)]
pub struct NpyArray {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

pub fn read_npy(path: &std::path::Path) -> Result<NpyArray, std::io::Error> {
    let mut file = File::open(path)?;
    // 1. read the 6 magic bytes, check they equal \x93NUMPY
    let mut magic = [0u8; 6];
    file.read_exact(&mut magic)?;
    if &magic != b"\x93NUMPY" {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "Invalid magic field",
        ));
    }
    // 2. read version (2 bytes) - fine to ignore if 1.0, error otherwise
    let mut version = [0u8; 2];
    file.read_exact(&mut version)?;
    let (major, _) = (version[0], version[1]);
    if major != 1 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "Invalid version field",
        ));
    }
    // 3. read the header length (2 bytes, little-endian u16)
    let mut header_len_bytes = [0u8; 2];
    file.read_exact(&mut header_len_bytes)?;
    let header_len: usize = u16::from_le_bytes(header_len_bytes) as usize;
    let mut header_buffer = vec![0u8; header_len];

    // 4. read that many bytes, interpret them as an ASCII string
    file.read_exact(&mut header_buffer)?;
    let header_result = String::from_utf8(header_buffer);

    let header_str = match header_result {
        Ok(s) => s,
        Err(_) => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Invalid Utf-8",
            ))
        }
    };

    // 5. extract descr, fortran_order, shape from it (plain text parsing,
    //    no real Python parser needed, just look for substrings)

    let descr_re = Regex::new(r"'descr': '([^']+)'").unwrap();
    let fortran_re = Regex::new(r"'fortran_order': (True|False)").unwrap();
    let shape_re = Regex::new(r"'shape': \(([^)]+)\)").unwrap();

    let descr = descr_re
        .captures(&header_str)
        .and_then(|c| c.get(1))
        .map(|m| m.as_str());
    let fortran = fortran_re
        .captures(&header_str)
        .and_then(|c| c.get(1))
        .map(|m| m.as_str() == "True");
    let shape_result = shape_re
        .captures(&header_str)
        .and_then(|c| c.get(1))
        .map(|m| m.as_str());
    // 6. check descr == "<f4" and fortran_order == False, else Err
    if descr != Some("<f4") {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "bad descr ",
        ));
    }
    if fortran != Some(false) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "Fortran order incompatibility ",
        ));
    }
    let shape_str = match shape_result {
        Some(s) => s.to_string(),
        None => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "bad shape",
            ))
        }
    };
    let shape: Vec<usize> = shape_str
        .split(',')
        .map(|s| s.trim().parse::<usize>().unwrap())
        .collect();

    // 7. read the rest of the file, reinterpret it as little-endian f32
    let mut data = Vec::new();
    file.read_to_end(&mut data)?;
    let floats: Vec<f32> = data
        .chunks(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();

    Ok(NpyArray {
        shape: shape,
        data: floats,
    })
}

pub fn write_npy(
    path: &std::path::Path,
    shape: &[usize],
    data: &[f32],
) -> Result<(), std::io::Error> {
    use std::io::Write;

    let mut file = std::fs::File::create(path)?;

    // 1. Magic
    file.write_all(b"\x93NUMPY")?;

    // 2. Version
    file.write_all(&[1u8, 0u8])?;

    // 3. Header dict (WITHOUT \n)
    let shape_str = shape
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    let header_dict = format!(
        "{{'descr': '<f4', 'fortran_order': False, 'shape': ({}), }}",
        shape_str
    );

    // 4. Compute padding to align on 64 bytes
    // Total = 10 (magic + version + len) + header_len + padding + 1 (\n) = 64k
    let header_len = header_dict.len();
    let total_with_newline = header_len + 1;
    let aligned = (((10 + total_with_newline + 63) / 64) * 64) - 10;
    let padding = aligned - header_len - 1;

    // 5. Write header length
    file.write_all(&(aligned as u16).to_le_bytes())?;

    // 6. Write header dict + padding + \n (in that order!)
    file.write_all(header_dict.as_bytes())?;
    file.write_all(&vec![b' '; padding])?;
    file.write_all(b"\n")?;

    // 7. Write data (f32 little-endian)
    for &val in data {
        file.write_all(&val.to_le_bytes())?;
    }

    Ok(())
}
