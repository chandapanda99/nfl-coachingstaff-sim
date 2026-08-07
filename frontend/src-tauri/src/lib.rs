use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use tauri::{Manager, WebviewUrl, WebviewWindowBuilder};

const APPLICATION_DIRECTORY_NAME: &str = "nfl-coachingstaff-sim";
const LEGACY_APPLICATION_DIRECTORY_NAMES: [&str; 2] = [
    "com.aayushchanda.nfl-coachingstaff-sim",
    "NFL Virtual Coaching Staff",
];

fn available_destination(destination: &Path) -> PathBuf {
    if !destination.exists() {
        return destination.to_path_buf();
    }

    let parent = destination.parent().unwrap_or_else(|| Path::new("."));
    let name = destination
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("legacy-data");
    let mut index = 2;
    let mut candidate = parent.join(format!("{name}-2"));
    while candidate.exists() {
        index += 1;
        candidate = parent.join(format!("{name}-{index}"));
    }
    candidate
}

fn migrate_legacy_app_directories(local_data_root: &Path) -> io::Result<PathBuf> {
    let canonical_root = local_data_root.join(APPLICATION_DIRECTORY_NAME);

    for legacy_name in LEGACY_APPLICATION_DIRECTORY_NAMES {
        let legacy_root = local_data_root.join(legacy_name);
        if !legacy_root.is_dir() {
            continue;
        }

        if !canonical_root.exists() {
            fs::rename(&legacy_root, &canonical_root)?;
            continue;
        }

        let archive_root = canonical_root.join("legacy").join(legacy_name);
        for entry in fs::read_dir(&legacy_root)? {
            let entry = entry?;
            let preferred = canonical_root.join(entry.file_name());
            let destination = if preferred.exists() {
                fs::create_dir_all(&archive_root)?;
                available_destination(&archive_root.join(entry.file_name()))
            } else {
                preferred
            };
            fs::rename(entry.path(), destination)?;
        }
        fs::remove_dir(&legacy_root)?;
    }

    fs::create_dir_all(&canonical_root)?;
    Ok(canonical_root)
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .setup(|app| {
            let local_data_root = app.path().local_data_dir()?;
            let application_root = migrate_legacy_app_directories(&local_data_root)?;

            WebviewWindowBuilder::new(app, "main", WebviewUrl::App("index.html".into()))
                .title("NFL Virtual Coaching Staff")
                .inner_size(1440.0, 940.0)
                .min_inner_size(920.0, 680.0)
                .resizable(true)
                .center()
                .data_directory(application_root.join("EBWebView"))
                .build()?;
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running the NFL Virtual Coaching Staff desktop shell");
}
