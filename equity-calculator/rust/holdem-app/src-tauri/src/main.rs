// Prevents additional console window on Windows in release
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod commands;

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .invoke_handler(tauri::generate_handler![
            commands::calculate_equity,
            commands::analyze_draws,
            commands::get_canonical_hands,
            commands::parse_cards,
            commands::evaluate_hand,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
