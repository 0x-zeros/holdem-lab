//! Utility functions for WASM module initialization.

/// Set panic hook for better error messages in browser console.
pub fn set_panic_hook() {
    // Always set the hook since we include the dependency unconditionally
    console_error_panic_hook::set_once();
}
