"""Generate temporary icons for Tauri build."""

from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("Installing Pillow...")
    import subprocess
    subprocess.run(["pip", "install", "Pillow"], check=True)
    from PIL import Image, ImageDraw, ImageFont


def create_icon(size: int) -> Image.Image:
    """Create a simple poker-themed icon with spade symbol."""
    # Create image with blue background (matching --primary color)
    img = Image.new("RGBA", (size, size), (59, 130, 246, 255))  # #3B82F6
    draw = ImageDraw.Draw(img)

    # Draw a white spade symbol
    center_x, center_y = size // 2, size // 2
    scale = size / 128  # Scale factor based on 128px reference

    # Spade shape (simplified polygon)
    spade_points = [
        (center_x, int(center_y - 40 * scale)),  # Top
        (int(center_x + 35 * scale), int(center_y + 10 * scale)),  # Right
        (int(center_x + 20 * scale), int(center_y + 30 * scale)),  # Right bottom
        (int(center_x + 10 * scale), int(center_y + 45 * scale)),  # Right stem top
        (int(center_x + 15 * scale), int(center_y + 55 * scale)),  # Right stem bottom
        (int(center_x - 15 * scale), int(center_y + 55 * scale)),  # Left stem bottom
        (int(center_x - 10 * scale), int(center_y + 45 * scale)),  # Left stem top
        (int(center_x - 20 * scale), int(center_y + 30 * scale)),  # Left bottom
        (int(center_x - 35 * scale), int(center_y + 10 * scale)),  # Left
    ]

    draw.polygon(spade_points, fill=(255, 255, 255, 255))

    # Draw the top arc of the spade (two circles)
    arc_radius = int(25 * scale)
    left_arc_center = (int(center_x - 18 * scale), int(center_y - 5 * scale))
    right_arc_center = (int(center_x + 18 * scale), int(center_y - 5 * scale))

    draw.ellipse(
        [
            left_arc_center[0] - arc_radius,
            left_arc_center[1] - arc_radius,
            left_arc_center[0] + arc_radius,
            left_arc_center[1] + arc_radius,
        ],
        fill=(255, 255, 255, 255),
    )
    draw.ellipse(
        [
            right_arc_center[0] - arc_radius,
            right_arc_center[1] - arc_radius,
            right_arc_center[0] + arc_radius,
            right_arc_center[1] + arc_radius,
        ],
        fill=(255, 255, 255, 255),
    )

    return img


def main():
    icons_dir = Path(__file__).parent.parent / "icons"
    icons_dir.mkdir(exist_ok=True)

    # Generate PNG icons
    sizes = [32, 128, 256]
    for size in sizes:
        img = create_icon(size)
        if size == 256:
            filename = "128x128@2x.png"
        else:
            filename = f"{size}x{size}.png"
        img.save(icons_dir / filename)
        print(f"Generated {filename}")

    # Generate ICO (Windows) - contains multiple sizes
    ico_sizes = [16, 32, 48, 64, 128, 256]
    ico_images = [create_icon(s) for s in ico_sizes]
    # Save as ICO with all sizes
    ico_images[-1].save(
        icons_dir / "icon.ico",
        format="ICO",
        append_images=ico_images[:-1],
    )
    print("Generated icon.ico")

    # Generate ICNS (macOS) - just save as PNG, Tauri will handle conversion
    # Actually, let's create a placeholder icns by copying the 128x128 PNG
    img_128 = create_icon(128)
    img_128.save(icons_dir / "icon.png")
    print("Generated icon.png (for macOS)")

    print("\nAll icons generated successfully!")


if __name__ == "__main__":
    main()
