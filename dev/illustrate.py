from typing import Literal
import colorsys
import numpy as np


def illustrate(
    tokens_and_values: list[tuple[str, float]],
    gradient: Literal[
        "one-dark",
        "one-dark-simple",
        "fire",
        "ocean",
        "forest",
        "sunset",
        "monochrome",
        "plasma",
    ] = "one-dark",
) -> str:
    """
    Returns a renderable string concatenating the tokens colored by their values in an aesthetically pleasing way.

    Args:
        tokens_and_values: A list of tuples of token, value pairs. The tokens are word or subword language model tokens.
        The values are floats and will be normalized with zero mean and unit variance.
        gradient: Color gradient to use. Options: "one-dark", "one-dark-simple", "fire", "ocean", "forest", "sunset", "monochrome", "plasma"

    Returns:
        A renderable string with colored tokens based on their normalized values.
    """
    if not tokens_and_values:
        return ""

    # Extract tokens and values
    tokens = [t[0] for t in tokens_and_values]
    values = np.array([t[1] for t in tokens_and_values])

    # Normalize values to zero mean and unit variance
    if len(values) > 1:
        mean = values.mean()
        std = values.std()
        normalized_values = (values - mean) / std if std > 0 else values - mean
    else:
        normalized_values = values

    # Define gradient color stops in HSL space for smooth interpolation
    gradients = {
        "one-dark": [
            (0.0, (355, 0.65, 0.60)),  # Red (#e06c75)
            (0.25, (20, 0.50, 0.58)),  # Orange (#d19a66)
            (0.5, (219, 0.14, 0.71)),  # Gray (#abb2bf)
            (0.75, (95, 0.35, 0.58)),  # Green (#98c379)
            (1.0, (207, 0.82, 0.66)),  # Blue (#61afef)
        ],
        "one-dark-simple": [
            (0.0, (355, 0.75, 0.45)),  # Darker red
            (0.35, (10, 0.30, 0.60)),  # Subtle warm transition
            (0.5, (220, 0.08, 0.70)),  # Nearly neutral gray
            (0.65, (207, 0.30, 0.68)),  # Subtle blue transition
            (1.0, (207, 0.85, 0.55)),  # Darker blue
        ],
        "fire": [
            (0.0, (0, 0.70, 0.30)),  # Dark red
            (0.3, (355, 0.65, 0.60)),  # Red
            (0.5, (20, 0.60, 0.58)),  # Orange
            (0.7, (40, 0.70, 0.65)),  # Yellow-orange
            (1.0, (50, 0.90, 0.75)),  # Bright yellow
        ],
        "ocean": [
            (0.0, (200, 0.60, 0.25)),  # Dark blue
            (0.3, (207, 0.82, 0.66)),  # Blue
            (0.5, (187, 0.47, 0.66)),  # Cyan
            (0.7, (170, 0.40, 0.65)),  # Teal
            (1.0, (160, 0.35, 0.70)),  # Light teal
        ],
        "forest": [
            (0.0, (80, 0.40, 0.25)),  # Dark green
            (0.3, (95, 0.35, 0.58)),  # Green
            (0.5, (120, 0.30, 0.55)),  # True green
            (0.7, (150, 0.35, 0.60)),  # Teal-green
            (1.0, (170, 0.40, 0.70)),  # Light teal
        ],
        "sunset": [
            (0.0, (286, 0.60, 0.65)),  # Purple
            (0.3, (340, 0.55, 0.60)),  # Pink
            (0.5, (20, 0.60, 0.58)),  # Orange
            (0.7, (40, 0.70, 0.65)),  # Yellow-orange
            (1.0, (50, 0.80, 0.75)),  # Light yellow
        ],
        "monochrome": [
            (0.0, (220, 0.13, 0.25)),  # Dark gray
            (0.3, (220, 0.13, 0.40)),  # Medium-dark gray
            (0.5, (219, 0.14, 0.71)),  # Gray
            (0.7, (220, 0.10, 0.85)),  # Light gray
            (1.0, (220, 0.05, 0.95)),  # Near white
        ],
        "plasma": [
            (0.0, (270, 0.70, 0.50)),  # Deep purple
            (0.25, (320, 0.60, 0.60)),  # Magenta
            (0.5, (355, 0.65, 0.60)),  # Red
            (0.75, (20, 0.70, 0.65)),  # Orange
            (1.0, (50, 0.90, 0.80)),  # Bright yellow
        ],
    }

    def interpolate_hsl(val, gradient_stops, gradient_name):
        """Interpolate HSL color based on normalized value (-3 to 3 range)"""
        # Map from [-3, 3] to [0, 1]
        t = np.clip((val + 3) / 6, 0, 1)

        # Apply non-linear transformation for one-dark-simple
        if gradient_name == "one-dark-simple":
            t = 0.5 * (2 * t) ** 2.5 if t < 0.5 else 1 - 0.5 * (2 * (1 - t)) ** 2.5

        # Find the two stops to interpolate between
        for i in range(len(gradient_stops) - 1):
            t1, color1 = gradient_stops[i]
            t2, color2 = gradient_stops[i + 1]

            if t1 <= t <= t2:
                # Interpolate between these two stops
                local_t = (t - t1) / (t2 - t1)
                h1, s1, l1 = color1
                h2, s2, l2 = color2

                # Interpolate hue (handling wraparound)
                h_diff = h2 - h1
                if h_diff > 180:
                    h_diff -= 360
                elif h_diff < -180:
                    h_diff += 360

                h = (h1 + h_diff * local_t) % 360
                s = s1 + (s2 - s1) * local_t
                l = l1 + (l2 - l1) * local_t

                # Convert HSL to RGB
                r, g, b = colorsys.hls_to_rgb(h / 360, l, s)
                r, g, b = int(r * 255), int(g * 255), int(b * 255)
                return f"\033[38;2;{r};{g};{b}m"

        # Fallback to last color
        _, (h, s, l) = gradient_stops[-1]
        r, g, b = colorsys.hls_to_rgb(h / 360, l, s)
        r, g, b = int(r * 255), int(g * 255), int(b * 255)
        return f"\033[38;2;{r};{g};{b}m"

    # Get the selected gradient
    gradient_stops = gradients[gradient]

    # Build the output string
    result = []

    # Add header with value statistics
    stats = f"Stats: μ={values.mean():6.3f}, σ={values.std() if len(values) > 1 else 0:6.3f}, min={values.min():6.3f}, max={values.max():6.3f}"
    box_width = 63
    padded_stats = stats.center(box_width - 4)

    result.append(f"\033[90m╔{'═' * (box_width - 2)}╗\033[0m\n")
    result.append(f"\033[90m║ {padded_stats} ║\033[0m\n")
    result.append(f"\033[90m╚{'═' * (box_width - 2)}╝\033[0m\n\n")

    # Add colored tokens
    for token, norm_val in zip(tokens, normalized_values):
        color = interpolate_hsl(norm_val, gradient_stops, gradient)
        result.append(f"{color}{token}\033[0m")
    result.append("\n\n")

    # Add value distribution bar chart
    result.append("Value Distribution:\n")
    max_token_len = max(len(token) for token in tokens)

    for token, orig_val, norm_val in zip(tokens, values, normalized_values):
        # Build the bar
        bar_width = 20
        bar_pos = np.clip(int((norm_val + 3) / 6 * bar_width), 0, bar_width - 1)
        bar = "".join(
            (
                "│"
                if i == bar_width // 2
                else (
                    f"{interpolate_hsl(norm_val, gradient_stops, gradient)}█\033[0m"
                    if i == bar_pos
                    else "─"
                )
            )
            for i in range(bar_width)
        )

        # Format the line
        color = interpolate_hsl(norm_val, gradient_stops, gradient)
        result.append(
            f"  {color}{token.ljust(max_token_len)}\033[0m  [{bar}]  {orig_val:7.3f}\n"
        )

    # Add gradient legend
    result.append(f"\n\033[90m{'─' * 60}\033[0m\n")
    result.append(f"Gradient ({gradient}): ")

    for i in range(20):
        t = (i / 19) * 6 - 3  # Map to [-3, 3]
        color = interpolate_hsl(t, gradient_stops, gradient)
        result.append(f"{color}█\033[0m")

    result.append("\033[0m")
    return "".join(result)
