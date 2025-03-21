from __future__ import division, print_function
import contextlib
from typing import Dict, List, Optional, Union, Tuple, Any

import numpy as np
import pandas as pd
import seaborn as sns

from rich import box
from rich.console import Console
from rich.errors import NotRenderableError
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

def _format_numeric_value(value, decimal_places, is_null_check=True):
    """Format a numeric value with specified decimal places.
    Args:
        value: The value to format
        decimal_places (int): Number of decimal places
        is_null_check (bool): Whether to check for null values
    Returns:
        str: Formatted string representation of the value
    """
    if is_null_check and pd.isna(value):
        return ""
    if isinstance(value, (int, float)):
        return f"{value:.{decimal_places}f}"
    return str(value)
def _apply_decimal_formatting(df, decimals):
    """Apply decimal formatting to dataframe columns.
    Args:
        df (pd.DataFrame): DataFrame to format
        decimals (int, dict, None): Decimal specification
    Returns:
        pd.DataFrame: Formatted DataFrame
    """
    if decimals is None:
        return df
    df_display = df.copy()
    numeric_cols = df_display.select_dtypes(include=[np.number]).columns
    if isinstance(decimals, int):
        # Apply same decimal places to all numeric columns
        for col in numeric_cols:
            df_display[col] = df_display[col].map(
                lambda x: _format_numeric_value(x, decimals)
            )
    elif isinstance(decimals, dict):
        # Apply specific decimal places to specified columns
        for col, decimal in decimals.items():
            if col in df_display.columns:
                df_display[col] = df_display[col].map(
                    lambda x: _format_numeric_value(x, decimal)
                )
    return df_display

def _get_styled_value(value, col_name, dynamic_color_cols, style):
    """Apply styling to a value based on column and value properties.
    Args:
        value: The value to style
        col_name (str): Column name
        dynamic_color_cols (list): Columns to apply dynamic coloring
        style (dict): Style configuration
    Returns:
        str or Text: Styled value
    """
    from rich.text import Text
    # Convert to string if not already
    str_value = str(value)
    # Apply dynamic coloring if column is in the list
    if col_name in dynamic_color_cols:
        with contextlib.suppress(ValueError, TypeError):
            # Handle percentage strings (e.g., "-4.81%")
            if isinstance(value, str) and "%" in value:
                # Remove % and convert to float
                num_value = float(value.replace("%", ""))
            else:
                # Regular numeric conversion
                num_value = float(value) if pd.notnull(value) else 0
            if num_value > 0:
                return Text(str_value, style=style['positive_style'])
            elif num_value < 0:
                return Text(str_value, style=style['negative_style'])
    return str_value

def _get_column_stats(df, col_name, decimals=None):
    """Generate statistics for a column based on its data type.

    Args:
        df (pd.DataFrame): DataFrame containing the column
        col_name (str): Name of the column
        decimals (int, dict, None): Decimal specification for formatting

    Returns:
        str: Formatted statistics for the column
    """
    col_data = df[col_name]

    # Get decimal places for this column if specified
    decimal_places = 2
    if isinstance(decimals, int):
        decimal_places = decimals
    elif isinstance(decimals, dict) and col_name in decimals:
        decimal_places = decimals[col_name]

    # Handle different data types
    if pd.api.types.is_numeric_dtype(col_data):
        mean = col_data.mean()
        return f"mean: {_format_numeric_value(mean, decimal_places, False)}"
    elif pd.api.types.is_categorical_dtype(col_data) or col_data.dtype == 'object':
        n_unique = col_data.nunique()
        return f"unique: {n_unique}"
    elif pd.api.types.is_datetime64_dtype(col_data):
        return f"range: {col_data.min().strftime('%Y-%m-%d')} to {col_data.max().strftime('%Y-%m-%d')}"
    else:
        return ""

def rich_display_dataframe(
    df,
    title="Dataframe",
    decimals=None,
    dynamic_color_cols=None,
    gradient_cols=None,
    style=None,
    max_rows=50,
    show_index=True
) -> None:
    """Display dataframe as table using rich library with enhanced formatting.
    Args:
        df (pd.DataFrame): dataframe to display
        title (str, optional): title of the table. Defaults to "Dataframe".
        decimals (int, dict, optional): Number of decimal places to display.
            If int, applies to all numeric columns.
            If dict, maps column names to decimal places.
        dynamic_color_cols (list, optional): List of columns to color based on value sign.
            Positive values will be green, negative values will be red.
        gradient_cols (list, optional): List of float columns to apply a color gradient.
            Values will be colored from light blue (min) to dark blue (max).
        style (dict, optional): Style configuration with keys:
            - 'header_style': Style for header (default: 'bold dim blue')
            - 'title_style': Style for title (default: 'bold blue')
            - 'border_style': Style for borders (default: 'dim')
            - 'positive_style': Style for positive values (default: 'green')
            - 'negative_style': Style for negative values (default: 'red')
            - 'min_gradient': Style for minimum gradient values (default: 'rgb(173,216,230)')
            - 'max_gradient': Style for maximum gradient values (default: 'rgb(0,0,139)')
        max_rows (int, optional): Maximum number of rows to display. Defaults to 50.
        show_index (bool, optional): Whether to display the index. Defaults to True.
    """
    from rich import print
    from rich.table import Table
    import pandas as pd
    import numpy as np
    # Default style settings
    default_style = {
        'header_style': 'bold dim blue',
        'title_style': 'bold blue',
        'border_style': 'dim',
        'positive_style': 'bold green',
        'negative_style': 'red',
        'min_gradient': 'rgb(173,216,230)',  # light blue
        'max_gradient': 'rgb(0,0,139)'       # dark blue
    }
    # Update with user-provided styles
    style = {**default_style, **(style or {})}
    # Format numeric columns with specified decimal places
    df_display = _apply_decimal_formatting(df, decimals)
    # Initialize dynamic color information
    dynamic_color_cols = dynamic_color_cols or []

    # Handle gradient columns
    if gradient_cols == 'all':
        # Get all float columns that aren't in dynamic_color_cols
        float_cols = df.select_dtypes(include=['float']).columns.tolist()
        gradient_cols = [col for col in float_cols if col not in dynamic_color_cols]
    elif gradient_cols is None or (isinstance(gradient_cols, list) and len(gradient_cols) == 0):
        gradient_cols = []

    # Prepare gradient color mapping for each gradient column
    gradient_mappings = {}
    for col in gradient_cols:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            col_min = df[col].min()
            col_max = df[col].max()
            gradient_mappings[col] = (col_min, col_max)

    # Truncate if necessary
    total_rows = len(df)
    truncated = total_rows > max_rows
    if truncated:
        df_display = df_display.iloc[:max_rows]
        title = f"{title} (showing {max_rows} of {total_rows} rows)"

    # Create rich table
    table = Table(
        title=title,
        title_style=style['title_style'],
        border_style=style['border_style'],
        header_style=style['header_style']
    )

    # Add index column if requested
    if show_index:
        index_name = str(df.index.name) if df.index.name else "Index"
        table.add_column(index_name, header_style=style['header_style'])

    # Add columns with statistics
    for col in df_display.columns:
        col_stats = _get_column_stats(df, col, decimals)
        table.add_column(str(col), header_style=style['header_style'], footer=col_stats)

    # Process rows with dynamic coloring and gradients
    for idx, row in df_display.iterrows():
        # Prepare row data including index if requested
        row_data = []
        if show_index:
            row_data.append(str(idx))

        # Add column values with appropriate styling
        for col_name, value in row.items():
            if col_name in gradient_cols and col_name in gradient_mappings:
                styled_value = _get_gradient_value(value, col_name, gradient_mappings, style)
            elif col_name in dynamic_color_cols:
                styled_value = _get_styled_value(value, col_name, dynamic_color_cols, style)
            else:
                styled_value = str(value)
            row_data.append(styled_value)

        with contextlib.suppress(NotRenderableError):
            table.add_row(*row_data)

    print(table)

def _get_gradient_value(value, col_name, gradient_mappings, style):
    """Apply gradient coloring to a value based on its position in the range.

    Args:
        value: The value to style
        col_name: Column name
        gradient_mappings: Dictionary mapping column names to (min, max) tuples
        style: Style configuration

    Returns:
        Text: Styled value with gradient color
    """
    from rich.text import Text

    # Convert to string if not already
    str_value = str(value)

    # Apply gradient coloring
    with contextlib.suppress(ValueError, TypeError):
        if pd.notnull(value):
            col_min, col_max = gradient_mappings[col_name]

            # Avoid division by zero
            if col_max == col_min:
                normalized = 1.0
            else:
                # Normalize value between 0 and 1
                normalized = (float(value) - col_min) / (col_max - col_min)

            # Use seaborn's color_palette to create a gradient
            # Extract RGB values from style strings for start and end colors
            min_rgb = [int(c) for c in style['min_gradient'].replace('rgb(', '').replace(')', '').split(',')]
            max_rgb = [int(c) for c in style['max_gradient'].replace('rgb(', '').replace(')', '').split(',')]

            # Convert RGB to hex for seaborn
            min_hex = f"#{min_rgb[0]:02x}{min_rgb[1]:02x}{min_rgb[2]:02x}"
            max_hex = f"#{max_rgb[0]:02x}{max_rgb[1]:02x}{max_rgb[2]:02x}"

            # Create a color palette with seaborn
            palette = sns.color_palette([min_hex, max_hex], as_cmap=False, n_colors=100)

            # Get the color at the normalized position
            color_idx = min(int(normalized * 99), 99)  # Ensure index is within bounds
            r, g, b = [int(c * 255) for c in palette[color_idx]]

            # Create color style for rich
            color_style = f"rgb({r},{g},{b})"
            return Text(str_value, style=color_style)

    return str_value
